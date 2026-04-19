from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


PAPER_HYPERALIGN_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


class WrappedLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        hyper_aux_dims: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.enabled = False
        self.dynamic_alpha: float | None = None
        self.dynamic_down: Tensor | None = None
        self.dynamic_up: Tensor | None = None
        self.dynamic_hyper_down: Tensor | None = None
        self.dynamic_hyper_up: Tensor | None = None
        self.hyper_aux_dims = hyper_aux_dims

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

        if self.hyper_aux_dims is not None:
            aux_down_dim, aux_up_dim = self.hyper_aux_dims
            self.down_aux = nn.Parameter(torch.empty(linear.in_features, aux_down_dim))
            self.up_aux = nn.Parameter(torch.empty(aux_up_dim, linear.out_features))
            nn.init.normal_(self.down_aux, std=0.02)
            nn.init.zeros_(self.up_aux)
        else:
            self.register_parameter("down_aux", None)
            self.register_parameter("up_aux", None)

    @property
    def in_features(self) -> int:
        return int(self.linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.linear.out_features)

    def set_dynamic_parameters(self, down: Tensor, up: Tensor, alpha: float | None = None) -> None:
        self.dynamic_down = down
        self.dynamic_up = up
        self.dynamic_hyper_down = None
        self.dynamic_hyper_up = None
        self.dynamic_alpha = alpha

    def set_dynamic_hyper_factors(self, down_hyper: Tensor, up_hyper: Tensor, alpha: float | None = None) -> None:
        if self.hyper_aux_dims is None or self.down_aux is None or self.up_aux is None:
            raise RuntimeError("This LoRA wrapper does not expose HyperAlign auxiliary matrices.")
        aux_down_dim, aux_up_dim = self.hyper_aux_dims
        if down_hyper.dim() != 3 or down_hyper.shape[1:] != (aux_down_dim, self.rank):
            raise ValueError("Dynamic HyperAlign down factors must have shape [batch, aux_down_dim, rank].")
        if up_hyper.dim() != 3 or up_hyper.shape[1:] != (self.rank, aux_up_dim):
            raise ValueError("Dynamic HyperAlign up factors must have shape [batch, rank, aux_up_dim].")
        self.dynamic_down = None
        self.dynamic_up = None
        self.dynamic_hyper_down = down_hyper
        self.dynamic_hyper_up = up_hyper
        self.dynamic_alpha = alpha

    def clear_dynamic_parameters(self) -> None:
        self.dynamic_down = None
        self.dynamic_up = None
        self.dynamic_hyper_down = None
        self.dynamic_hyper_up = None
        self.dynamic_alpha = None

    def forward(self, x: Tensor) -> Tensor:
        output = self.linear(x)
        scale = (self.dynamic_alpha if self.dynamic_alpha is not None else self.alpha) / self.rank
        if self.dynamic_hyper_down is not None and self.dynamic_hyper_up is not None:
            if self.down_aux is None or self.up_aux is None:
                raise RuntimeError("HyperAlign dynamic factors require auxiliary matrices.")
            down = torch.einsum("ia,bar->bri", self.down_aux, self.dynamic_hyper_down)
            up = torch.einsum("ao,bra->bor", self.up_aux, self.dynamic_hyper_up)
            return output + _apply_batched_low_rank(x, down, up) * scale
        if self.dynamic_down is not None and self.dynamic_up is not None:
            return output + _apply_batched_low_rank(x, self.dynamic_down, self.dynamic_up) * scale
        if not self.enabled:
            return output
        return output + self.up(self.down(x)) * (self.alpha / self.rank)


@dataclass(slots=True)
class LoRAHandle:
    qualified_name: str
    parent: nn.Module
    child_name: str
    wrapped: WrappedLoRALinear


def inject_lora_layers(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_modules: tuple[str, ...],
    *,
    exact_match: bool = False,
    hyper_aux_dims: tuple[int, int] | None = None,
) -> list[LoRAHandle]:
    handles: list[LoRAHandle] = []
    for qualified_name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not _matches_target_module(qualified_name, target_modules, exact_match=exact_match):
            continue
        parent, child_name = _resolve_parent(module, qualified_name)
        wrapped = WrappedLoRALinear(child, rank=rank, alpha=alpha, hyper_aux_dims=hyper_aux_dims)
        setattr(parent, child_name, wrapped)
        handles.append(
            LoRAHandle(
                qualified_name=qualified_name,
                parent=parent,
                child_name=child_name,
                wrapped=wrapped,
            )
        )
    return handles


def inject_hyperalign_lora_layers(
    module: nn.Module,
    rank: int,
    alpha: float,
    aux_down_dim: int,
    aux_up_dim: int,
    target_modules: tuple[str, ...] = PAPER_HYPERALIGN_TARGET_MODULES,
) -> list[LoRAHandle]:
    return inject_lora_layers(
        module,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        exact_match=True,
        hyper_aux_dims=(aux_down_dim, aux_up_dim),
    )


def clear_dynamic_lora_parameters(handles: list[LoRAHandle]) -> None:
    for handle in handles:
        handle.wrapped.clear_dynamic_parameters()


def _resolve_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_target_module(qualified_name: str, target_modules: tuple[str, ...], *, exact_match: bool) -> bool:
    leaf_name = qualified_name.rsplit(".", maxsplit=1)[-1]
    if exact_match:
        return any(qualified_name == token or qualified_name.endswith(f".{token}") for token in target_modules)
    return any(token in qualified_name or token == leaf_name for token in target_modules)


def _apply_batched_low_rank(x: Tensor, down: Tensor, up: Tensor) -> Tensor:
    if x.dim() < 2:
        raise ValueError("WrappedLoRALinear expects inputs with an explicit batch dimension.")
    if down.dim() != 3 or up.dim() != 3:
        raise ValueError("Dynamic LoRA weights must have shape [batch, rank, in] and [batch, out, rank].")

    batch_size = x.shape[0]
    valid_batch_sizes = {1, batch_size}
    if down.shape[0] not in valid_batch_sizes and batch_size % down.shape[0] != 0:
        raise ValueError("Dynamic LoRA weight batch dimension must be 1, match the input batch size, or divide it evenly.")
    if up.shape[0] not in valid_batch_sizes and batch_size % up.shape[0] != 0:
        raise ValueError("Dynamic LoRA weight batch dimension must be 1, match the input batch size, or divide it evenly.")
    if down.shape[2] != x.shape[-1]:
        raise ValueError("Dynamic LoRA down projection does not match the wrapped layer input dimension.")
    if up.shape[2] != down.shape[1]:
        raise ValueError("Dynamic LoRA up/down projections must agree on rank.")

    x_flat = x.reshape(batch_size, -1, x.shape[-1])
    if down.shape[0] == 1 and batch_size != 1:
        down = down.expand(batch_size, -1, -1)
    elif down.shape[0] != batch_size:
        down = down.repeat_interleave(batch_size // down.shape[0], dim=0)
    if up.shape[0] == 1 and batch_size != 1:
        up = up.expand(batch_size, -1, -1)
    elif up.shape[0] != batch_size:
        up = up.repeat_interleave(batch_size // up.shape[0], dim=0)

    low_rank = torch.einsum("bsi,bri->bsr", x_flat, down)
    delta = torch.einsum("bor,bsr->bso", up, low_rank)
    return delta.reshape(*x.shape[:-1], up.shape[1])
