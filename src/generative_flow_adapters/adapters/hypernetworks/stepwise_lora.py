from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.common import ContextProjector, resolve_condition_embedding
from generative_flow_adapters.adapters.hypernetworks.interface import HyperNetworkAdapterInterface
from generative_flow_adapters.adapters.low_rank.common import (
    LoRAHandle,
    clear_dynamic_lora_parameters,
    inject_lora_layers,
)


class SimpleHyperLoRAAdapter(HyperNetworkAdapterInterface):
    """Lightweight hypernetwork baseline that predicts per-step LoRA weights."""

    def __init__(
        self,
        rank: int,
        alpha: float,
        target_modules: list[str],
        cond_dim: int,
        hidden_dim: int,
        input_summary_dim: int,
        use_base_output_summary: bool = False,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_modules = tuple(target_modules)
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.input_summary_dim = input_summary_dim
        self.use_base_output_summary = use_base_output_summary

        self.context = ContextProjector(cond_dim=cond_dim, hidden_dim=hidden_dim)
        self.input_summary = nn.Sequential(
            nn.Linear(input_summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_input_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._handles: list[LoRAHandle] = []
        self._module_heads = nn.ModuleDict()

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = getattr(base_model, "module", base_model)
        self._handles = inject_lora_layers(module, self.rank, self.alpha, self.target_modules)
        if not self._handles:
            raise ValueError("No matching linear layers found for hyper-LoRA injection. Adjust adapter.target_modules.")
        for handle in self._handles:
            self._module_heads[self._key(handle.qualified_name)] = _LoRAParameterHead(
                context_dim=self.hidden_dim,
                in_features=handle.wrapped.in_features,
                out_features=handle.wrapped.out_features,
                rank=self.rank,
            )

    def clear_dynamic_parameters(self) -> None:
        clear_dynamic_lora_parameters(self._handles)

    def build_hyper_input(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        context = self.context(t, resolve_condition_embedding(cond))
        summary_source = base_output if self.use_base_output_summary and base_output is not None else x_t
        pooled = _pool_reference(summary_source)
        pooled_features = self.input_summary(pooled)
        return self.hyper_input_fuse(torch.cat([context, pooled_features], dim=-1))

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        if self.base_model is None:
            raise RuntimeError("SimpleHyperLoRAAdapter must be attached to a base model before use.")

        reference = base_output if base_output is not None else self.base_model(x_t, t, cond=self._resolve_base_condition(cond))
        hyper_input = self.build_hyper_input(x_t=x_t, t=t, cond=cond, base_output=base_output)

        self.clear_dynamic_parameters()
        for handle in self._handles:
            head = self._module_heads[self._key(handle.qualified_name)]
            down, up = head(hyper_input)
            handle.wrapped.set_dynamic_parameters(down=down, up=up, alpha=self.alpha)

        with self._dynamic_enabled():
            adapted = self.base_model(x_t, t, cond=self._resolve_base_condition(cond))
        return adapted - reference

    @contextmanager
    def _dynamic_enabled(self):
        try:
            yield
        finally:
            self.clear_dynamic_parameters()

    def _resolve_base_condition(self, cond: object | None) -> object | None:
        if isinstance(cond, dict) and "embedding" in cond:
            base_cond = dict(cond)
            base_cond.pop("embedding", None)
            return base_cond
        return cond

    def _key(self, qualified_name: str) -> str:
        return qualified_name.replace(".", "__")


class _LoRAParameterHead(nn.Module):
    def __init__(self, context_dim: int, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.down = nn.Linear(context_dim, rank * in_features)
        self.up = nn.Linear(context_dim, out_features * rank)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = context.shape[0]
        down = self.down(context).view(batch_size, self.rank, self.in_features)
        up = self.up(context).view(batch_size, self.out_features, self.rank)
        return down, up


def _pool_reference(reference: Tensor) -> Tensor:
    if reference.dim() < 2:
        raise ValueError("Hypernetwork input reference must include a batch dimension.")
    if reference.dim() == 2:
        return reference
    reduce_dims = tuple(range(2, reference.dim()))
    return reference.mean(dim=reduce_dims)
