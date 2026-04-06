from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.enabled = False
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        output = self.linear(x)
        if not self.enabled:
            return output
        return output + self.up(self.down(x)) * (self.alpha / self.rank)


@dataclass(slots=True)
class _LoRAHandle:
    parent: nn.Module
    child_name: str
    wrapped: LoRALinear


class LoRAAdapter(Adapter):
    def __init__(self, rank: int, alpha: float, target_modules: list[str]) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_modules = tuple(target_modules)
        self._handles: list[_LoRAHandle] = []

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = getattr(base_model, "module", base_model)
        self._handles = _inject_lora_layers(module, self.rank, self.alpha, self.target_modules)
        if not self._handles:
            raise ValueError("No matching linear layers found for LoRA injection. Adjust adapter.target_modules.")

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        del cond
        if self.base_model is None:
            raise RuntimeError("LoRAAdapter must be attached to a base model before use.")
        reference = base_output if base_output is not None else self.base_model(x_t, t, cond=None)
        with self._enabled():
            adapted = self.base_model(x_t, t, cond=None)
        return adapted - reference

    @contextmanager
    def _enabled(self):
        try:
            for handle in self._handles:
                handle.wrapped.enabled = True
            yield
        finally:
            for handle in self._handles:
                handle.wrapped.enabled = False


def _inject_lora_layers(module: nn.Module, rank: int, alpha: float, target_modules: tuple[str, ...]) -> list[_LoRAHandle]:
    handles: list[_LoRAHandle] = []
    for qualified_name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        leaf_name = qualified_name.rsplit(".", maxsplit=1)[-1]
        if not any(token in qualified_name or token == leaf_name for token in target_modules):
            continue
        parent, child_name = _resolve_parent(module, qualified_name)
        wrapped = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(parent, child_name, wrapped)
        handles.append(_LoRAHandle(parent=parent, child_name=child_name, wrapped=wrapped))
    return handles


def _resolve_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
