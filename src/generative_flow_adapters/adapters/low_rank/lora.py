from __future__ import annotations

from contextlib import contextmanager

from torch import Tensor

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.low_rank.common import LoRAHandle, inject_lora_layers


class LoRAAdapter(Adapter):
    def __init__(self, rank: int, alpha: float, target_modules: list[str]) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_modules = tuple(target_modules)
        self._handles: list[LoRAHandle] = []

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = getattr(base_model, "module", base_model)
        self._handles = inject_lora_layers(module, self.rank, self.alpha, self.target_modules)
        if not self._handles:
            raise ValueError("No matching linear layers found for LoRA injection. Adjust adapter.target_modules.")

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        if self.base_model is None:
            raise RuntimeError("LoRAAdapter must be attached to a base model before use.")
        reference = base_output if base_output is not None else self.base_model(x_t, t, cond=self._resolve_base_condition(cond))
        with self._enabled():
            adapted = self.base_model(x_t, t, cond=self._resolve_base_condition(cond))
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

    def _resolve_base_condition(self, cond: object | None) -> object | None:
        if isinstance(cond, dict) and "embedding" in cond:
            base_cond = dict(cond)
            base_cond.pop("embedding", None)
            return base_cond
        return cond
