from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from generative_flow_adapters.adapters.base import Adapter


@dataclass(slots=True)
class OutputAdapterResult:
    adapter_output: Tensor
    output_kind: str = "delta"
    gate: Tensor | None = None


class OutputAdapterInterface(Adapter, ABC):
    """Common interface for adapters that modify the base model output."""

    @abstractmethod
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult | Tensor:
        raise NotImplementedError
