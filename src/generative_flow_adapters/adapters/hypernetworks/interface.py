from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor

from generative_flow_adapters.adapters.base import Adapter


class HyperNetworkAdapterInterface(Adapter, ABC):
    """Interface for adapters that generate step-specific parameters for the base model."""

    def clear_captured_base_features(self) -> None:
        """Optional hook for variants that capture frozen-model activations."""

    @abstractmethod
    def build_hyper_input(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def clear_dynamic_parameters(self) -> None:
        raise NotImplementedError
