from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel


class Adapter(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.base_model: BaseGenerativeModel | None = None

    def attach_base_model(self, base_model: BaseGenerativeModel) -> None:
        self.base_model = base_model

    @abstractmethod
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError
