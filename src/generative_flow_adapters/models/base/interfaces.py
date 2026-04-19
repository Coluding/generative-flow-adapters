from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig


class BaseGenerativeModel(nn.Module, ABC):
    def __init__(self, model_type: str, prediction_type: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.prediction_type = prediction_type

    def freeze(self) -> "BaseGenerativeModel":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    @property
    def diffusion_schedule_config(self) -> DiffusionScheduleConfig | None:
        return None

    @abstractmethod
    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        raise NotImplementedError


class ModuleBackboneWrapper(BaseGenerativeModel):
    def __init__(self, module: nn.Module, model_type: str, prediction_type: str) -> None:
        super().__init__(model_type=model_type, prediction_type=prediction_type)
        self.module = module
        signature = inspect.signature(module.forward)
        self.accepts_cond = "cond" in signature.parameters or "encoder_hidden_states" in signature.parameters

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        if self.accepts_cond:
            try:
                output = self.module(x_t, t, cond=cond)
            except TypeError:
                output = self.module(sample=x_t, timestep=t, encoder_hidden_states=cond)
        else:
            try:
                output = self.module(x_t, t)
            except TypeError:
                output = self.module(sample=x_t, timestep=t)

        if hasattr(output, "sample"):
            return output.sample
        return output


def infer_prediction_type(model_type: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    if model_type == "diffusion":
        return "noise"
    if model_type == "flow":
        return "velocity"
    raise ValueError(f"Unsupported model type: {model_type}")
