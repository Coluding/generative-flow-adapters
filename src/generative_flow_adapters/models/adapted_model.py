from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.conditioning.encoders import ConditionEncoder
from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig
from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel


class AdaptedModel(nn.Module):
    def __init__(
        self,
        base_model: BaseGenerativeModel,
        adapter: Adapter,
        condition_encoder: ConditionEncoder | None = None,
        pass_cond_to_base: bool = False,
        condition_drop_prob: float = 0.0,
        output_composition: str = "add",
        gate_bias: float = 0.0,
        include_base_direction: bool = False,
        normalize_base_direction: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.condition_encoder = condition_encoder
        self.pass_cond_to_base = pass_cond_to_base
        self.condition_drop_prob = condition_drop_prob
        self.output_composition = output_composition
        self.gate_bias = gate_bias
        self.include_base_direction = include_base_direction
        self.normalize_base_direction = normalize_base_direction
        self.adapter.attach_base_model(base_model)

    @property
    def model_type(self) -> str:
        return self.base_model.model_type

    @property
    def prediction_type(self) -> str:
        return self.base_model.prediction_type

    @property
    def diffusion_schedule_config(self) -> DiffusionScheduleConfig | None:
        adapter_schedule = getattr(self.adapter, "diffusion_schedule_config", None)
        if adapter_schedule is not None:
            return adapter_schedule
        return self.base_model.diffusion_schedule_config

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        drop_mask = self._sample_condition_drop_mask(x_t)
        encoded_cond = self.condition_encoder(cond, drop_mask=drop_mask) if self.condition_encoder is not None else cond
        if hasattr(self.adapter, "clear_captured_base_features"):
            self.adapter.clear_captured_base_features()
        with torch.no_grad():
            base_output = self.base_model(x_t, t, cond=cond)
        base_direction = self._build_base_direction(base_output)
        adapter_cond = self._build_adapter_condition(raw_cond=cond, encoded_cond=encoded_cond, base_direction=base_direction)
        adapter_result = self.adapter(x_t, t, adapter_cond, base_output=base_output)
        return self._compose(base_output, adapter_result)

    def _compose(self, base_output: Tensor, adapter_result: Tensor | OutputAdapterResult) -> Tensor:
        if isinstance(adapter_result, Tensor):
            return base_output + adapter_result

        output_kind = adapter_result.output_kind.lower()
        composition = self.output_composition.lower()

        if composition == "add":
            if output_kind == "delta":
                return base_output + adapter_result.adapter_output
            if output_kind == "prediction":
                return base_output + adapter_result.adapter_output
        elif composition in {"replace", "adapter_only"}:
            return adapter_result.adapter_output
        elif composition in {"mask_mix", "avid_mask_mix"}:
            if output_kind != "prediction":
                raise ValueError("Mask-mix composition requires adapter outputs with output_kind='prediction'.")
            if adapter_result.gate is None:
                raise ValueError("Mask-mix composition requires an adapter gate output.")
            gate = torch.sigmoid(adapter_result.gate + self.gate_bias)
            return base_output * gate + adapter_result.adapter_output * (1.0 - gate)

        raise ValueError(f"Unsupported output composition: {self.output_composition}")

    def _build_adapter_condition(
        self,
        raw_cond: object | None,
        encoded_cond: object | None,
        base_direction: Tensor | None,
    ) -> object | None:
        if raw_cond is None and encoded_cond is None:
            return {"base_direction": base_direction} if base_direction is not None else None
        if isinstance(raw_cond, Mapping):
            adapter_cond = dict(raw_cond)
            if isinstance(encoded_cond, Tensor):
                adapter_cond["embedding"] = encoded_cond
            if base_direction is not None:
                adapter_cond["base_direction"] = base_direction
            return adapter_cond
        if isinstance(encoded_cond, Tensor):
            adapter_cond = {"raw": raw_cond, "embedding": encoded_cond}
            if base_direction is not None:
                adapter_cond["base_direction"] = base_direction
            return adapter_cond
        return raw_cond

    def _build_base_direction(self, base_output: Tensor) -> Tensor | None:
        if not self.include_base_direction:
            return None
        if base_output.dim() != 2:
            return base_output
        if not self.normalize_base_direction:
            return base_output
        return base_output / base_output.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _sample_condition_drop_mask(self, x_t: Tensor) -> Tensor | None:
        if self.condition_encoder is None or not self.training or self.condition_drop_prob <= 0.0:
            return None
        batch_size = x_t.shape[0]
        if batch_size <= 0:
            return None
        return torch.rand(batch_size, device=x_t.device) < self.condition_drop_prob
