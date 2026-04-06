from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.conditioning.encoders import ConditionEncoder
from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel


class AdaptedModel(nn.Module):
    def __init__(
        self,
        base_model: BaseGenerativeModel,
        adapter: Adapter,
        condition_encoder: ConditionEncoder | None = None,
        pass_cond_to_base: bool = False,
        output_composition: str = "add",
        gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.condition_encoder = condition_encoder
        self.pass_cond_to_base = pass_cond_to_base
        self.output_composition = output_composition
        self.gate_bias = gate_bias
        self.adapter.attach_base_model(base_model)

    @property
    def model_type(self) -> str:
        return self.base_model.model_type

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        encoded_cond = self.condition_encoder(cond) if self.condition_encoder is not None else cond
        base_cond = encoded_cond if self.pass_cond_to_base else None
        base_output = self.base_model(x_t, t, cond=base_cond)
        adapter_cond = self._build_adapter_condition(raw_cond=cond, encoded_cond=encoded_cond)
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

    def _build_adapter_condition(self, raw_cond: object | None, encoded_cond: object | None) -> object | None:
        if raw_cond is None and encoded_cond is None:
            return None
        if isinstance(raw_cond, Mapping):
            adapter_cond = dict(raw_cond)
            if encoded_cond is not None:
                adapter_cond["embedding"] = encoded_cond
            return adapter_cond
        if encoded_cond is not None:
            return {"raw": raw_cond, "embedding": encoded_cond}
        return raw_cond
