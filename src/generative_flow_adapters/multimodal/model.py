"""Multi-stream adapted model.

Sibling to :class:`generative_flow_adapters.models.adapted_model.AdaptedModel`,
but predicts a *dict* of coupled output streams instead of one tensor. Owns the
frozen base (video prior), the action/video adapter, the per-modality prediction
heads, the optional per-modality video-adjustment heads (compositional), and a
fusion module that assembles the video prediction.

``forward(x_t: dict, t: dict, cond) -> dict`` — ``x_t``/``t`` are keyed by
modality name; the return dict carries one prediction per stream. Modality
streams have **no frozen prior** (the base only knows video); their heads
predict the whole target (sub-decision 1).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.output.interface import (
    OutputAdapterInterface,
    OutputAdapterResult,
)
from generative_flow_adapters.multimodal.fusion import LearnedMaskFusion, TrivialFusion
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


class MultiModalAdaptedModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        video_adapter: nn.Module,
        modality_heads: Mapping[str, nn.Module],
        modality_specs: list[OutputModalitySpec],
        *,
        condition_encoder: nn.Module | None = None,
        fusion: nn.Module | None = None,
        modality_video_adjusters: Mapping[str, nn.Module] | None = None,
        video_name: str = "video",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.video_adapter = video_adapter
        self.modality_heads = nn.ModuleDict(dict(modality_heads))
        self.modality_video_adjusters = (
            nn.ModuleDict(dict(modality_video_adjusters)) if modality_video_adjusters else None
        )
        self.condition_encoder = condition_encoder
        self.video_name = video_name
        self.modality_specs = list(modality_specs)
        self._adapter_order = [m.name for m in self.modality_specs if not m.has_frozen_prior]

        if fusion is None:
            n_contrib = 1 + (len(self._adapter_order) if self.modality_video_adjusters else 0)
            fusion = TrivialFusion() if self.modality_video_adjusters is None else LearnedMaskFusion(n_contrib)
        self.fusion = fusion

    # --- introspection mirrors AdaptedModel so the trainer can stay generic ---
    @property
    def model_type(self) -> str:
        return self.base_model.model_type

    @property
    def prediction_type(self) -> str:
        return self.base_model.prediction_type

    @property
    def diffusion_schedule_config(self):
        return getattr(self.base_model, "diffusion_schedule_config", None)

    def forward(
        self,
        x_t: Mapping[str, Tensor],
        t: Mapping[str, Tensor],
        cond: object | None = None,
    ) -> dict[str, Tensor]:
        cond_emb = self.condition_encoder(cond) if self.condition_encoder is not None else None

        video_x = x_t[self.video_name]
        video_t = t[self.video_name]
        with torch.no_grad():
            base_output = self.base_model(video_x, video_t, cond=cond)

        contributions = [self._video_adjustment(video_x, video_t, cond, cond_emb, base_output)]
        if self.modality_video_adjusters is not None:
            for name in self._adapter_order:
                contributions.append(
                    self.modality_video_adjusters[name](video_x, video_t, cond_emb)
                )

        predictions: dict[str, Tensor] = {self.video_name: self.fusion(base_output, contributions)}
        for name in self._adapter_order:
            predictions[name] = self.modality_heads[name](x_t[name], t[name], cond_emb)
        return predictions

    def _video_adjustment(
        self,
        video_x: Tensor,
        video_t: Tensor,
        cond: object | None,
        cond_emb: Tensor | None,
        base_output: Tensor,
    ) -> Tensor:
        if isinstance(self.video_adapter, OutputAdapterInterface):
            adapter_cond = dict(cond) if isinstance(cond, Mapping) else {}
            if isinstance(cond_emb, Tensor):
                adapter_cond["embedding"] = cond_emb
            result = self.video_adapter(video_x, video_t, adapter_cond, base_output=base_output)
            if isinstance(result, OutputAdapterResult):
                return result.adapter_output
            return result
        # Plain head (dummy-base substrate path).
        return self.video_adapter(video_x, video_t, cond_emb)
