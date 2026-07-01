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
        modality_video_adapters: Mapping[str, nn.Module] | None = None,
        modality_encoders: Mapping[str, nn.Module] | None = None,
        video_readout: nn.Module | None = None,
        context_key: str = "context",
        video_name: str = "video",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.video_adapter = video_adapter
        self.modality_heads = nn.ModuleDict(dict(modality_heads))
        self.modality_video_adjusters = (
            nn.ModuleDict(dict(modality_video_adjusters)) if modality_video_adjusters else None
        )
        self.modality_video_adapters = (
            nn.ModuleDict(dict(modality_video_adapters)) if modality_video_adapters else None
        )
        self.modality_encoders = (
            nn.ModuleDict(dict(modality_encoders)) if modality_encoders else None
        )
        self.video_readout = video_readout
        self.context_key = context_key
        self.condition_encoder = condition_encoder
        self.video_name = video_name
        self.modality_specs = list(modality_specs)
        self._adapter_order = [m.name for m in self.modality_specs if not m.has_frozen_prior]

        if fusion is None and self.modality_video_adapters is None:
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

        if self.modality_video_adapters is not None:
            return self._forward_compositional(x_t, t, cond, cond_emb, video_x, video_t)

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

    def _forward_compositional(
        self,
        x_t: Mapping[str, Tensor],
        t: Mapping[str, Tensor],
        cond: object | None,
        cond_emb: Tensor | None,
        video_x: Tensor,
        video_t: Tensor,
    ) -> dict[str, Tensor]:
        """Real-backbone **compositional** path — the contribution (docs/composite (2).png).

        ``ε_video = m_{n+2}·ε_pre + m_{n+1}·ε_adj + Σ_i m_i·Δ_i`` with a learned
        mask ``m ∈ ℝ^{n+2}`` (``LearnedMaskFusion``):

        - ``ε_pre`` — frozen base prediction.
        - ``ε_adj`` — the action adapter (``video_adapter``), context = CLIP only.
        - ``Δ_i``  — one AVID adapter *per modality*, each seeing only its own
          modality tokens in ``context`` (video←m, one-adapter-per-modality, no
          modality↔modality coupling — that edge is reserved for the *fused*
          variant).

        The per-modality heads denoise each modality, conditioned on the pooled
        base video features (m←video).
        """
        with torch.no_grad():
            base_output = self.base_model(video_x, video_t, cond=cond)

        base_context = cond.get(self.context_key) if isinstance(cond, Mapping) else None

        # ε_adj — action adapter; context = CLIP only (no modality tokens).
        contributions = [
            self._adapter_prediction(
                self.video_adapter, video_x, video_t,
                self._adapter_cond(cond, cond_emb), base_output,
            )
        ]

        # Δ_i — one adapter per modality, each seeing ONLY its own tokens
        # (appended to the CLIP context). Action conditioning is shared across
        # every adapter (carried by `_adapter_cond`), not just the action one.
        for name in self._adapter_order:
            m_tokens = self.modality_encoders[name](x_t[name], t[name])
            context = (
                torch.cat([base_context, m_tokens], dim=1)
                if isinstance(base_context, Tensor)
                else m_tokens
            )
            contributions.append(
                self._adapter_prediction(
                    self.modality_video_adapters[name], video_x, video_t,
                    self._adapter_cond(cond, cond_emb, context=context), base_output,
                )
            )

        video_pred = self.fusion(base_output, contributions)  # learned mask m ∈ ℝ^{n+2}

        # m←video: fold pooled video features into the modality conditioning.
        head_cond = cond_emb
        if self.video_readout is not None:
            video_feat = self.video_readout(base_output)  # (B, cond_dim)
            if cond_emb is None:
                head_cond = video_feat
            else:
                # cond_emb may be per-frame (B, T, cond_dim) when the action
                # conditioning has a time axis; broadcast the per-sample video
                # feature across the leading (time) axes before adding.
                while video_feat.dim() < cond_emb.dim():
                    video_feat = video_feat.unsqueeze(1)
                head_cond = cond_emb + video_feat

        predictions: dict[str, Tensor] = {self.video_name: video_pred}
        for name in self._adapter_order:
            predictions[name] = self.modality_heads[name](x_t[name], t[name], head_cond)
        return predictions

    def _adapter_cond(self, cond: object | None, cond_emb: Tensor | None, *, context: Tensor | None = None) -> dict:
        """Assemble one adapter's conditioning.

        **Every** adapter is action-conditioned: the action (``cond["act"]``) and
        the rest of the batch conditioning (``fs`` etc.) are carried over by
        copying ``cond``, and the encoded ``embedding`` (which also encodes the
        action) is attached. ``context`` overrides the cross-attention tokens —
        left as the CLIP context for the action adapter, or CLIP + that modality's
        tokens for a per-modality adapter. This is the single place action
        conditioning is wired, so it is identical across all adapters.
        """
        adapter_cond = dict(cond) if isinstance(cond, Mapping) else {}
        if isinstance(cond_emb, Tensor):
            adapter_cond["embedding"] = cond_emb
        if context is not None:
            adapter_cond[self.context_key] = context
        return adapter_cond

    @staticmethod
    def _adapter_prediction(
        adapter: nn.Module, video_x: Tensor, video_t: Tensor, adapter_cond: dict, base_output: Tensor
    ) -> Tensor:
        """Run one output adapter and return its raw prediction tensor.

        The per-adapter gate (if any) is intentionally ignored — the global
        learned mask owns the blend, so each adapter contributes a single
        full-resolution stream into ``LearnedMaskFusion``.
        """
        result = adapter(video_x, video_t, adapter_cond, base_output=base_output)
        if isinstance(result, OutputAdapterResult):
            return result.adapter_output
        return result

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
