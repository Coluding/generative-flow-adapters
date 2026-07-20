"""Conditioning-agnostic Wan output adapter (the Wan analogue of
``DynamicCrafterOutputAdapter`` / AVID).

Wraps a small :class:`ActionWanModel` (a structural copy of the Wan DiT at
11M/34M/150M params) as the trainable delta on the frozen 1.3B base:
``prediction = base(x_t, t) + Wan21OutputAdapter(x_t, t, cond, step_level)``.

Conditioning is injected through the tiny DiT's AdaLN modulation; its head is
zero-initialised, so the delta is ~0 at init (identity composition). Like the
transformer head and the DynamiCrafter adapter, it consumes the **fused
``embedding``** from the external condition encoder (modality-agnostic — action
today, + proprio/goal/language later by changing only the encoder). The
shortcut **step size** keeps its own dedicated path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor

from generative_flow_adapters.adapters.common import resolve_condition_embedding
from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult
from generative_flow_adapters.backbones.wan.modules.action_model import ActionWanModel


class Wan21OutputAdapter(OutputAdapterInterface):
    def __init__(
        self,
        cond_dim: int,
        dim: int = 256,
        num_layers: int = 10,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        in_dim: int = 16,
        out_dim: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        condition_on_base_outputs: bool = True,
        use_step_level: bool = True,
        step_level_key: str = "step_level",
        step_level_transform: str = "log2",
        action_injection: str = "adaln",
        action_token_dim: int = 0,
        action_token_key: str = "action_seq",
        action_fallback_key: str = "action",
        action_max_len: int = 64,
        output_mask: bool = False,
        predict_full: bool = False,
    ) -> None:
        super().__init__()
        self.step_level_key = step_level_key
        self.action_token_key = action_token_key
        self.action_fallback_key = action_fallback_key
        # Cross-attention modes consume per-frame action tokens; used to reject
        # the aggregated-action fallback loudly (see forward).
        self.uses_action_tokens = action_injection in {"cross_attention", "both"}
        self.output_mask = output_mask
        self.predict_full = predict_full
        self.module = ActionWanModel(
            in_dim=in_dim,
            out_dim=out_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            cond_dim=cond_dim,
            use_step_level=use_step_level,
            step_level_transform=step_level_transform,
            condition_on_base_outputs=condition_on_base_outputs,
            action_injection=action_injection,
            action_token_dim=action_token_dim,
            action_max_len=action_max_len,
            output_mask=output_mask,
            predict_full=predict_full,
        )

    @classmethod
    def from_config(cls, wan_adapter_config_path: str, cond_dim: int, **overrides: Any) -> "Wan21OutputAdapter":
        params = _load_action_wan_params(wan_adapter_config_path)
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(cond_dim=cond_dim, **params)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        # Diffusion-forcing bases (Wan2.2 TI2V) feed a per-latent-frame timestep
        # [B, T'] (observation frames at 0, predicted frames at sigma). The
        # ActionWanModel's time/action conditioning is global (AdaLN broadcast),
        # so collapse to a per-sample denoising level — the max over frames
        # recovers the predicted-frame sigma. Per-batch [B] t is untouched.
        if t.dim() > 1:
            t = t.flatten(1).amax(dim=1)
        # Fused conditioning embedding from the encoder (action/proprio/goal/…);
        # condition dropout / CFG is applied upstream by the encoder's null path.
        cond_embedding = resolve_condition_embedding(cond)
        step_level = cond.get(self.step_level_key) if isinstance(cond, Mapping) else None
        # Per-frame action tokens for the cross-attention path. Training always
        # sees the preprocessor's per-frame `action_seq`; silently falling back
        # to the aggregated `action` (one summed token, values ~sum over the
        # clip) is exactly the train/inference mismatch that collapsed the
        # replace-run rollouts (cos vs base 0.997 -> 0.63; 2026-07-20). In a
        # cross-attention mode a present-but-aggregated action therefore raises
        # instead of degrading silently. Truly action-free conds stay allowed.
        action_tokens = None
        if isinstance(cond, Mapping):
            at = cond.get(self.action_token_key)
            if isinstance(at, Tensor):
                action_tokens = at
            elif self.uses_action_tokens and isinstance(cond.get(self.action_fallback_key), Tensor):
                raise ValueError(
                    f"Cross-attention action injection needs per-frame '{self.action_token_key}' tokens "
                    f"(trained on them), but cond only has the aggregated '{self.action_fallback_key}'. "
                    "Pass the preprocessor's action_seq through to generation instead of the summed action."
                )
        result = self.module(
            x_t,
            t,
            cond_embedding=cond_embedding,
            step_level=step_level if isinstance(step_level, Tensor) else None,
            base_output=base_output,
            action_tokens=action_tokens,
        )
        if self.output_mask:
            # Gated composition: the main head plus a per-pixel gate.
            #   mask_mix (predict_full):  main is a standalone *prediction*;
            #     AdaptedModel composes base*σ(gate+b) + pred*(1-σ(gate+b)).
            #   gated_residual:           main is a ~0-init *delta*;
            #     AdaptedModel composes base + σ(gate+b)*Δ.
            main, gate = result
            kind = "prediction" if self.predict_full else "delta"
            return OutputAdapterResult(
                adapter_output=main.to(x_t.dtype),
                output_kind=kind,
                gate=gate.to(x_t.dtype),
            )
        return OutputAdapterResult(adapter_output=result.to(x_t.dtype), output_kind="delta")


def _load_action_wan_params(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_params = raw.get("model", {}).get("params", {})
    params = model_params.get("dit_config", {}).get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Could not find `model.params.dit_config.params` in {path}")
    params = dict(params)
    for key in ("patch_size",):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    # action_dim is supplied by the conditioning config, not the tier file.
    params.pop("action_dim", None)
    return params
