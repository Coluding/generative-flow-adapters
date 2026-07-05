from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.conditioning.encoders import ConditionEncoder
from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig
from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel

# Env-gated phase timer (shared switch with the trainer: GFA_PROFILE=1). Splits
# the base vs adapter forward, which the trainer's single `model forward` can't see.
_PROFILE = os.environ.get("GFA_PROFILE", "").strip().lower() not in ("", "0", "false", "no")


@contextlib.contextmanager
def _prof(name: str):
    if not _PROFILE:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[prof]     {name}: {(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)


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
        """Single-step composition: compute the frozen base prediction, then
        compose the adapter residual onto it. This is the training seam."""
        if hasattr(self.adapter, "clear_captured_base_features"):
            self.adapter.clear_captured_base_features()
        # Some adapters (e.g. HyperAlign) inject dynamic LoRA weights into the
        # base model and intentionally leave them set across forward+backward
        # so that gradient checkpointing's recomputation matches the original
        # forward. Clear before our frozen reference pass so it sees the
        # unmodified base.
        if hasattr(self.adapter, "clear_dynamic_parameters"):
            self.adapter.clear_dynamic_parameters()
        with torch.no_grad(), _prof("base forward (frozen 5B, no_grad)"):
            base_output = self.base_model(x_t, t, cond=cond)
        return self._compose_with_adapter(x_t, t, cond, base_output)

    def _compose_with_adapter(
        self, x_t: Tensor, t: object, cond: object | None, base_output: Tensor
    ) -> Tensor:
        """Given the base prediction ``base_output`` for ``(x_t, t, cond)``, run
        the adapter and compose. Shared by :meth:`forward` (which computes the
        base itself) and :meth:`generate` (where the base's native loop already
        computed it and passes it in via ``compose_fn``)."""
        drop_mask = self._sample_condition_drop_mask(x_t)
        encoded_cond = self.condition_encoder(cond, drop_mask=drop_mask) if self.condition_encoder is not None else cond
        base_direction = self._build_base_direction(base_output)
        adapter_cond = self._build_adapter_condition(raw_cond=cond, encoded_cond=encoded_cond, base_direction=base_direction)
        with _prof("adapter forward (delta, trainable)"):
            adapter_result = self.adapter(x_t, t, adapter_cond, base_output=base_output)
        return self._compose(base_output, adapter_result)

    @torch.no_grad()
    def generate(self, conditioning: object, *, cond: object | None = None, **kwargs: object) -> Tensor:
        """Generate through the base backbone's **own** native pipeline, with the
        adapter injected at each denoiser step.

        Requires ``base_model`` to implement
        :class:`~...models.base.video_model.BaseVideoModel` (i.e. expose
        ``generate(conditioning, *, compose_fn, ...)``). ``conditioning`` is the
        backbone's native conditioning (e.g. an observation frame); ``cond`` is
        the adapter's conditioning (e.g. an action), held constant across the
        rollout. Extra ``kwargs`` pass through to the backbone's generator
        (steps, resolution, seed, ...)."""
        generate = getattr(self.base_model, "generate", None)
        if not callable(generate):
            raise TypeError(
                "AdaptedModel.generate requires a BaseVideoModel base_model exposing .generate(); "
                f"got {type(self.base_model).__name__}."
            )

        def _compose_step(x_t: Tensor, t: object, base_pred: Tensor) -> Tensor:
            return self._compose_with_adapter(x_t, t, cond, base_pred)

        return generate(conditioning, compose_fn=_compose_step, **kwargs)

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
            # AVID prediction-blend: the adapter output is a *standalone*
            # prediction competing with the base. gate≈1 ⇒ keep base.
            if output_kind != "prediction":
                raise ValueError("Mask-mix composition requires adapter outputs with output_kind='prediction'.")
            if adapter_result.gate is None:
                raise ValueError("Mask-mix composition requires an adapter gate output.")
            gate = torch.sigmoid(adapter_result.gate + self.gate_bias)
            return base_output * gate + adapter_result.adapter_output * (1.0 - gate)
        elif composition in {"gated_residual", "gated_add", "residual_mask"}:
            # Gated residual: the adapter output is a *contribution* added to the
            # base, with a learned gate deciding where/how much to apply it
            # (the thesis core rule f = base + g·Δ). Works for any contribution
            # regardless of how it was formed (direct delta or affine scale/shift),
            # and is identity at init whenever Δ→0 — no gate_bias needed.
            if adapter_result.gate is None:
                raise ValueError("gated_residual composition requires an adapter gate output.")
            gate = torch.sigmoid(adapter_result.gate + self.gate_bias)
            return base_output + gate * adapter_result.adapter_output

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
