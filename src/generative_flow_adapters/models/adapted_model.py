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
        gate_cap: float | None = None,
        pretrain_steps: int = 0,
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
        # Upper clamp on the post-sigmoid gate for gated compositions. In
        # mask_mix, gate -> 1 means "all base": the single-clip overfit run
        # uxrst2k5 showed the gate saturating 0.5 -> 0.99 within ~70 steps,
        # after which the adapter branch's gradient (scaled by 1-gate) and the
        # gate's own gradient (sigmoid slope) both die together. Capping at
        # e.g. 0.9 guarantees the adapter prediction keeps >= (1-cap) of the
        # gradient forever. None = no cap (original behaviour).
        self.gate_cap = gate_cap
        # AVID pure-adapter warmup (see AdapterConfig.pretrain_steps). The
        # Trainer feeds the current optimizer step into `_train_step` before
        # each forward; while `_train_step < pretrain_steps` the composition is
        # bypassed (loss on the standalone adapter prediction) so the pred head
        # is competent before the gate starts learning.
        self.pretrain_steps = int(pretrain_steps)
        self._train_step = 0
        self.include_base_direction = include_base_direction
        self.normalize_base_direction = normalize_base_direction
        self.adapter.attach_base_model(base_model)
        # Post-sigmoid gate tensor from the most recent `_compose` call, for
        # gated compositions only (`mask_mix`/`avid_mask_mix`/`gated_residual`).
        # None for `add`/`replace` (no gate) or before the first forward.
        # Detached diagnostic snapshot only — read by Trainer for
        # gate_mean/gate_std/gate histogram logging; not used in the loss.
        self._last_gate: Tensor | None = None
        # Raw adapter-branch output (pre-composition) from the most recent
        # `_compose` call, detached; None before the first forward. Read by the
        # Trainer for the pred-vs-base cosine diagnostic.
        self._last_adapter_out: Tensor | None = None

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

    # The trainer can request the intermediate frozen-base prediction alongside
    # the composed output (for paired base-vs-adapted diagnostics) via
    # ``return_base=True`` — see Trainer._forward_and_loss. Advertised so the
    # trainer can feature-detect without importing this class.
    supports_return_base = True

    def forward(
        self, x_t: Tensor, t: Tensor, cond: object | None = None, *, return_base: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Single-step composition: compute the frozen base prediction, then
        compose the adapter residual onto it. This is the training seam. With
        ``return_base=True`` also return the frozen base prediction (already
        computed here), so the caller can score both on the same batch without a
        second base forward."""
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
        composed = self._compose_with_adapter(x_t, t, cond, base_output)
        if return_base:
            return composed, base_output
        return composed

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

    def set_train_step(self, step: int) -> None:
        """Trainer hook: current optimizer step, drives the AVID pure-adapter
        warmup schedule in `_compose`. No-op effect unless `pretrain_steps > 0`."""
        self._train_step = int(step)

    def _compose(self, base_output: Tensor, adapter_result: Tensor | OutputAdapterResult) -> Tensor:
        self._last_gate = None
        # Raw adapter-branch output from this call (pre-composition), detached —
        # diagnostic only, read by the Trainer for the pred-vs-base cosine.
        # Distinct from the composed output: under mask_mix a high gate makes
        # the composed output ≈base regardless of what the pred head learned,
        # so only the raw branch reveals base-cloning there.
        self._last_adapter_out = (
            adapter_result.detach() if isinstance(adapter_result, Tensor)
            else adapter_result.adapter_output.detach()
        )
        if isinstance(adapter_result, Tensor):
            return base_output + adapter_result

        output_kind = adapter_result.output_kind.lower()
        composition = self.output_composition.lower()

        # AVID pure-adapter warmup: for the first `pretrain_steps`, bypass the
        # gate and return the STANDALONE adapter output (mask_mix) / full residual
        # (gated_residual). The gate tensor is not used, so it gets no gradient
        # and stays at init until the warmup ends — by which point the pred head
        # is already competent, so the gate has something worth mixing in.
        if self.pretrain_steps > 0 and self._train_step < self.pretrain_steps:
            if composition in {"mask_mix", "avid_mask_mix"}:
                if output_kind != "prediction":
                    raise ValueError("Mask-mix warmup requires adapter output_kind='prediction'.")
                self._last_gate = torch.zeros_like(adapter_result.adapter_output).detach()
                return adapter_result.adapter_output
            if composition in {"gated_residual", "gated_add", "residual_mask"}:
                self._last_gate = torch.ones_like(adapter_result.adapter_output).detach()
                return base_output + adapter_result.adapter_output
            # add / replace: already "pure adapter" or ungated — warmup is a no-op.

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
            if self.gate_cap is not None:
                gate = gate.clamp(max=self.gate_cap)
            self._last_gate = gate.detach()
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
            self._last_gate = gate.detach()
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
