"""Strict, backbone-agnostic base-video-model interface.

The lesson from the vendored-Wan detour: reimplementing a backbone's sampling
loop (scheduler, shift, CFG, diffusion-forcing mask, resolution handling) is
where correctness quietly dies. So the contract here is deliberately narrow and
delegates *generation* to the backbone's own native pipeline.

A :class:`BaseVideoModel` is a **plug-and-play, frozen** video backbone. It
exposes exactly four capabilities:

- ``encode`` / ``decode`` — the pixel<->latent VAE seam.
- ``denoise`` — a single denoiser step ``(x_t, t, cond) -> prediction``. This is
  the seam an :class:`~...adapters.base.Adapter` composes onto additively
  (``prediction = base.denoise(...) + adapter(...)``), and the one used for
  training.
- ``generate`` — a **full native rollout** conditioned on one or more frames.
  It runs the backbone's *own* denoising loop (not ours), so base-only output
  matches the upstream repo by construction. An optional ``adapter`` is injected
  at the ``denoise`` seam inside that native loop.

Any backbone (Wan2.2, DynamiCrafter, ...) that implements these four methods is
interchangeable to the adapter and training code, which never learns which
backbone it is driving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor, nn

# Per-step composition callback used inside a backbone's native generation loop:
# ``(x_t, t, base_prediction) -> final_prediction``. ``AdaptedModel`` supplies
# one that adds the adapter residual; base-only generation passes ``None``.
ComposeFn = Callable[[Tensor, object, Tensor], Tensor]


class BaseVideoModel(nn.Module, ABC):
    """Frozen video backbone behind a narrow, delegation-first interface.

    ``model_type`` is ``"diffusion"`` or ``"flow"`` and ``prediction_type`` is
    ``"noise"`` or ``"velocity"`` — they carry the loss semantics, exactly as in
    the older :class:`BaseGenerativeModel`, so existing losses keep working.
    """

    # When True, the Trainer's periodic native generation eval must drive this
    # backbone with a whole *batch* + backbone-native kwargs (DynamiCrafter's
    # ``generate(batch, ddim_steps, fs, ...)``) rather than the Wan-shaped
    # per-frame ``generate(frame, max_area, frame_num, ...)`` path. Subclasses
    # that own a batch-based native loop set this True to select the matching
    # branch in ``Trainer._native_eval_grid`` / ``_native_quality_eval``.
    native_eval_batch_mode = False

    def __init__(self, model_type: str, prediction_type: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.prediction_type = prediction_type

    def freeze(self) -> "BaseVideoModel":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    @property
    def diffusion_schedule_config(self):
        """Diffusion schedule (DDPM/DDIM) config, or ``None`` for flow models.
        Present for parity with the older ``BaseGenerativeModel`` so
        :class:`AdaptedModel` can read it uniformly."""
        return None

    # -- VAE seam ---------------------------------------------------------
    @abstractmethod
    def encode(self, pixels: Tensor) -> Tensor:
        """``[B, 3, T, H, W]`` pixels in ``[-1, 1]`` -> latent ``[B, C, T', h, w]``."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: Tensor) -> Tensor:
        """Latent ``[B, C, T', h, w]`` -> ``[B, 3, T, H, W]`` pixels in ``[-1, 1]``."""
        raise NotImplementedError

    # -- denoiser seam (training + adapter composition) -------------------
    @abstractmethod
    def denoise(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """One denoiser step: noisy latent ``x_t`` at timestep ``t`` -> the
        backbone's prediction (velocity or noise). This is the point an adapter
        composes onto additively."""
        raise NotImplementedError

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """Alias of :meth:`denoise` so a base is drop-in for the single-step
        composition in :class:`~...models.adapted_model.AdaptedModel` (which
        calls ``base_model(x_t, t, cond)``)."""
        return self.denoise(x_t, t, cond)

    # -- native generation ------------------------------------------------
    @abstractmethod
    def generate(
        self,
        conditioning: object,
        *,
        compose_fn: ComposeFn | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run the backbone's **own** generation loop conditioned on
        ``conditioning`` (e.g. an observation frame). Returns decoded pixels.

        With ``compose_fn is None`` the output must equal the upstream
        backbone's native generation. Otherwise, at every denoiser step of that
        native loop, the backbone must replace its prediction ``base_pred`` with
        ``compose_fn(x_t, t, base_pred)`` — the seam through which
        :class:`AdaptedModel` injects the composed adapter residual. ``x_t`` and
        ``base_pred`` are the batched latent / prediction ``[B, C, T', h, w]``;
        ``t`` is whatever timestep form the backbone feeds its DiT."""
        raise NotImplementedError
