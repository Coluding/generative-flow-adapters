"""Diffusion-forcing batch preprocessor for the Wan2.2 TI2V backbone.

Wan2.2 TI2V conditions on an observation frame *without* CLIP or a concat
channel: it holds the observed frame's clean latent in the sequence and gives
those tokens timestep 0, while the future frames are noised at the sampled
timestep ("diffusion forcing"; see ``backbones/wan/utils/diffusion_forcing.py``
and the upstream ``textimage2video.i2v``). This preprocessor builds that batch
for the action-conditioned world model:

    z0          = Wan2.2-VAE.encode(clip)              # clean 48-ch latent [B,48,T',h,w]
    frame_mask  = 0 on the observation frame(s), 1 on the predicted future
    x_t         = z0 on obs frames, (1-σ)·z0 + σ·noise on future frames
    t (per-frame) = 0 on obs frames, σ·timestep_scale on future frames
    target (v)  = noise - z0                            # loss masked to the future via frame_mask

The frozen Wan2.2 base does the observation conditioning natively; the trainable
adapter is conditioned on the action. ``frame_mask`` and ``x0`` are returned so
the trainer can mask the velocity loss to the predicted frames and the eval
logger can decode a correct ground-truth panel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from generative_flow_adapters.data.wan_batch_preprocessor import (
    WanBatchPreprocessConfig,
    WanBatchPreprocessor,
)


class Wan22DiffusionForcingPreprocessor(WanBatchPreprocessor):
    """Encode MetaWorld clips to Wan2.2 latents and build a diffusion-forcing
    (first-frame-conditioned) flow-matching batch.

    The number of leading observation frames held clean (timestep 0) is ``k``;
    the rest are the predicted future. ``k`` is fixed to ``cond_frames`` by
    default, but when ``cond_frames_dist`` is given, *training* draws ``k``
    per-sample from that categorical distribution over history lengths (so the
    model learns to roll out from a variable amount of context). *Eval* always
    uses the fixed scalar ``cond_frames`` so the logged panel stays stable.
    Reuses the parent's VAE encode, video normalisation, and action helpers.
    """

    def __init__(
        self,
        vae: Any,
        config: WanBatchPreprocessConfig,
        condition_keys: tuple[str, ...] = (),
        cond_frames: int = 1,
        cond_frames_dist: Mapping[int, float] | None = None,
    ) -> None:
        super().__init__(vae=vae, config=config, condition_keys=condition_keys)
        self.cond_frames = int(cond_frames)
        if cond_frames_dist:
            self._cond_values, self._cond_weights = self._parse_cond_frames_dist(
                cond_frames_dist
            )
        else:
            self._cond_values = None
            self._cond_weights = None

    @staticmethod
    def _parse_cond_frames_dist(
        dist: Mapping[int, float] | Any,
    ) -> tuple[list[int], list[float]]:
        """Validate a categorical distribution over the number of clean
        observation frames. Accepts a ``{k: weight}`` mapping (or an iterable of
        ``(k, weight)`` pairs). Keys are non-negative ints, weights are
        non-negative and must sum to 1."""
        items = list(dist.items()) if isinstance(dist, Mapping) else [tuple(p) for p in dist]
        if not items:
            raise ValueError("cond_frames_dist must have at least one entry")
        values: list[int] = []
        weights: list[float] = []
        for key, weight in items:
            ki, wf = int(key), float(weight)
            if ki < 0:
                raise ValueError(f"cond_frames_dist keys (frame counts) must be >= 0, got {ki}")
            if wf < 0:
                raise ValueError(f"cond_frames_dist weights must be >= 0, got {wf}")
            values.append(ki)
            weights.append(wf)
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"cond_frames_dist weights must sum to 1, got {total:.6f}")
        return values, weights

    def _sample_cond_frames(
        self, batch_size: int, t_lat: int, train: bool, device: torch.device
    ) -> torch.Tensor:
        """Per-sample number of clean observation frames ``k`` as a ``[B]`` long
        tensor, clamped so at least one predicted frame remains. Training draws
        from ``cond_frames_dist`` when set; eval (and the no-dist case) use the
        fixed scalar ``cond_frames``."""
        if t_lat <= 1:
            # No temporal future to condition on: pure generation (k = 0).
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if train and self._cond_values is not None:
            weights = torch.tensor(self._cond_weights, dtype=torch.float32, device=device)
            idx = torch.multinomial(weights, batch_size, replacement=True)
            values = torch.tensor(self._cond_values, dtype=torch.long, device=device)
            ks = values[idx]
        else:
            ks = torch.full((batch_size,), self.cond_frames, dtype=torch.long, device=device)
        return ks.clamp_(min=0, max=t_lat - 1)

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        raw_video, batch_size = self._batch_video_and_size(batch)

        # Wan2.2-VAE: list of [3, T, H, W] clips -> list of [48, T', h, w] latents.
        # `_encode_z0` is cache-first (data/latent_cache.py): on a hit it returns
        # the cached latent WITHOUT resizing/encoding the raw frames — skipping both
        # the VAE encode and the CPU LANCZOS resize (the per-step bottleneck). Pass
        # the RAW uint8 video so the resize is deferred to the miss path only.
        z0 = self._encode_z0(raw_video, batch, batch_size)
        t_lat = z0.shape[2]

        # Per-sample number of clean observation frames (>= 1 predicted frame
        # always remains). Training may draw it from cond_frames_dist; eval is
        # fixed to the scalar cond_frames for a stable panel.
        ks = self._sample_cond_frames(batch_size, t_lat, train, z0.device)  # [B] long

        noise = torch.randn_like(z0)
        sigma = torch.rand(batch_size, device=z0.device, dtype=z0.dtype)
        # Timestep shift toward high noise (train only — eval stays U(0,1) so
        # eval losses are comparable across runs). See WanBatchPreprocessConfig.
        shift = self.config.sigma_shift
        if train and shift is not None and shift != 1.0:
            sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        sigma = sigma.clamp_min(self.config.sigma_min)
        sigma_b = sigma.view(batch_size, *([1] * (z0.dim() - 1)))

        # frame_mask: 1 = predicted (noised) future frame, 0 = clean observation
        # (the first ks[i] frames of sample i).
        frame_idx = torch.arange(t_lat, device=z0.device).unsqueeze(0)  # [1, T']
        frame_mask = (frame_idx >= ks.unsqueeze(1)).to(z0.dtype)        # [B, T']
        fm = frame_mask.view(batch_size, 1, t_lat, 1, 1)

        x_noised = (1.0 - sigma_b) * z0 + sigma_b * noise
        x_t = (1.0 - fm) * z0 + fm * x_noised  # obs frames clean, future frames noised
        target = noise - z0                    # rectified-flow velocity (sigma-independent)

        # Per-latent-frame timestep fed to the base: 0 on obs frames, σ·scale on
        # the future. The Wan2.2 wrapper expands this [B, T'] form across each
        # frame's patch tokens.
        t = frame_mask * (sigma.view(batch_size, 1) * self.config.timestep_scale)

        # Pass `train` through (fix 2026-07-23): omitting it defaulted to True,
        # so EVAL batches drew a random, unseeded pool prompt instead of the
        # deterministic pool[0] — nondeterministic eval, and the probable cause
        # of the ACWM compare rollouts failing while the pinned-prompt probe
        # succeeded (only pool[0] was ever validated against the base).
        cond = self._build_condition(batch, batch_size, train=train)
        return {
            "x_t": x_t,
            "t": t,
            "target": target,
            "x0": z0,
            "frame_mask": frame_mask,
            "cond": cond,
        }
