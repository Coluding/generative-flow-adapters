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

    ``cond_frames`` observation frames at the front of the clip are held clean
    (timestep 0); the rest are the predicted future. Reuses the parent's VAE
    encode, video normalisation, and action-conditioning helpers.
    """

    def __init__(
        self,
        vae: Any,
        config: WanBatchPreprocessConfig,
        condition_keys: tuple[str, ...] = (),
        cond_frames: int = 1,
    ) -> None:
        super().__init__(vae=vae, config=config, condition_keys=condition_keys)
        self.cond_frames = int(cond_frames)

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        del train
        video = self._normalize_video(batch["video"]).to(device=self.device, dtype=torch.float32)
        batch_size = video.shape[0]

        # Wan2.2-VAE: list of [3, T, H, W] clips -> list of [48, T', h, w] latents.
        z0 = torch.stack(self.vae.encode([video[i] for i in range(batch_size)]), dim=0).float()
        t_lat = z0.shape[2]

        # Keep at least one predicted frame; for a single-latent-frame clip there
        # is no temporal future to condition on (k=0 -> pure generation).
        k = min(self.cond_frames, t_lat - 1) if t_lat > 1 else 0

        noise = torch.randn_like(z0)
        sigma = torch.rand(batch_size, device=z0.device, dtype=z0.dtype).clamp_min(self.config.sigma_min)
        sigma_b = sigma.view(batch_size, *([1] * (z0.dim() - 1)))

        # frame_mask: 1 = predicted (noised) future frame, 0 = clean observation.
        frame_mask = torch.ones(batch_size, t_lat, device=z0.device, dtype=z0.dtype)
        if k > 0:
            frame_mask[:, :k] = 0.0
        fm = frame_mask.view(batch_size, 1, t_lat, 1, 1)

        x_noised = (1.0 - sigma_b) * z0 + sigma_b * noise
        x_t = (1.0 - fm) * z0 + fm * x_noised  # obs frames clean, future frames noised
        target = noise - z0                    # rectified-flow velocity (sigma-independent)

        # Per-latent-frame timestep fed to the base: 0 on obs frames, σ·scale on
        # the future. The Wan2.2 wrapper expands this [B, T'] form across each
        # frame's patch tokens.
        t = frame_mask * (sigma.view(batch_size, 1) * self.config.timestep_scale)

        cond = self._build_condition(batch, batch_size)
        return {
            "x_t": x_t,
            "t": t,
            "target": target,
            "x0": z0,
            "frame_mask": frame_mask,
            "cond": cond,
        }
