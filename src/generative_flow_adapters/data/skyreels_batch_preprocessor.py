"""Classic-i2v batch preprocessor for the SkyReels-V2-I2V-1.3B backbone.

Unlike Wan2.2 TI2V (diffusion-forcing: the observed frame's *clean latent* rides
in the sequence at timestep 0), SkyReels-V2-I2V is a **classic i2v** model: every
frame is noised, and the observation is injected through two side channels the
DiT consumes each step —

    y        = cat([mask[:, :4], VAE.encode(cond_frame + zero-pad)], dim=1)   # [B, 20, T', h, w]
    clip_fea = CLIP.encode_video(cond_frame)                                   # [B, 257, D]

(exactly how ``Image2VideoPipeline`` builds them). Text rides in ``context`` from
SkyReels' **own** T5 (NOT Wan umT5). So this preprocessor emits the same batch
contract as :class:`Wan22DiffusionForcingPreprocessor`
(``{x_t, t, target, x0, frame_mask, cond}``) but:

- ``x_t`` noises ALL frames (no clean-obs frames held in the latent);
- ``frame_mask`` is all-ones (every frame is a predicted/loss frame);
- ``cond`` additionally carries ``context`` / ``clip_fea`` / ``y`` — precisely the
  keys :meth:`SkyReelsVideoModel.denoise` requires.

``y`` / ``clip_fea`` / ``context`` depend only on the conditioning frame + prompt,
so they are built live per batch (only ``z0`` benefits from the latent cache).

DRAFT status: the tensor plumbing mirrors the vendored ``Image2VideoPipeline``
(external_repos/SkyReels-V2/.../pipelines/image2video_pipeline.py:88-116); the
exact text-encoder batching and CLIP call signature still need a GPU run to
confirm (flagged inline with ``# GPU-VALIDATE``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from generative_flow_adapters.data.wan_batch_preprocessor import (
    WanBatchPreprocessConfig,
    WanBatchPreprocessor,
)


class SkyReelsI2VPreprocessor(WanBatchPreprocessor):
    """Encode ACWM/MetaWorld clips to SkyReels 16-ch latents and build a
    classic-i2v flow-matching batch (obs injected via ``y``/``clip_fea``).

    Reuses the parent's action aggregation (``_build_condition``), pixel
    normalisation (``_normalize_video``) and latent cache; overrides the VAE
    encode (SkyReels' ``WanVAE`` takes a BATCHED tensor, not the Wan list) and
    ``__call__`` (classic noising + i2v side channels).
    """

    def __init__(
        self,
        model: Any,                       # SkyReelsVideoModel
        config: WanBatchPreprocessConfig,
        condition_keys: tuple[str, ...] = (),
        *,
        device: str = "cuda",
        default_prompt: str = "",
    ) -> None:
        pipe = model._pipeline
        super().__init__(vae=pipe.vae, config=config, condition_keys=condition_keys)
        self._model = model
        self._pipe = pipe
        self._clip = pipe.clip
        self._text_encoder = pipe.text_encoder
        self._device = torch.device(device)
        self._default_prompt = default_prompt

    # SkyReels' WanVAE lacks .device/.dtype attributes the parent reads.
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    # -- VAE encode (batched-tensor convention, NOT the Wan list) ----------
    def _encode_batched(self, pixels: Tensor) -> Tensor:
        """``[B,3,T,H,W]`` in ``[-1,1]`` -> ``[B,16,T',h,w]`` via SkyReels' VAE."""
        return self.vae.encode(pixels.to(self._device).float()).float()

    def _encode_z0(self, raw_video: Tensor, batch: Mapping[str, Any], batch_size: int) -> Tensor:
        """Cache-first z0 encode. Mirrors the parent's cache structure but uses the
        SkyReels batched-tensor VAE. On a full hit the raw frames are never
        resized/encoded; misses are normalised + encoded as a batch."""
        normalized = None
        if self.latent_cache is None:
            normalized = self._normalize_video(raw_video).to(self._device, torch.float32)
            self.last_encoded = batch_size
            return self._encode_batched(normalized)

        standard = raw_video.dim() == 5 and raw_video.shape[-1] == 3 and raw_video.dtype == torch.uint8
        if standard:
            frames = int(raw_video.shape[1])
            out_h, out_w = self._output_hw(int(raw_video.shape[2]), int(raw_video.shape[3]))
        else:
            normalized = self._normalize_video(raw_video).to(self._device, torch.float32)
            frames, out_h, out_w = int(normalized.shape[2]), int(normalized.shape[3]), int(normalized.shape[4])

        keys = self._latent_keys(batch, batch_size, frames, out_h, out_w)
        z_list: list[Tensor | None] = [None] * batch_size
        miss_idx: list[int] = []
        for i, k in enumerate(keys):
            cached = self.latent_cache.get(k) if k is not None else None
            if cached is not None:
                z_list[i] = cached.to(self._device).float()
            else:
                miss_idx.append(i)
        self.last_encoded = len(miss_idx)
        if miss_idx:
            if normalized is None:
                miss = self._normalize_video(raw_video[miss_idx]).to(self._device, torch.float32)
            else:
                miss = normalized[miss_idx]
            enc = self._encode_batched(miss)  # [M,16,T',h,w]
            for j, i in enumerate(miss_idx):
                z = enc[j].float()
                z_list[i] = z
                if keys[i] is not None and self.config.write_latent_cache:
                    self.latent_cache.put(keys[i], z)
        return torch.stack([z for z in z_list if z is not None], dim=0)

    # -- i2v side channels (built live from the cond frame) ----------------
    def _build_i2v_conditioning(self, cond_pixels: Tensor, num_pixel_frames: int) -> dict[str, Tensor]:
        """From the normalised conditioning frame ``[B,3,1,H,W]`` build the
        ``y`` (encoded frame + mask) and ``clip_fea`` the SkyReels DiT needs.
        ``num_pixel_frames`` is the window's pixel length so ``img_cond`` encodes to
        the same T' as z0. Mirrors image2video_pipeline.py:88-116."""
        b, c, _, h, w = cond_pixels.shape
        pad_len = max(int(num_pixel_frames) - 1, 0)  # zero pixel-frames after the cond frame
        padding = torch.zeros(b, c, pad_len, h, w, device=self._device, dtype=cond_pixels.dtype)
        img_cond_px = torch.cat([cond_pixels, padding], dim=2)          # [B,3,1+pad,H,W]
        img_cond = self._encode_batched(img_cond_px)                    # [B,16,T',h,w]
        mask = torch.ones_like(img_cond)
        mask[:, :, 1:] = 0.0                                            # only the first latent frame is observed
        y = torch.cat([mask[:, :4], img_cond], dim=1)                  # [B,20,T',h,w]
        # GPU-VALIDATE: clip.encode_video expects the [B,3,1,H,W] conditioning frame.
        clip_fea = self._clip.encode_video(cond_pixels.to(self._device))
        return {"y": y, "clip_fea": clip_fea}

    def _encode_text(self, prompts: list[str]) -> Any:
        """SkyReels' own T5 encode of per-sample prompts. Returns whatever the DiT
        text_embedding expects ([B,L,C] tensor, or a list of [L,C])."""
        # GPU-VALIDATE: pipeline calls text_encoder.encode(<str>); confirm list batching.
        return self._text_encoder.encode(prompts)

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        raw_video = batch["video"]
        batch_size = int(raw_video.shape[0])

        z0 = self._encode_z0(raw_video, batch, batch_size)   # [B,16,T',h,w]
        t_lat = z0.shape[2]

        # Classic flow-matching: noise ALL frames (obs rides in y/clip_fea, not
        # the latent). sigma ~ U(0,1) with the optional high-noise shift (train only).
        noise = torch.randn_like(z0)
        sigma = torch.rand(batch_size, device=z0.device, dtype=z0.dtype)
        shift = self.config.sigma_shift
        if train and shift is not None and shift != 1.0:
            sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        sigma = sigma.clamp_min(self.config.sigma_min)
        sigma_b = sigma.view(batch_size, *([1] * (z0.dim() - 1)))
        x_t = (1.0 - sigma_b) * z0 + sigma_b * noise
        target = noise - z0                                   # rectified-flow velocity
        t = sigma * self.config.timestep_scale                # [B] uniform (SkyReels DiT broadcasts)
        frame_mask = torch.ones(batch_size, t_lat, device=z0.device, dtype=z0.dtype)  # all frames predicted

        # Conditioning: action (parent) + i2v side channels + text.
        cond = self._build_condition(batch, batch_size, train)
        norm_px = self._normalize_video(raw_video).to(self._device, torch.float32)  # [B,3,T,H,W]
        num_pixel_frames = int(norm_px.shape[2])
        cond.update(self._build_i2v_conditioning(norm_px[:, :, 0:1], num_pixel_frames))
        prompts = self._prompts_for(batch, batch_size)
        cond["context"] = self._encode_text(prompts)
        return {"x_t": x_t, "t": t, "target": target, "x0": z0, "frame_mask": frame_mask, "cond": cond}

    def _prompts_for(self, batch: Mapping[str, Any], batch_size: int) -> list[str]:
        """Per-sample prompt strings. Uses the dataset's ``task_name`` when present,
        else the config default. (SkyReels encodes text live, so no umT5 table.)"""
        names = batch.get("task_name")
        if isinstance(names, (list, tuple)) and len(names) == batch_size:
            return [str(n) or self._default_prompt for n in names]
        return [self._default_prompt] * batch_size
