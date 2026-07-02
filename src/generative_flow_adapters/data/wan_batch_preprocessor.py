"""Batch preprocessor for the Wan2.1 flow-matching backbone.

Turns a MetaWorld clip batch (raw pixels + per-frame actions) into the
trainer-expected flow-matching batch. Unlike the diffusion path (where the
trainer builds ``x_t`` from ``target`` via ``q_sample``), the flow branch reads
``x_t`` and ``target`` straight from the batch — so this preprocessor builds the
consistent rectified-flow triple itself:

    z0   = WanVAE.encode(video)            # clean 16-ch latent
    noise ~ N(0, I)
    x_t  = (1 - t) * z0 + t * noise        # linear interpolation
    target (velocity) = noise - z0

Run with ``training.extra.use_batch_timesteps_for_flow: true`` so the trainer
uses the ``t`` paired with these tensors instead of resampling its own.

Conditioning is kept minimal and Wan-native: the frozen base sees only an
(optional) text ``context`` — None here, so it uses a null context — while the
trainable adapter is conditioned on the action via the ``action`` key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from generative_flow_adapters.data.batch_preprocessor import (
    _center_crop_to,
    resize_stretch,
    resize_with_pad,
)


def _lanczos_cover_crop_uint8(video: Tensor, out_h: int, out_w: int) -> Tensor:
    """PIL **LANCZOS** cover-resize + center-crop on raw uint8 frames, byte-for-byte
    matching the upstream Wan i2v conditioning-frame preprocessing
    (``img.resize(..., Image.LANCZOS)`` then center ``crop``).

    ``video``: ``[B, T, H, W, 3]`` uint8 (RGB) -> ``[B, T, out_h, out_w, 3]`` uint8.
    Done in uint8/PIL space *before* normalization (upstream resizes then
    ``to_tensor``), so the whole pipeline — filter, quantization, and
    normalization — matches what the frozen base sees at inference."""
    b, t, ih, iw, c = video.shape
    scale = max(out_w / iw, out_h / ih)
    rw, rh = round(iw * scale), round(ih * scale)  # upstream: round(iw*scale), round(ih*scale)
    x1 = (rw - out_w) // 2                          # upstream: (img.width  - ow)//2
    y1 = (rh - out_h) // 2                          # upstream: (img.height - oh)//2
    arr = video.detach().cpu().numpy()
    out = np.empty((b, t, out_h, out_w, c), dtype=np.uint8)
    for bi in range(b):
        for ti in range(t):
            img = Image.fromarray(arr[bi, ti])                 # HWC uint8 RGB
            img = img.resize((rw, rh), Image.LANCZOS)          # PIL takes (width, height)
            img = img.crop((x1, y1, x1 + out_w, y1 + out_h))   # center crop to (out_w, out_h)
            out[bi, ti] = np.asarray(img)
    return torch.from_numpy(out)


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> tuple[int, int]:
    """Largest ``(width, height)`` that fits within ``expected_area`` pixels, is
    divisible by ``(dw, dh)``, and best preserves the ``w/h`` aspect ratio.

    First-party port of Wan2.2's ``wan/utils/utils.best_output_size`` (Apache-2.0)
    — the resolution policy the upstream i2v pipeline uses. Rounding both sides to
    the ``(dw, dh)`` grid independently would drift the aspect ratio, so it tries
    snapping width-first and height-first and keeps whichever stays closest to the
    source ratio. ``dw``/``dh`` are ``patch_size · vae_stride`` (32 for Wan2.2
    TI2V), so the result tiles into whole latent/patch tokens with no padding.
    """
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow
    # width-first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / max(ow1, 1) // dh * dh)
    ratio1 = ow1 / oh1
    # height-first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / max(oh2, 1) // dw * dw)
    ratio2 = ow2 / oh2
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


@dataclass(slots=True)
class WanBatchPreprocessConfig:
    target_height: int | None = 256
    target_width: int | None = 256
    resize_mode: str = "stretch"  # "stretch" | "pad"
    # Aspect-preserving alternative to a fixed target_h/w: when `max_area` is set,
    # frames are resized to the largest (w, h) that fits `max_area` pixels, is
    # divisible by (align_w, align_h), and best matches the source aspect ratio
    # (Wan `best_output_size`), then center-cropped. This matches how the upstream
    # Wan i2v pipeline sizes frames — the base's native, patch/VAE-aligned regime —
    # instead of a fixed 256² that runs the frozen base off-distribution.
    max_area: int | None = None
    align_h: int = 32  # patch_size[1] * vae_stride[1] (Wan2.2 TI2V: 2 * 16)
    align_w: int = 32  # patch_size[2] * vae_stride[2]
    action_key: str = "action"
    action_aggregation: str = "sum"  # "sum" | "mean" | "last" over the clip's frames
    sigma_min: float = 1e-5
    # The model is fed t = sigma * timestep_scale. Wan's pretrained convention is
    # t in [0, num_train_timesteps] = [0, 1000], so the interpolation coordinate
    # sigma in [0, 1] is scaled up before the DiT sees it. Feeding raw sigma
    # (scale=1) puts the frozen base off-distribution — see the fix ticket.
    timestep_scale: float = 1000.0


class WanBatchPreprocessor:
    """Encode MetaWorld clips to Wan latents and build a flow-matching batch."""

    def __init__(self, vae: Any, config: WanBatchPreprocessConfig, condition_keys: tuple[str, ...] = ()) -> None:
        self.vae = vae
        self.config = config
        # Dataset-emitted structured conditions (e.g. ("act",)); the first is
        # routed to the adapter's MLP action encoder under `action_key`.
        self.condition_keys = tuple(condition_keys)

    @property
    def device(self) -> torch.device:
        return self.vae.device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        del train
        video = self._normalize_video(batch["video"]).to(device=self.device, dtype=torch.float32)
        batch_size = video.shape[0]

        # Wan-VAE encodes a list of [C, T, H, W] clips -> list of [16, f, h, w].
        z0 = torch.stack(self.vae.encode([video[i] for i in range(batch_size)]), dim=0).float()

        noise = torch.randn_like(z0)
        # sigma in [0,1] is the interpolation coordinate; t fed to the model is
        # sigma * timestep_scale (Wan native = [0,1000]).
        sigma = torch.rand(batch_size, device=z0.device, dtype=z0.dtype).clamp_min(self.config.sigma_min)
        sigma_b = sigma.view(batch_size, *([1] * (z0.dim() - 1)))
        x_t = (1.0 - sigma_b) * z0 + sigma_b * noise
        target = noise - z0  # rectified-flow velocity (sigma-independent)
        t = sigma * self.config.timestep_scale

        cond = self._build_condition(batch, batch_size)
        # `x0` (clean latent) is kept alongside the velocity `target` so the
        # eval video logger can decode a correct ground-truth panel — decoding
        # `target` (= noise - z0) would render noise, not the source clip.
        return {"x_t": x_t, "t": t, "target": target, "x0": z0, "cond": cond}

    def _build_condition(self, batch: Mapping[str, Any], batch_size: int) -> dict[str, Tensor]:
        cond: dict[str, Tensor] = {}
        keys = self.condition_keys or ("act",)
        for i, key in enumerate(keys):
            value = batch.get(key)
            if not isinstance(value, Tensor):
                continue
            agg = self._aggregate_action(value).to(device=self.device, dtype=torch.float32)
            # The first structured condition feeds the adapter's action encoder.
            out_key = self.config.action_key if i == 0 else key
            cond[out_key] = agg
        return cond

    def _aggregate_action(self, action: Tensor) -> Tensor:
        # action: [B, T, A] per-frame -> [B, A]. MetaWorld stores delta-actions,
        # so SUM matches the strided-window aggregation used elsewhere.
        if action.dim() == 2:  # already [B, A]
            return action
        if action.dim() != 3:
            raise ValueError(f"Expected action [B,T,A] or [B,A], got {tuple(action.shape)}")
        mode = self.config.action_aggregation
        if mode == "sum":
            return action.sum(dim=1)
        if mode == "mean":
            return action.mean(dim=1)
        if mode == "last":
            return action[:, -1]
        raise ValueError(f"Unknown action_aggregation: {mode!r}")

    def _normalize_video(self, video: Any) -> Tensor:
        if not isinstance(video, Tensor):
            raise TypeError(f"Expected tensor 'video', got {type(video).__name__}.")
        cfg = self.config

        # Wan-native path (max_area set): resize the RAW uint8 frames with PIL
        # LANCZOS + center-crop *before* normalizing — exactly the upstream i2v
        # preprocessing — so the base's conditioning-latent distribution is
        # identical at train and inference. (Falls through to the tensor/bicubic
        # branch below for non-uint8 input, which the dataset never emits.)
        if cfg.max_area is not None and video.dtype == torch.uint8:
            if video.dim() != 5 or video.shape[-1] != 3:
                raise ValueError(f"Expected uint8 video [B,T,H,W,3], got {tuple(video.shape)}")
            src_h, src_w = int(video.shape[2]), int(video.shape[3])
            out_w, out_h = best_output_size(src_w, src_h, cfg.align_w, cfg.align_h, cfg.max_area)
            video = _lanczos_cover_crop_uint8(video, out_h, out_w)  # [B,T,out_h,out_w,3] uint8
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0  # == to_tensor().sub(.5).div(.5)
            return normalized.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C]->[B,C,T,H,W]

        if video.dtype == torch.uint8:
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0
            normalized = normalized.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C]->[B,C,T,H,W]
        elif video.dim() == 5 and video.is_floating_point():
            normalized = video
        else:
            raise ValueError(f"Unsupported video tensor: shape={tuple(video.shape)} dtype={video.dtype}")

        if cfg.max_area is None and (cfg.target_height is None or cfg.target_width is None):
            return normalized

        batch, channels, frames = normalized.shape[:3]
        # Reshape to [B*T, C, H, W] for the 2D resize helpers.
        flat = normalized.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *normalized.shape[3:])
        if cfg.max_area is not None:
            # Float fallback (non-uint8 input): tensor bicubic cover-crop.
            _, _, src_h, src_w = flat.shape
            out_w, out_h = best_output_size(src_w, src_h, cfg.align_w, cfg.align_h, cfg.max_area)
            resized = _center_crop_to(flat, out_h, out_w)
        elif cfg.resize_mode == "pad":
            out_h, out_w = cfg.target_height, cfg.target_width
            resized = resize_with_pad(flat, out_h, out_w)
        else:
            out_h, out_w = cfg.target_height, cfg.target_width
            resized = resize_stretch(flat, out_h, out_w)
        return resized.reshape(batch, frames, channels, out_h, out_w).permute(0, 2, 1, 3, 4)
