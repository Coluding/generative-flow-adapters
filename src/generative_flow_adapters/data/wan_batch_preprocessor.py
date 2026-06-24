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

import torch
from torch import Tensor

from generative_flow_adapters.data.batch_preprocessor import resize_stretch, resize_with_pad


@dataclass(slots=True)
class WanBatchPreprocessConfig:
    target_height: int | None = 256
    target_width: int | None = 256
    resize_mode: str = "stretch"  # "stretch" | "pad"
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
        if video.dtype == torch.uint8:
            normalized = video.to(dtype=torch.float32) / 127.5 - 1.0
            normalized = normalized.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C]->[B,C,T,H,W]
        elif video.dim() == 5 and video.is_floating_point():
            normalized = video
        else:
            raise ValueError(f"Unsupported video tensor: shape={tuple(video.shape)} dtype={video.dtype}")

        cfg = self.config
        if cfg.target_height is None or cfg.target_width is None:
            return normalized

        batch, channels, frames = normalized.shape[:3]
        # Reshape to [B*T, C, H, W] for the 2D resize helpers.
        flat = normalized.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *normalized.shape[3:])
        if cfg.resize_mode == "pad":
            resized = resize_with_pad(flat, cfg.target_height, cfg.target_width)
        else:
            resized = resize_stretch(flat, cfg.target_height, cfg.target_width)
        return resized.reshape(batch, frames, channels, cfg.target_height, cfg.target_width).permute(0, 2, 1, 3, 4)
