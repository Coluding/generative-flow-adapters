"""Wandb logger for training: scalar metrics every step + side-by-side
ground-truth / base / adapted videos at the eval cadence.

Wandb is imported lazily so the rest of the package stays usable without it.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any

import torch
from torch import Tensor


class WandbLogger:
    """Single wandb run that handles scalar metrics and eval videos."""

    def __init__(
        self,
        *,
        decode_fn=None,
        num_samples: int = 2,
        fps: int = 4,
        project: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        metrics_prefix: str = "train",
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandbLogger requires the `wandb` package. Install with `pip install wandb`."
            ) from exc
        self._wandb = wandb
        self._decode_fn = decode_fn
        self.num_samples = max(1, int(num_samples))
        self.fps = int(fps)
        self.metrics_prefix = metrics_prefix.rstrip("/")
        if wandb.run is None:
            wandb.init(project=project, name=run_name, config=dict(config) if config else None)

    # ------------------------------------------------------------------ metrics

    def log_metrics(self, metrics: Mapping[str, object], step: int) -> None:
        """Push scalar entries from `metrics` to wandb. Tensor / array entries
        (e.g. `generated_samples`) are skipped — videos go through `log_videos`."""
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            scalar = _coerce_scalar(value)
            if scalar is None:
                continue
            payload[f"{self.metrics_prefix}/{key}" if self.metrics_prefix else key] = scalar
        if payload:
            self._wandb.log(payload, step=int(step))

    # ------------------------------------------------------------------- videos

    def log_videos(
        self,
        *,
        prediction_latents: Tensor,
        target_latents: Tensor,
        cond: object | None,
        step: int,
        base_prediction_latents: Tensor | None = None,
    ) -> None:
        if self._decode_fn is None:
            raise RuntimeError("WandbLogger.log_videos requires a decode_fn (set at construction).")
        if prediction_latents.shape != target_latents.shape:
            raise ValueError(
                f"Prediction and target latents must have matching shapes; "
                f"got {tuple(prediction_latents.shape)} vs {tuple(target_latents.shape)}."
            )
        if base_prediction_latents is not None and base_prediction_latents.shape != prediction_latents.shape:
            raise ValueError(
                f"Base-prediction latents must match adapted prediction shape; "
                f"got {tuple(base_prediction_latents.shape)} vs {tuple(prediction_latents.shape)}."
            )
        sample_count = min(self.num_samples, prediction_latents.shape[0])
        pred_pixels = self._decode_to_uint8(prediction_latents[:sample_count])
        target_pixels = self._decode_to_uint8(target_latents[:sample_count])
        base_pixels = (
            self._decode_to_uint8(base_prediction_latents[:sample_count])
            if base_prediction_latents is not None
            else None
        )
        actions = _maybe_extract_actions(cond)

        videos: dict[str, Any] = {}
        for i in range(sample_count):
            panels = [target_pixels[i]]
            if base_pixels is not None:
                panels.append(base_pixels[i])
            panels.append(pred_pixels[i])
            side_by_side = torch.cat(panels, dim=-1)
            caption = self._format_caption(
                actions=actions,
                sample_index=i,
                include_base=base_pixels is not None,
            )
            videos[f"eval/sample_{i}"] = self._wandb.Video(
                side_by_side.numpy(),
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
        self._wandb.log(videos, step=int(step))

    # --------------------------------------------------------------- internals

    @torch.no_grad()
    def _decode_to_uint8(self, latents: Tensor) -> Tensor:
        decoded = self._decode_fn(latents)
        if decoded.dim() != 5:
            raise ValueError(f"Decoder must return 5D [B, 3, T, H, W]; got {tuple(decoded.shape)}.")
        decoded = decoded.clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
        return decoded.permute(0, 2, 1, 3, 4).to(torch.uint8).cpu()

    @staticmethod
    def _format_caption(actions: Tensor | None, sample_index: int, include_base: bool = False) -> str:
        if include_base:
            layout = "left=ground_truth | middle=base_model | right=adapted"
        else:
            layout = "left=ground_truth | right=adapted"
        header = f"sample={sample_index} | {layout}"
        if not (isinstance(actions, Tensor) and actions.shape[0] > sample_index):
            return header
        sample_actions = actions[sample_index].detach().cpu()
        rows = ["t " + " ".join(f"a{j}" for j in range(sample_actions.shape[-1]))]
        for frame_index in range(sample_actions.shape[0]):
            row = sample_actions[frame_index].tolist()
            rows.append(f"{frame_index} " + " ".join(f"{x:+.2f}" for x in row))
        return header + "\nactions:\n" + "\n".join(rows)


def _coerce_scalar(value: object) -> float | None:
    """Return a float for scalar-like inputs, None for everything else."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None


def _maybe_extract_actions(cond: object | None) -> Tensor | None:
    if isinstance(cond, Mapping):
        act = cond.get("act")
        if isinstance(act, Tensor):
            return act
    return None
