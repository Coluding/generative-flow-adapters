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
        config_path: str | None = None,
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
        # Upload the raw config YAML so every run carries its exact source config.
        if config_path:
            from pathlib import Path

            path = Path(config_path)
            if path.is_file():
                wandb.save(str(path), base_path=str(path.parent), policy="now")

    def set_decode_fn(self, decode_fn) -> None:
        """Attach the latent->pixel decoder after construction.

        Used when the VAE lives outside ``build_experiment`` (e.g. the Wan
        training script loads ``WanVAE`` itself): the logger is built with no
        decoder and the script injects it here before the first eval."""
        self._decode_fn = decode_fn

    @property
    def can_decode(self) -> bool:
        """Whether a latent->pixel decoder is attached (gates quality metrics)."""
        return self._decode_fn is not None

    @torch.no_grad()
    def decode_to_uint8(self, latents: Tensor) -> Tensor | None:
        """Public latent->pixel decode → uint8 ``(B, T, 3, H, W)`` on CPU.

        Thin wrapper over the internal decoder so the trainer can score quality
        metrics on decoded pixels without reaching into a private method.
        Returns ``None`` when no decoder is attached."""
        if self._decode_fn is None:
            return None
        return self._decode_to_uint8(latents)

    # ------------------------------------------------------------------ metrics

    def log_metrics(
        self, metrics: Mapping[str, object], step: int, *, prefix: str | None = None
    ) -> None:
        """Push scalar entries from `metrics` to wandb. Tensor / array entries
        (e.g. `generated_samples`) are skipped — videos go through `log_videos`.

        `prefix` overrides the default `metrics_prefix` for this call; pass ``""``
        for keys that already carry their own namespace (e.g. eval metrics keyed
        ``eval/base/fvd_i3d``) so they aren't nested under ``train/``."""
        active_prefix = self.metrics_prefix if prefix is None else prefix.rstrip("/")
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            scalar = _coerce_scalar(value)
            if scalar is None:
                continue
            payload[f"{active_prefix}/{key}" if active_prefix else key] = scalar
        if payload:
            self._wandb.log(payload, step=int(step))

    def log_histogram(
        self, key: str, values: Tensor, step: int, *, max_samples: int = 100_000
    ) -> None:
        """Log a wandb Histogram from a (possibly large) tensor, e.g. the
        per-pixel gate values from a mask_mix/gated_residual composition.
        Randomly subsampled to `max_samples` to keep this cheap on big tensors
        (per-pixel gates can be millions of elements)."""
        flat = values.detach().float().flatten()
        if flat.numel() == 0:
            return
        if flat.numel() > max_samples:
            idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
            flat = flat[idx]
        self._wandb.log({key: self._wandb.Histogram(flat.cpu().numpy())}, step=int(step))

    # ------------------------------------------------ modality prediction vs gt

    def log_prediction_vs_gt(
        self,
        *,
        name: str,
        pred: Tensor,
        gt: Tensor,
        kind: str,
        step: int,
        sample_index: int = 0,
    ) -> None:
        """Log a rolled-out modality prediction against ground truth.

        ``vector`` streams (e.g. proprio, shape ``(T, D)``) become one
        ``line_series`` chart per dimension with ``gt`` and ``pred`` lines over
        time. ``map`` streams (shape ``(T, C, H, W)`` / ``(T, H, W)``) log a
        mid-clip frame as a side-by-side ``gt | pred`` image. No matplotlib.
        """
        pred = pred.detach().float().cpu()
        gt = gt.detach().float().cpu()
        tag = f"eval/{name}/sample_{sample_index}"

        if kind == "vector":
            t_len = pred.shape[0]
            xs = list(range(t_len))
            pred2 = pred.reshape(t_len, -1)
            gt2 = gt.reshape(t_len, -1)
            panels = {
                f"{tag}/dim_{d}": self._wandb.plot.line_series(
                    xs=xs,
                    ys=[gt2[:, d].tolist(), pred2[:, d].tolist()],
                    keys=["gt", "pred"],
                    title=f"{name}[{d}] (sample {sample_index})",
                    xname="frame",
                )
                for d in range(pred2.shape[1])
            }
            self._wandb.log(panels, step=int(step))
            return

        if kind == "map":
            frame = pred.shape[0] // 2
            pr = pred[frame]
            gtf = gt[frame]
            if pr.dim() == 3:  # (C, H, W) -> first channel
                pr, gtf = pr[0], gtf[0]
            panel = torch.cat([_minmax(gtf), _minmax(pr)], dim=-1).numpy()
            self._wandb.log(
                {f"{tag}/frame{frame}": self._wandb.Image(panel, caption=f"{name}: gt | pred")},
                step=int(step),
            )

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

    def log_step_size_grid_pixels(
        self,
        *,
        target_pixels: Tensor,
        adapted_by_steps: list[tuple[int, Tensor]],
        base_by_steps: list[tuple[int, Tensor]] | None,
        cond: object | None,
        step: int,
    ) -> None:
        """Pixel-space twin of :meth:`log_step_size_grid` for backbones whose
        native ``generate`` already returns decoded pixels (no VAE decode here).

        All tensors are uint8 ``[S, T, C, H, W]`` (``S`` samples already sliced to
        ``num_samples``); ``adapted_by_steps`` / ``base_by_steps`` are lists of
        ``(num_steps, pixels)``. Layout matches ``log_step_size_grid``: each row is
        ``[ground_truth | base@N | adapted@N]`` along width, rows stacked along
        height top→bottom in schedule order.
        """
        if not adapted_by_steps:
            return
        sample_count = target_pixels.shape[0]
        base_pixels = {n: px for n, px in base_by_steps} if base_by_steps else {}
        actions = _maybe_extract_actions(cond)

        videos: dict[str, Any] = {}
        for i in range(sample_count):
            rows = []
            for num_steps, pixels in adapted_by_steps:
                panels = [target_pixels[i]]
                if num_steps in base_pixels:
                    panels.append(base_pixels[num_steps][i])
                panels.append(pixels[i])
                rows.append(torch.cat(panels, dim=-1))
            grid = torch.cat(rows, dim=-2)
            cols = "gt | base | adapted" if base_pixels else "gt | adapted"
            order = ", ".join(f"N={n}" for n, _ in adapted_by_steps)
            caption = f"sample={i} | rows top→bottom: {order} | cols: {cols}"
            if isinstance(actions, Tensor) and actions.shape[0] > i:
                caption += self._format_action_block(actions[i])
            videos[f"eval_step_grid/sample_{i}"] = self._wandb.Video(
                grid.numpy(),
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
        self._wandb.log(videos, step=int(step))

    def log_step_size_grid(
        self,
        *,
        target_latents: Tensor,
        adapted_by_steps: list[tuple[int, Tensor]],
        base_by_steps: list[tuple[int, Tensor]] | None,
        cond: object | None,
        step: int,
    ) -> None:
        """Log one video per sample, stacking a row for each sampling step
        count so few-step vs. many-step rollouts can be compared at a glance.

        Each row is ``[ground_truth | base@N | adapted@N]`` concatenated along
        width; rows (top→bottom) follow the order of ``adapted_by_steps``. This
        is the shortcut-model diagnostic: a well-trained adapter should stay
        close to the high-N reference as N drops, while the frozen base
        degrades. ``base_by_steps`` may be ``None`` (then rows omit the base
        column); when given it must cover the same step counts as
        ``adapted_by_steps``.
        """
        if self._decode_fn is None:
            raise RuntimeError("WandbLogger.log_step_size_grid requires a decode_fn (set at construction).")
        if not adapted_by_steps:
            return
        sample_count = min(self.num_samples, target_latents.shape[0])
        target_pixels = self._decode_to_uint8(target_latents[:sample_count])
        adapted_pixels = [(n, self._decode_to_uint8(lat[:sample_count])) for n, lat in adapted_by_steps]
        base_pixels = (
            {n: self._decode_to_uint8(lat[:sample_count]) for n, lat in base_by_steps}
            if base_by_steps is not None
            else {}
        )
        actions = _maybe_extract_actions(cond)

        videos: dict[str, Any] = {}
        for i in range(sample_count):
            rows = []
            for num_steps, pixels in adapted_pixels:
                panels = [target_pixels[i]]
                if num_steps in base_pixels:
                    panels.append(base_pixels[num_steps][i])
                panels.append(pixels[i])
                rows.append(torch.cat(panels, dim=-1))
            grid = torch.cat(rows, dim=-2)
            cols = "gt | base | adapted" if base_pixels else "gt | adapted"
            order = ", ".join(f"N={n}" for n, _ in adapted_pixels)
            caption = f"sample={i} | rows top→bottom: {order} | cols: {cols}"
            if isinstance(actions, Tensor) and actions.shape[0] > i:
                caption += self._format_action_block(actions[i])
            videos[f"eval_step_grid/sample_{i}"] = self._wandb.Video(
                grid.numpy(),
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
        self._wandb.log(videos, step=int(step))

    def log_cond_frames_grid(
        self,
        *,
        target_latents: Tensor,
        rows: list[tuple[int | None, Tensor | None, list[tuple[int, Tensor]]]],
        cond: object | None,
        step: int,
    ) -> None:
        """Log one video per sample as a 2-D grid: **columns** sweep the number
        of clean observation (history) frames ``k``; **rows** sweep the sampling
        step count ``N``.

        ``rows`` is top→bottom; each entry is
        ``(num_steps, base_latents, adapted_by_k)`` — the row's step count (or
        ``None`` when there is no vertical sweep), a single base reference for
        that row (at the eval ``cond_frames``; ``None`` omits the base column),
        and one adapted latent per ``k`` (left→right). Within a row the columns
        are ``ground_truth | base | adapted@k1 | adapted@k2 | …`` concatenated
        along width; rows are stacked along height. The swept ``k`` (columns) and
        ``N`` (rows) are named in the caption. This is the diffusion-forcing
        analogue of :meth:`log_step_size_grid`: it shows how the prediction
        sharpens as more observed history is held clean, across step counts.
        """
        if self._decode_fn is None:
            raise RuntimeError("WandbLogger.log_cond_frames_grid requires a decode_fn (set at construction).")
        if not rows or not rows[0][2]:
            return
        sample_count = min(self.num_samples, target_latents.shape[0])
        target_pixels = self._decode_to_uint8(target_latents[:sample_count])
        decoded_rows = [
            (
                num_steps,
                self._decode_to_uint8(base[:sample_count]) if base is not None else None,
                [(k, self._decode_to_uint8(lat[:sample_count])) for k, lat in adapted_by_k],
            )
            for num_steps, base, adapted_by_k in rows
        ]
        actions = _maybe_extract_actions(cond)

        # Columns identical across rows; take names from the first row.
        ks_str = ", ".join(f"k={k}" for k, _ in decoded_rows[0][2])
        any_base = any(base is not None for _, base, _ in decoded_rows)
        prefix = "gt | base | " if any_base else "gt | "
        row_labels = [n for n, _, _ in decoded_rows if n is not None]

        videos: dict[str, Any] = {}
        for i in range(sample_count):
            strips = []
            for _, base_pixels, adapted_pixels in decoded_rows:
                panels = [target_pixels[i]]
                if base_pixels is not None:
                    panels.append(base_pixels[i])
                for _, pixels in adapted_pixels:
                    panels.append(pixels[i])
                strips.append(torch.cat(panels, dim=-1))
            grid = torch.cat(strips, dim=-2) if len(strips) > 1 else strips[0]
            caption = f"sample={i} | cols: {prefix}adapted@({ks_str})"
            if row_labels:
                caption += " | rows top→bottom: " + ", ".join(f"N={n}" for n in row_labels)
            if isinstance(actions, Tensor) and actions.shape[0] > i:
                caption += self._format_action_block(actions[i])
            videos[f"eval_cond_grid/sample_{i}"] = self._wandb.Video(
                grid.numpy(),
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
        return header + WandbLogger._format_action_block(actions[sample_index])

    @staticmethod
    def _format_action_block(sample_actions: Tensor) -> str:
        """Render a single sample's per-frame actions as an aligned text table,
        prefixed so it can be appended to any caption."""
        sample_actions = sample_actions.detach().cpu()
        rows = ["t " + " ".join(f"a{j}" for j in range(sample_actions.shape[-1]))]
        for frame_index in range(sample_actions.shape[0]):
            row = sample_actions[frame_index].tolist()
            rows.append(f"{frame_index} " + " ".join(f"{x:+.2f}" for x in row))
        return "\nactions:\n" + "\n".join(rows)


def _minmax(x: Tensor) -> Tensor:
    """Min-max normalise a 2-D tensor to [0, 1] for image display."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)


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
