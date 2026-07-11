"""Paper-standard generative-visual quality metrics for training-time eval.

Self-contained implementation over ``torchmetrics`` (PSNR / SSIM / LPIPS / FID)
and ``cd-fvd`` (FVD). Deliberately does **not** import from ``external_deps``
(hard rule) — the vendored AVID ``metrics.py`` is not touched.

* **Paired, per-frame vs. ground truth** — ``psnr``, ``ssim``, ``lpips``,
  ``mse``. Reliable on small eval batches because the world-model eval has
  aligned ground-truth future frames. Cheap; run every eval cycle.
* **Distribution** — ``fid`` (per-frame Inception features) and ``fvd`` (video
  features, i3d / videomae). Need a feature extractor and many samples to be
  meaningful, so they are gated behind their own rarer cadence (see
  ``TrainingConfig.quality_dist_*``). Both accumulate correctly across batches
  via the underlying libraries' running feature stats.

Decoded pixels enter as uint8 ``(B, T, 3, H, W)`` — the layout the
``WandbLogger`` decoder emits. Image metrics flatten to ``(B*T, 3, H, W)``
frames in ``[0, 1]``; FVD takes clips as uint8 ``(B, T, H, W, C)``.

See thesis-vault ``20_Tickets/feat-eval-training-quality-metrics.md`` and
``30_Knowledge/tech/generative-visual-metrics.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

PAIRED_METRICS = frozenset({"psnr", "ssim", "lpips", "mse"})
DISTRIBUTION_METRICS = frozenset({"fid", "fvd"})
SUPPORTED_METRICS = PAIRED_METRICS | DISTRIBUTION_METRICS

# FID's Inception features need a minimum spatial size; the video nets in cd-fvd
# switch to VideoMAE below this frame count (i3d is unreliable on very short
# clips), matching common FVD practice.
_FVD_I3D_MIN_FRAMES = 10

# LPIPS (VGG16) and FID (Inception) run a full CNN over every frame, so feeding
# all ``B*T`` frames of an eval batch in one forward pass materialises huge
# activation tensors and OOMs the GPU. Both metrics accumulate running state, so
# we push frames through in sub-batches of this many frames — mathematically
# identical to a single pass, bounded memory. Set to 0 to disable chunking.
DEFAULT_FRAME_CHUNK = 16


def _frame_chunks(n: int, chunk: int):
    """Yield ``(start, stop)`` slices covering ``[0, n)`` in ``chunk``-sized
    steps. ``chunk <= 0`` yields a single full-range slice (chunking disabled)."""
    if chunk <= 0 or n <= chunk:
        yield 0, n
        return
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


@dataclass
class _Batch:
    """One update's pixels in the representations the metrics consume.

    ``frames01`` — ``(B*T, 3, H, W)`` float in ``[0, 1]`` on the metric device,
    for the torchmetrics image metrics. ``pred_pixels`` / ``real_pixels`` —
    the raw uint8 ``(B, T, 3, H, W)`` tensors, from which FVD builds its
    channels-last clip arrays only when it needs them.
    """

    pred_frames01: Tensor
    real_frames01: Tensor
    pred_pixels: Tensor
    real_pixels: Tensor


class _MSE:
    """Mean squared pixel error in ``[0, 1]`` space, accumulated over frames."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sq_sum = 0.0
        self._count = 0

    def update(self, batch: _Batch) -> None:
        err = (batch.pred_frames01 - batch.real_frames01) ** 2
        self._sq_sum += float(err.sum().item())
        self._count += err.numel()

    def compute(self, prefix: str) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {f"{prefix}/mse": self._sq_sum / self._count}


class _TorchmetricPaired:
    """Wrap a torchmetrics metric with an ``update(preds, target)`` signature
    over ``[0, 1]`` frames (PSNR, SSIM, LPIPS)."""

    def __init__(self, metric, key: str, *, frame_chunk: int = DEFAULT_FRAME_CHUNK) -> None:
        self._metric = metric
        self._key = key
        self._frame_chunk = frame_chunk
        self._updated = False

    def reset(self) -> None:
        self._metric.reset()
        self._updated = False

    def update(self, batch: _Batch) -> None:
        pred, real = batch.pred_frames01, batch.real_frames01
        for start, stop in _frame_chunks(pred.shape[0], self._frame_chunk):
            self._metric.update(pred[start:stop], real[start:stop])
        self._updated = True

    def compute(self, prefix: str) -> dict[str, float]:
        # No successful update this cycle (e.g. every eval batch OOM'd before it
        # reached this metric): torchmetrics.compute() would raise on empty state.
        if not self._updated:
            return {}
        return {f"{prefix}/{self._key}": float(self._metric.compute().item())}


class _FID:
    """Fréchet Inception Distance over frames treated as independent images.

    Accumulates real/fake Inception features across batches; ``normalize=True``
    accepts float pixels in ``[0, 1]``.
    """

    def __init__(self, device: torch.device, *, frame_chunk: int = DEFAULT_FRAME_CHUNK) -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance

        self._metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self._frame_chunk = frame_chunk
        self._updated = False

    def reset(self) -> None:
        self._metric.reset()
        self._updated = False

    def update(self, batch: _Batch) -> None:
        real, pred = batch.real_frames01, batch.pred_frames01
        for start, stop in _frame_chunks(real.shape[0], self._frame_chunk):
            self._metric.update(real[start:stop], real=True)
        for start, stop in _frame_chunks(pred.shape[0], self._frame_chunk):
            self._metric.update(pred[start:stop], real=False)
        self._updated = True

    def compute(self, prefix: str) -> dict[str, float]:
        if not self._updated:
            return {}
        return {f"{prefix}/fid": float(self._metric.compute().item())}


class _FVD:
    """Fréchet Video Distance via cd-fvd.

    Uses the library's running feature stats (``add_real_stats`` /
    ``add_fake_stats`` → ``compute_fvd_from_stats``) so the score reflects **all**
    batches seen this cycle, not just the last one. Backbone is i3d for clips of
    ``>= _FVD_I3D_MIN_FRAMES`` frames, else VideoMAE.
    """

    def __init__(self, device: torch.device, num_video_frames: int) -> None:
        from cdfvd import fvd

        self.model_name = "i3d" if num_video_frames >= _FVD_I3D_MIN_FRAMES else "videomae"
        self._evaluator = fvd.cdfvd(self.model_name, device=str(device))
        self.reset()

    def reset(self) -> None:
        self._evaluator.empty_real_stats()
        self._evaluator.empty_fake_stats()
        self._updated = False

    @staticmethod
    def _to_clip_uint8(pixels: Tensor):
        # (B, T, 3, H, W) uint8 -> (B, T, H, W, C) uint8 numpy, as cd-fvd expects.
        return pixels.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()

    def update(self, batch: _Batch) -> None:
        self._evaluator.add_real_stats(self._to_clip_uint8(batch.real_pixels))
        self._evaluator.add_fake_stats(self._to_clip_uint8(batch.pred_pixels))
        self._updated = True

    def compute(self, prefix: str) -> dict[str, float]:
        if not self._updated:
            return {}
        return {f"{prefix}/fvd_{self.model_name}": float(self._evaluator.compute_fvd_from_stats())}


def _build_metric(name: str, *, device: torch.device, num_video_frames: int, frame_chunk: int):
    """Instantiate one metric. Heavy backends (Inception via torchmetrics[image],
    i3d/VideoMAE via cd-fvd) are imported inside the wrappers so nothing is
    loaded at trainer-import time — only when quality metrics are switched on."""
    if name == "mse":
        return _MSE()
    if name == "psnr":
        from torchmetrics.image import PeakSignalNoiseRatio

        return _TorchmetricPaired(
            PeakSignalNoiseRatio(data_range=1.0).to(device), "psnr", frame_chunk=frame_chunk
        )
    if name == "ssim":
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        return _TorchmetricPaired(
            StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            "ssim",
            frame_chunk=frame_chunk,
        )
    if name == "lpips":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # net_type='vgg' + normalize=True → accepts [0,1] frames (mapped to
        # [-1,1] internally), matching the paired [0,1] representation.
        return _TorchmetricPaired(
            LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device),
            "lpips",
            frame_chunk=frame_chunk,
        )
    if name == "fid":
        return _FID(device, frame_chunk=frame_chunk)
    if name == "fvd":
        return _FVD(device, num_video_frames)
    raise ValueError(f"Unknown quality metric {name!r}; supported: {sorted(SUPPORTED_METRICS)}")


class QualityMetricSuite:
    """Accumulate a set of quality metrics over (predicted, real) pixel pairs.

    Stateless between runs: call :meth:`update` per eval batch, then
    :meth:`compute` once at the end of the cycle. A fresh suite starts empty,
    so no explicit reset is needed for single-cycle use.
    """

    def __init__(
        self,
        metric_names,
        *,
        device: torch.device,
        num_video_frames: int,
        frame_chunk: int = DEFAULT_FRAME_CHUNK,
    ):
        unknown = [m for m in metric_names if m not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"Unsupported quality metric(s) {unknown}; supported: {sorted(SUPPORTED_METRICS)}"
            )
        self.device = device
        self.metric_names = list(dict.fromkeys(metric_names))  # de-dup, keep order
        self._metrics = [
            _build_metric(
                name, device=device, num_video_frames=num_video_frames, frame_chunk=frame_chunk
            )
            for name in self.metric_names
        ]

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    @staticmethod
    def _validate(pixels: Tensor) -> None:
        if pixels.dim() != 5 or pixels.shape[2] != 3:
            raise ValueError(f"Expected decoded pixels (B, T, 3, H, W); got {tuple(pixels.shape)}.")
        if pixels.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 pixels; got {pixels.dtype}.")

    def _frames01(self, pixels: Tensor) -> Tensor:
        """``(B, T, 3, H, W)`` uint8 → ``(B*T, 3, H, W)`` float ``[0,1]`` on device."""
        b, t, c, h, w = pixels.shape
        return (pixels.reshape(b * t, c, h, w).float() / 255.0).to(self.device)

    @torch.no_grad()
    def update(self, pred_pixels: Tensor, real_pixels: Tensor) -> None:
        self._validate(pred_pixels)
        self._validate(real_pixels)
        batch = _Batch(
            pred_frames01=self._frames01(pred_pixels),
            real_frames01=self._frames01(real_pixels),
            pred_pixels=pred_pixels,
            real_pixels=real_pixels,
        )

        for metric, name in zip(self._metrics, self.metric_names):
            try:
                metric.update(batch)
            except torch.cuda.OutOfMemoryError:
                # One metric OOM'ing must not sink the whole eval cycle; skip its
                # contribution this batch (compute() tolerates never-updated
                # metrics). Free the partial allocation before the next metric.
                torch.cuda.empty_cache()
                warnings.warn(
                    f"Quality metric {name!r} ran out of GPU memory on an eval batch of "
                    f"{batch.pred_frames01.shape[0]} frames and was skipped for it. "
                    f"Lower the eval batch size or DEFAULT_FRAME_CHUNK.",
                    stacklevel=2,
                )

    def compute(self, prefix: str) -> dict[str, float]:
        results: dict[str, float] = {}
        for metric in self._metrics:
            results.update(metric.compute(prefix))
        return results
