"""JSONL metrics logger: one JSON object per line, scalar metrics only.

Mirrors :meth:`WandbLogger.log_metrics` but writes to a local file so offline
runs (or runs without wandb) keep a machine-readable training/eval history.
Non-scalar entries (e.g. the ``generated_samples`` tensor) are dropped, matching
the wandb path. Each record carries ``step``, ``split`` ("train"/"eval"), and a
``wall_time`` (seconds since the logger was constructed) alongside the metrics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from numbers import Number
from pathlib import Path

from torch import Tensor


class JsonlMetricsLogger:
    """Append-only JSON-lines sink for scalar training/eval metrics."""

    def __init__(self, path: str | Path, *, start_time: float | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append so a crashed run keeps everything up to the last
        # completed step.
        self._handle = self.path.open("a", encoding="utf-8")
        self._start = time.time() if start_time is None else float(start_time)

    def log(self, metrics: Mapping[str, object], *, step: int, split: str = "train") -> None:
        record: dict[str, object] = {
            "step": int(step),
            "split": split,
            "wall_time": round(time.time() - self._start, 3),
        }
        for key, value in metrics.items():
            scalar = _coerce_scalar(value)
            if scalar is not None:
                record[key] = scalar
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "JsonlMetricsLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _coerce_scalar(value: object) -> float | None:
    """Return a float for scalar-like inputs, ``None`` for everything else."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None
