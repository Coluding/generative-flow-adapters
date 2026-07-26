"""Checkpoint manager for adapter training.

Saves only the *trainable* weights (adapter + condition encoder) plus optimizer
state. The frozen base backbone is reconstructed from its own pretrained
checkpoint at build time, so writing it on every cadence would waste disk and
time for no benefit — hence :func:`trainable_state_dict`. Load back with
``strict=False`` so the rebuilt frozen base is left untouched; its keys show up
as "missing" and are expected.

Two checkpoint flavours:
  * step-tagged (``step_00012000.pt``) — written at a step cadence, rotated to
    the most recent K via ``keep_last``;
  * named (``best.pt`` / ``final.pt``) — written via ``tag=...`` and never
    rotated.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _probe_write_error(directory: Path) -> str | None:
    """Re-attempt a tiny write in ``directory`` to recover the real ``errno``.

    ``torch.save`` reports a failed write as its own ``PytorchStreamWriter`` /
    ``unexpected pos`` ``RuntimeError`` and discards the underlying ``OSError``,
    so a traceback never names the actual cause. A 4 KiB write plus ``fsync`` in
    the same directory brings ``EDQUOT`` / ``ENOSPC`` straight back. Note that
    ``shutil.disk_usage`` is useless here: on a quota'd shared filesystem the
    device reports terabytes free while the user's own quota is exhausted.
    """
    probe = directory / ".checkpoint_write_probe"
    try:
        with open(probe, "wb") as handle:
            handle.write(b"\0" * 4096)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno == errno.EDQUOT:
            return "disk quota exceeded"
        if exc.errno == errno.ENOSPC:
            return "no space left on device"
        return f"{exc.strerror} (errno {exc.errno})"
    finally:
        probe.unlink(missing_ok=True)
    return None


def _save_atomic(payload: dict[str, Any], path: Path) -> None:
    """``torch.save`` into a sibling temp file, then rename it into place.

    Saving straight to ``path`` means a write that dies partway — full disk,
    exhausted quota, killed job — leaves a truncated file where a checkpoint
    should be. That is worse than no checkpoint at all: a short ``best.pt`` is
    still a readable zip prefix, so the next run may load it without complaint.
    Staging through ``.tmp`` keeps the last good checkpoint intact, and the
    rename is atomic within a directory.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
    except (OSError, RuntimeError) as exc:
        tmp.unlink(missing_ok=True)
        cause = _probe_write_error(path.parent)
        detail = (
            f"the filesystem holding {path.parent} reports: {cause}"
            if cause
            else f"a follow-up test write to {path.parent} succeeded, so the cause is unclear"
        )
        raise RuntimeError(
            f"Failed writing checkpoint {path} ({type(exc).__name__}: {exc}) — {detail}. "
            f"If this is a quota problem, check it (`myquota`) and point "
            f"training.output_dir at scratch rather than $HOME."
        ) from exc
    os.replace(tmp, path)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """``state_dict`` restricted to parameters with ``requires_grad`` set.

    Buffers and frozen-base parameters are intentionally excluded; pair with a
    ``load_state_dict(..., strict=False)`` so the rebuilt model supplies them.
    """
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {key: value for key, value in model.state_dict().items() if key in trainable}


class CheckpointManager:
    def __init__(self, directory: str | Path, *, keep_last: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._step_checkpoints: list[Path] = []

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        extra: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> Path:
        """Write a checkpoint and return its path.

        ``tag`` writes ``<tag>.pt`` (overwriting any previous one, never
        rotated); otherwise a step-tagged file subject to ``keep_last`` rotation.
        """
        payload: dict[str, Any] = {
            "global_step": int(global_step),
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
        }
        if extra:
            payload["extra"] = dict(extra)

        if tag is not None:
            path = self.directory / f"{tag}.pt"
        else:
            path = self.directory / f"step_{int(global_step):08d}.pt"
        _save_atomic(payload, path)

        if tag is None:
            self._step_checkpoints.append(path)
            self._rotate()
        return path

    def _rotate(self) -> None:
        if not self.keep_last or self.keep_last <= 0:
            return
        while len(self._step_checkpoints) > self.keep_last:
            stale = self._step_checkpoints.pop(0)
            stale.unlink(missing_ok=True)

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load trainable weights (``strict=False``) and optionally optimizer
        state. Returns the raw payload so callers can restore ``global_step`` /
        best-metric bookkeeping from ``payload["extra"]``."""
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model"], strict=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
