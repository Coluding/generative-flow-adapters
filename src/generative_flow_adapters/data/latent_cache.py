"""Disk-backed cache of VAE-encoded clip latents.

The Wan VAE encode is the per-step training bottleneck (~3.3s at 768²) and it is
pure recomputation: the VAE is frozen and each clip's pixels are deterministic, so
``z0`` only needs to be computed once. This caches ``z0`` per clip on disk (and in
RAM), keyed on the clip's stable identity plus its resolution.

Key = ``env_name | episode_idx | start_idx | frame_stride | TxHxW`` — the fields
that fully determine the encoded pixels. Resolution is in the key (via ``TxHxW``),
so latents from different ``max_area`` settings coexist without collision.

Latents are stored as bf16 (the VAE now encodes in bf16 anyway) to halve disk/RAM;
callers ``.float()`` them on use.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor


def latent_key(env_name: object, episode_idx: object, start_idx: object,
               frame_stride: object, t: int, h: int, w: int) -> str:
    """Stable string identity for a clip's encoded latent (see module docstring)."""
    return f"{env_name}|{int(episode_idx)}|{int(start_idx)}|{int(frame_stride)}|{int(t)}x{int(h)}x{int(w)}"


class LatentCache:
    """RAM + disk cache of ``{key: z0}``. Disk files are ``<sha1(key)>.pt`` under
    ``cache_dir``; a small in-memory dict avoids re-reading within a run."""

    def __init__(self, cache_dir: str) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Tensor] = {}

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.pt"

    def get(self, key: str) -> Tensor | None:
        """Return the cached latent (RAM first, then disk), or None on a miss."""
        z = self._mem.get(key)
        if z is not None:
            return z
        path = self._path(key)
        if path.exists():
            z = torch.load(path, map_location="cpu", weights_only=True)
            self._mem[key] = z
            return z
        return None

    def put(self, key: str, z: Tensor) -> None:
        """Store ``z`` (as bf16 on CPU) in RAM and on disk. Atomic write; skips the
        disk write if the file already exists."""
        z = z.detach().to("cpu", dtype=torch.bfloat16)
        self._mem[key] = z
        path = self._path(key)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            torch.save(z, tmp)
            os.replace(tmp, path)  # atomic rename so a crash never leaves a partial file

    def __contains__(self, key: str) -> bool:
        return key in self._mem or self._path(key).exists()

    def num_files(self) -> int:
        return sum(1 for _ in self.dir.glob("*.pt"))
