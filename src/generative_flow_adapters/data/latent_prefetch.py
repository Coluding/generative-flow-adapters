"""Read precomputed VAE latents in the DataLoader **workers**, not the train loop.

The latent cache (``data/latent_cache.py``) already removes the VAE encode from
the training step, but the *read* stayed in the main process: the preprocessor
looked each clip up on a cache hit and pulled it off disk synchronously, between
the forward passes. At Wan2.2 robot-arm geometry one latent is
``[48, 25, 42, 54]`` bf16 ≈ 5.4 MB, so a batch of 12 is ~65 MB of GPFS reads that
nothing overlaps — measured at **2.0-2.9 s per micro-step** (2026-07-29, H100
interactive node), second only to the 5B base forward itself.

Worse, the frames were decoded anyway just to be thrown away: ``__getitem__``
pulled 97 mp4 frames at 480x640 (~89 MB/sample) through decord and the collate,
and ``_encode_z0`` discarded them the moment the cache hit.

:class:`LatentPrefetchDataset` fixes both. It resolves the window's identity
first, builds the same cache key the preprocessor would, and on a hit returns the
latent with **no decode at all**; on a miss it falls back to the normal pixel
path so a cold or partial cache still trains. Because this happens inside
``__getitem__``, ``num_workers`` copies of it run in parallel and prefetch ahead
of the GPU — the read cost disappears behind compute instead of adding to it.

The batch it emits is deliberately shaped like the pixel batch (same identity
fields, same ``act``), plus ``z0`` — see
:func:`generative_flow_adapters.data.latent_prefetch.collate_latent_windows` for
the mixed hit/miss batch layout the preprocessor consumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from generative_flow_adapters.data.dataset import TranslatedClipDataset
from generative_flow_adapters.data.latent_cache import LatentCache, latent_key


class LatentPrefetchDataset(Dataset):
    """Wraps a :class:`TranslatedClipDataset` so cached windows skip the decode.

    Args:
        base: the clip dataset to wrap. Must expose ``resolve(idx)``.
        cache_dir: the latent cache directory the preprocessor uses.
        output_hw: the *resized* ``(H, W)`` the preprocessor would encode at
            (``WanBatchPreprocessor._output_hw`` of the source frames). Part of
            the cache key, so it must match what precompute wrote.
        keep_ram_cache: keep latents in the worker's RAM dict after reading. Off
            by default: with K windows per episode over thousands of episodes the
            working set is far larger than RAM, and each worker would hold its own
            copy.
    """

    def __init__(
        self,
        base: TranslatedClipDataset,
        *,
        cache_dir: str,
        output_hw: tuple[int, int],
        keep_ram_cache: bool = False,
    ) -> None:
        if not hasattr(base, "resolve"):
            raise TypeError(
                f"LatentPrefetchDataset needs a dataset exposing resolve(idx); got {type(base).__name__}."
            )
        self.base = base
        self.cache_dir = str(cache_dir)
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.keep_ram_cache = bool(keep_ram_cache)
        self._cache: LatentCache | None = None  # opened lazily, per worker process

    def __len__(self) -> int:
        return len(self.base)

    def _latent_cache(self) -> LatentCache:
        if self._cache is None:
            self._cache = LatentCache(self.cache_dir)
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, start = self.base.resolve(idx)
        length, stride = self.base.window_width, self.base.frame_stride
        translator = self.base.translator

        meta = translator.load_clip_meta(ep, start, length, stride)
        if meta is not None:
            out_h, out_w = self.output_hw
            key = latent_key(
                meta.get("env_name"), meta.get("episode_idx"), meta.get("start_idx"),
                meta.get("frame_stride"), length, out_h, out_w,
            )
            cache = self._latent_cache()
            z0 = cache.get(key)
            if z0 is not None:
                if not self.keep_ram_cache:
                    cache._mem.pop(key, None)
                meta.pop("source_hw", None)
                return {"z0": z0, **meta}

        # Miss (or a translator with no metadata-only path): decode as usual. The
        # preprocessor VAE-encodes these and writes them back to the cache.
        return self.base.translator.load_clip(ep, start=start, length=length, stride=stride)

    def __getattr__(self, name: str) -> Any:
        # Transparent passthrough for the attributes callers read off the clip
        # dataset (`sampling`, `translator`, `episodes`, `fixed_window_enumeration`,
        # ...). Only consulted for names this wrapper does not define itself.
        #
        # Both guards matter under the spawn-context DataLoader: unpickling probes
        # `__setstate__`/`__reduce_ex__` and reads attributes *before* `__dict__` is
        # populated, so a naive passthrough raises KeyError (not AttributeError)
        # and aborts every worker.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)


def collate_latent_windows(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch that may mix latent hits and pixel misses.

    A hit carries ``z0`` and no ``video``; a miss carries ``video`` and no ``z0``.
    Workers decide independently, so a batch can contain both and
    ``default_collate`` (which requires identical keys) would reject it. The
    result keeps the shared fields stacked as usual and adds:

    * ``z0`` — ``[B, C, T, h, w]`` stacked, when **every** item hit. This is the
      steady-state shape on a warm cache and the one the preprocessor's fast path
      wants.
    * ``z0_list`` / ``video_list`` — per-sample ``Tensor | None``, present only in
      the mixed case, so the preprocessor can encode just the misses.
    """
    hits = [it for it in items if "z0" in it]
    shared_keys = {k for it in items for k in it} - {"z0", "video"}
    batch: dict[str, Any] = default_collate(
        [{k: it[k] for k in shared_keys} for it in items]
    )

    if len(hits) == len(items):
        batch["z0"] = torch.stack([it["z0"] for it in items], dim=0)
        return batch
    if not hits:
        batch["video"] = default_collate([it["video"] for it in items])
        return batch
    batch["z0_list"] = [it.get("z0") for it in items]
    batch["video_list"] = [
        (None if "z0" in it else torch.as_tensor(it["video"])) for it in items
    ]
    return batch
