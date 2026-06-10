"""Multimodal batch preprocessor.

Turns a raw clip batch (uint8 video + per-frame proprio/depth/tactile/...) into
the multi-stream shape the multimodal trainer expects::

    {
        "targets": {"video": z, "proprio": p, "tactile": ta, "depth": d},
        "cond":    {... shared conditioning (context/concat/act/fs/...) ...},
    }

The video stream is produced by an optional inner ``video_preprocessor`` (the
real :class:`DynamiCrafterBatchPreprocessor`, which VAE-encodes pixels and
builds the DynamiCrafter conditioning). When it is ``None`` (dummy/test path)
the video stream is treated like any other: its raw tensor is codec-encoded and
conditioning is plucked from ``condition_keys``. Every non-video modality is
codec-encoded into its diffusion target.

Item 3 of the shared substrate in the multimodal decision note.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from generative_flow_adapters.multimodal.codecs import ModalityCodec
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


class MultiModalBatchPreprocessor:
    def __init__(
        self,
        modalities: list[OutputModalitySpec],
        codecs: Mapping[str, ModalityCodec],
        *,
        video_preprocessor: Callable[..., Mapping[str, Any]] | None = None,
        condition_keys: tuple[str, ...] = ("act",),
    ) -> None:
        self.modalities = list(modalities)
        self.codecs = dict(codecs)
        self.video_preprocessor = video_preprocessor
        self.condition_keys = tuple(condition_keys)

        videos = [m for m in self.modalities if m.has_frozen_prior]
        if len(videos) != 1:
            raise ValueError(f"Expected exactly one has_frozen_prior modality; got {len(videos)}.")
        self.video_spec = videos[0]
        self.adapter_specs = [m for m in self.modalities if not m.has_frozen_prior]
        missing = [m.name for m in self.modalities if m.name not in self.codecs]
        if missing:
            raise ValueError(f"No codec supplied for modalities: {missing}")

    def __call__(self, batch: Mapping[str, Any], train: bool = True) -> dict[str, Any]:
        targets: dict[str, Tensor] = {}

        if self.video_preprocessor is not None:
            video_out = self._call_video(batch, train=train)
            video_target = video_out["target"]
            cond = dict(video_out["cond"])
        else:
            raw_video = self._require(batch, self.video_spec.name)
            video_target = self.codecs[self.video_spec.name].encode(raw_video)
            cond = self._cond_from_keys(batch, device=video_target.device, dtype=video_target.dtype)

        targets[self.video_spec.name] = video_target
        device, dtype = video_target.device, video_target.dtype

        for spec in self.adapter_specs:
            raw = self._require(batch, spec.name)
            raw = raw.to(device=device, dtype=dtype) if isinstance(raw, Tensor) else raw
            targets[spec.name] = self.codecs[spec.name].encode(raw)

        return {"targets": targets, "cond": cond}

    def _call_video(self, batch: Mapping[str, Any], *, train: bool) -> Mapping[str, Any]:
        try:
            return self.video_preprocessor(batch, train=train)  # type: ignore[misc]
        except TypeError:
            return self.video_preprocessor(batch)  # type: ignore[misc]

    def _cond_from_keys(self, batch: Mapping[str, Any], *, device, dtype) -> dict[str, Tensor]:
        cond: dict[str, Tensor] = {}
        for key in self.condition_keys:
            value = batch.get(key)
            if isinstance(value, Tensor):
                cond[key] = value.to(device=device, dtype=dtype)
        return cond

    @staticmethod
    def _require(batch: Mapping[str, Any], name: str) -> Tensor:
        value = batch.get(name)
        if not isinstance(value, Tensor):
            raise KeyError(f"MultiModalBatchPreprocessor: batch is missing tensor modality {name!r}.")
        return value
