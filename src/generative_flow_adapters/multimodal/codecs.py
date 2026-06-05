"""Per-modality codecs: raw clip data <-> diffusion target.

Sub-decision 2 of the multimodal substrate (latent-vs-raw per modality). A
:class:`ModalityCodec` turns a raw modality tensor from the dataset into the
tensor the diffusion process actually noises (``encode``), and back
(``decode``) for logging / downstream use.

- ``IdentityCodec`` — raw diffusion with optional per-channel normalisation.
  Used for proprio / tactile, and depth at bring-up. Cheap, no learned params.
- ``ResizeCodec`` — IdentityCodec + spatial resize (down for big maps like
  128x128 depth, restored on decode). Lossy on decode but keeps the head small.
- ``VaeCodec`` — wraps a frozen ``VideoAutoencoderKL``; used for the video
  stream (and optionally depth via 3-channel replicate). The VAE only exists at
  real-run time, so this is constructed in the builder, not unit tests.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor


@runtime_checkable
class ModalityCodec(Protocol):
    """Maps raw modality data to a diffusion target and back."""

    def encode(self, raw: Tensor) -> Tensor: ...

    def decode(self, target: Tensor) -> Tensor: ...


def _as_tensor(value: float | list[float] | Tensor | None, default: float) -> Tensor | float:
    if value is None:
        return default
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (list, tuple)):
        return torch.tensor([float(v) for v in value])
    return float(value)


class IdentityCodec:
    """Affine normalisation: ``target = (raw - mean) / std``.

    ``mean``/``std`` may be scalars or per-(last-dim-channel) vectors. With the
    defaults (0, 1) this is a pure pass-through, so a modality with no known
    statistics still works — the head just learns the raw scale.
    """

    def __init__(self, mean: float | list[float] | Tensor | None = None, std: float | list[float] | Tensor | None = None) -> None:
        self.mean = _as_tensor(mean, 0.0)
        self.std = _as_tensor(std, 1.0)

    def _broadcast(self, stat: Tensor | float, ref: Tensor) -> Tensor | float:
        if isinstance(stat, Tensor):
            return stat.to(device=ref.device, dtype=ref.dtype)
        return stat

    def encode(self, raw: Tensor) -> Tensor:
        mean = self._broadcast(self.mean, raw)
        std = self._broadcast(self.std, raw)
        return (raw - mean) / std

    def decode(self, target: Tensor) -> Tensor:
        mean = self._broadcast(self.mean, target)
        std = self._broadcast(self.std, target)
        return target * std + mean


class ResizeCodec(IdentityCodec):
    """IdentityCodec + spatial resize of map-shaped streams ``(..., H, W)``.

    ``size`` is the (h, w) the target is downsampled to for diffusion; decode
    restores the original spatial size (recorded on the first encode call).
    Bilinear, no antialias — these are sensor maps, not natural images.
    """

    def __init__(self, size: tuple[int, int], mean=None, std=None) -> None:
        super().__init__(mean=mean, std=std)
        self.size = (int(size[0]), int(size[1]))
        self._orig_size: tuple[int, int] | None = None

    @staticmethod
    def _flatten_spatial(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        # Collapse every leading axis into the batch dim for F.interpolate, which
        # wants (N, C, H, W). We treat the last two dims as spatial and the rest
        # (B, T, maybe channel) as batch with a singleton channel.
        *lead, h, w = x.shape
        return x.reshape(-1, 1, h, w), tuple(lead)

    def encode(self, raw: Tensor) -> Tensor:
        normed = super().encode(raw)
        self._orig_size = (int(normed.shape[-2]), int(normed.shape[-1]))
        flat, lead = self._flatten_spatial(normed)
        resized = F.interpolate(flat, size=self.size, mode="bilinear", align_corners=False)
        return resized.reshape(*lead, *self.size)

    def decode(self, target: Tensor) -> Tensor:
        if self._orig_size is None:
            raise RuntimeError("ResizeCodec.decode called before encode; original size unknown.")
        flat, lead = self._flatten_spatial(target)
        restored = F.interpolate(flat, size=self._orig_size, mode="bilinear", align_corners=False)
        restored = restored.reshape(*lead, *self._orig_size)
        return super().decode(restored)


class VaeCodec:
    """Wraps a frozen ``VideoAutoencoderKL`` for the video (or VAE-encoded) stream.

    ``encode`` expects pixel video in the encoder's convention (``(B, C, T, H, W)``
    in ``[-1, 1]``) and returns the scaled latent; ``decode`` inverts it. For a
    1-channel stream (depth) set ``replicate_to=3`` to fan it out to the VAE's
    RGB input and average back on decode.
    """

    def __init__(self, vae, replicate_to: int | None = None) -> None:
        self.vae = vae
        self.replicate_to = replicate_to

    def encode(self, raw: Tensor) -> Tensor:
        if self.replicate_to is not None and raw.shape[1] != self.replicate_to:
            raw = raw.repeat(1, self.replicate_to, *([1] * (raw.dim() - 2)))
        return self.vae.encode_video(raw)

    def decode(self, target: Tensor) -> Tensor:
        decoded = self.vae.decode_video(target) if hasattr(self.vae, "decode_video") else self.vae.decode(target)
        if self.replicate_to is not None:
            decoded = decoded.mean(dim=1, keepdim=True)
        return decoded


def build_codec(name: str, **kwargs) -> ModalityCodec:
    """Factory used by the multimodal builder. ``vae`` requires a ``vae=`` kwarg."""
    key = name.lower()
    if key == "identity":
        return IdentityCodec(mean=kwargs.get("mean"), std=kwargs.get("std"))
    if key == "resize":
        if "size" not in kwargs:
            raise ValueError("ResizeCodec requires a 'size': [h, w] in codec_kwargs.")
        return ResizeCodec(size=tuple(kwargs["size"]), mean=kwargs.get("mean"), std=kwargs.get("std"))
    if key == "vae":
        if "vae" not in kwargs:
            raise ValueError("VaeCodec requires a 'vae' instance (supplied by the builder, not YAML).")
        return VaeCodec(vae=kwargs["vae"], replicate_to=kwargs.get("replicate_to"))
    raise ValueError(f"Unknown codec {name!r}; expected 'identity', 'resize', or 'vae'.")
