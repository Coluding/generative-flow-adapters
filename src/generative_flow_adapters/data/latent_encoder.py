"""First-party video autoencoder used to bridge pixel space and latent space.

This wraps the vendored Stable Diffusion ``Encoder`` / ``Decoder`` building
blocks at ``external_deps.lvdm.modules.networks.ae_modules`` and applies the
constant ``scale_factor = 0.18215`` used by DynamiCrafter / Stable Diffusion
to keep latent statistics near unit variance.

We deliberately do NOT inherit from the vendored ``AutoencoderKL`` Lightning
module — we only need the encode / decode path, and inheriting would pull
``pytorch_lightning`` and ``instantiate_from_config`` into our import graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

import torch
from einops import rearrange
from torch import Tensor, nn

# Standard Stable Diffusion 2.x VAE config (matches DynamiCrafter's
# `first_stage_config` in `dynamicrafter_512.yaml`).
SD_VAE_DDCONFIG: dict[str, Any] = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}

# Standard SD scale factor. Stored separately so callers can override.
SD_VAE_SCALE_FACTOR: float = 0.18215


class VideoAutoencoderKL(nn.Module):
    """Per-frame KL autoencoder for video tensors.

    Encode: ``(B, 3, T, H, W) float in [-1, 1] -> (B, embed_dim, T, H/8, W/8) float``.
    Decode: inverse, returning pixels in roughly ``[-1, 1]``.
    """

    def __init__(
        self,
        ddconfig: dict[str, Any] | None = None,
        embed_dim: int = 4,
        scale_factor: float = SD_VAE_SCALE_FACTOR,
        perframe: bool = True,
        sample_posterior: bool = True,
    ) -> None:
        super().__init__()
        from external_deps.lvdm.distributions import DiagonalGaussianDistribution
        from external_deps.lvdm.modules.networks.ae_modules import Decoder, Encoder

        if ddconfig is None:
            ddconfig = dict(SD_VAE_DDCONFIG)
        if not ddconfig.get("double_z", False):
            raise ValueError("VideoAutoencoderKL requires ddconfig['double_z']=True.")

        self.ddconfig = ddconfig
        self.embed_dim = int(embed_dim)
        self.scale_factor = float(scale_factor)
        self.perframe = bool(perframe)
        self.sample_posterior = bool(sample_posterior)

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * self.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
        self._distribution = DiagonalGaussianDistribution

    @torch.no_grad()
    def encode_video(self, video: Tensor) -> Tensor:
        if video.dim() != 5:
            raise ValueError(f"encode_video expects 5D (B, C, T, H, W); got shape {tuple(video.shape)}.")
        batch, _channels, frames, _h, _w = video.shape
        flat = rearrange(video, "b c t h w -> (b t) c h w")
        z_flat = self._run_codec(flat, encode=True)
        z = rearrange(z_flat, "(b t) c h w -> b c t h w", b=batch, t=frames)
        return self.scale_factor * z

    @torch.no_grad()
    def decode_video(self, latent: Tensor) -> Tensor:
        if latent.dim() != 5:
            raise ValueError(f"decode_video expects 5D (B, C, T, H, W); got shape {tuple(latent.shape)}.")
        batch, _channels, frames, _h, _w = latent.shape
        flat = rearrange(latent / self.scale_factor, "b c t h w -> (b t) c h w")
        x_flat = self._run_codec(flat, encode=False)
        return rearrange(x_flat, "(b t) c h w -> b c t h w", b=batch, t=frames)

    def _run_codec(self, flat: Tensor, *, encode: bool) -> Tensor:
        op = self._encode_frame if encode else self._decode_frame
        if not self.perframe:
            return op(flat)
        chunks = [op(flat[i : i + 1]) for i in range(flat.shape[0])]
        return torch.cat(chunks, dim=0)

    def _encode_frame(self, frame: Tensor) -> Tensor:
        moments = self.quant_conv(self.encoder(frame))
        posterior = self._distribution(moments)
        return posterior.sample() if self.sample_posterior else posterior.mode()

    def _decode_frame(self, latent: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(latent))

    def load_dynamicrafter_checkpoint(self, path: str | Path, strict: bool = False) -> list[str]:
        """Load VAE weights from a DynamiCrafter checkpoint.

        Filters the ``first_stage_model.*`` prefix (with or without a
        leading ``model.``) from the full DynamiCrafter state dict. Returns
        the list of keys we ended up loading, for debugging.
        """
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, MutableMapping) else state
        if not isinstance(state_dict, MutableMapping):
            raise TypeError(f"Could not interpret checkpoint at {path}.")

        prefixes = ("model.first_stage_model.", "first_stage_model.")
        filtered: dict[str, Tensor] = {}
        for prefix in prefixes:
            candidates = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if candidates:
                filtered = candidates
                break
        if not filtered:
            # Possibly a standalone VAE checkpoint (already at root).
            filtered = dict(state_dict)

        self.load_state_dict(filtered, strict=strict)
        return list(filtered.keys())
