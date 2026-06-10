"""Output adapter with a configurable backbone and a switchable output format.

This adapter exists to run the output-format ablation described in
``format.py``: a single adapter whose *backbone* (``mlp`` | ``transformer``)
and *output format* (``direct`` | ``affine``) are independent knobs, so a run
isolates the effect of the format from the effect of capacity. The ``unet``
backbone of the same experiment is served by
``DynamicCrafterOutputAdapter`` (it already wraps a full 3D UNet); the
factory routes ``backbone: unet`` there.

All backbones zero-initialise their final projection, so the adapter is the
identity residual at step 0 (delta ≈ 0 → base output unchanged), matching the
"identity at init" guarantee of the rest of the output family.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.common import resolve_condition_embedding
from generative_flow_adapters.adapters.output.format import (
    build_output_result,
    normalize_output_format,
    output_channel_multiplier,
    prepare_gate,
)
from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult
from generative_flow_adapters.utils import timestep_embedding

_GATE_KINDS = {"none", "channel", "dense"}


class OutputHeadAdapter(OutputAdapterInterface):
    """Output adapter with a pluggable backbone and ``direct``/``affine`` format."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int,
        backbone: str = "mlp",
        output_format: str = "affine",
        gate_kind: str = "none",
        condition_on_base_outputs: bool = True,
        patch_size: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if feature_dim is None or feature_dim <= 0:
            raise ValueError("OutputHeadAdapter requires a positive feature_dim (the latent channel count).")
        self.feature_dim = int(feature_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone_kind = str(backbone).strip().lower()
        self.output_format = normalize_output_format(output_format)
        self.gate_kind = str(gate_kind).strip().lower()
        if self.gate_kind not in _GATE_KINDS:
            raise ValueError(f"Unsupported gate_kind: {gate_kind!r} (expected one of {sorted(_GATE_KINDS)}).")
        self.condition_on_base_outputs = bool(condition_on_base_outputs)

        multiplier = output_channel_multiplier(self.output_format)
        # Channels carrying the delta/format, plus (optionally) one gate channel
        # per latent channel appended after them for gated_residual/mask_mix.
        self.format_channels = self.feature_dim * multiplier
        self.gate_channels = self.feature_dim if self.gate_kind != "none" else 0
        self.out_channels = self.format_channels + self.gate_channels

        if self.backbone_kind == "mlp":
            self.backbone = _MLPBackbone(
                feature_dim=self.feature_dim,
                cond_dim=self.cond_dim,
                hidden_dim=self.hidden_dim,
                out_channels=self.out_channels,
            )
        elif self.backbone_kind == "transformer":
            in_channels = self.feature_dim * (2 if self.condition_on_base_outputs else 1)
            self.backbone = _VideoTransformerBackbone(
                in_channels=in_channels,
                out_channels=self.out_channels,
                cond_dim=self.cond_dim,
                hidden_dim=self.hidden_dim,
                patch_size=int(patch_size),
                num_layers=int(num_layers),
                num_heads=int(num_heads),
            )
        else:
            raise ValueError(
                f"Unsupported OutputHeadAdapter backbone: {backbone!r} (expected 'mlp' or 'transformer'; "
                f"the 'unet' backbone is served by DynamicCrafterOutputAdapter)."
            )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        reference = base_output if base_output is not None else x_t
        cond_embedding = resolve_condition_embedding(cond)
        raw = self.backbone(x_t, t, cond_embedding, base_output=base_output)

        gate = None
        if self.gate_channels:
            raw, gate_raw = torch.split(raw, [self.format_channels, self.gate_channels], dim=1)
            gate = prepare_gate(gate_raw, reference, granularity=self.gate_kind, channel_dim=1)

        return build_output_result(
            raw,
            reference,
            output_format=self.output_format,
            gate=gate,
        )


class _MLPBackbone(nn.Module):
    """Lightweight (t, cond)-driven head — the global per-channel baseline.

    Mirrors the original ``AffineOutputAdapter``: scale/shift (or the direct
    delta) depend only on the timestep and condition, broadcast over space and
    time. Granularity is implicitly ``channel`` regardless of the configured
    value, since this backbone has no spatial extent to vary over.
    """

    def __init__(self, feature_dim: int, cond_dim: int, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, out_channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t: Tensor, t: Tensor, cond_embedding: Tensor | None, base_output: Tensor | None = None) -> Tensor:
        del x_t, base_output
        time_features = timestep_embedding(t, self.hidden_dim)
        if cond_embedding is None:
            cond_embedding = torch.zeros(time_features.shape[0], self.cond_dim, device=time_features.device, dtype=time_features.dtype)
        elif cond_embedding.dim() == 3:
            # Per-frame conditioning [B, T, cond_dim] → pool to a global vector.
            cond_embedding = cond_embedding.mean(dim=1)
        context = self.net(torch.cat([time_features, cond_embedding.to(dtype=time_features.dtype)], dim=-1))
        return self.head(context)  # [B, out_channels]; build_output_result broadcasts over space/time


class _VideoTransformerBackbone(nn.Module):
    """DiT-style transformer over patchified video latents.

    Patchifies the spatial dims of a ``[B, C, T, H, W]`` latent (temporal axis
    left intact), runs a Transformer encoder over the flattened token grid
    with an added (time + condition) token, then un-patchifies to
    ``[B, out_channels, T, H, W]``. The un-patch projection is zero-initialised
    so the backbone outputs zeros at init.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        hidden_dim: int,
        patch_size: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.patch_embed = nn.Conv3d(
            in_channels, hidden_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.unpatch = nn.ConvTranspose3d(
            hidden_dim, out_channels, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size)
        )
        nn.init.zeros_(self.unpatch.weight)
        nn.init.zeros_(self.unpatch.bias)

    def forward(self, x_t: Tensor, t: Tensor, cond_embedding: Tensor | None, base_output: Tensor | None = None) -> Tensor:
        if x_t.dim() != 5:
            raise ValueError("transformer backbone expects video latents shaped [batch, channels, frames, height, width].")
        batch, channels, frames, height, width = x_t.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"patch_size {self.patch_size} must divide the latent height/width ({height}x{width}).")

        x = x_t
        if self.in_channels == 2 * channels:
            # condition_on_base_outputs: concat the (detached) base output on channels.
            ref = base_output if base_output is not None else torch.zeros_like(x_t)
            x = torch.cat([x_t, ref.to(dtype=x_t.dtype)], dim=1)

        tokens = self.patch_embed(x.to(dtype=self.patch_embed.weight.dtype))  # [B, hidden, T, H', W']
        _, hidden, t_dim, h_dim, w_dim = tokens.shape
        seq = tokens.flatten(2).transpose(1, 2)  # [B, N, hidden], N = T*H'*W'

        position = _sinusoidal_position_embeddings(seq.shape[1], hidden).to(device=seq.device, dtype=seq.dtype)
        seq = seq + position.unsqueeze(0)

        conditioning = timestep_embedding(t, hidden).to(dtype=seq.dtype)  # [B, hidden]
        if self.cond_proj is not None and cond_embedding is not None:
            ce = cond_embedding
            if ce.dim() == 3:
                ce = ce.mean(dim=1)
            conditioning = conditioning + self.cond_proj(ce.to(dtype=self.cond_proj.weight.dtype)).to(dtype=seq.dtype)
        seq = seq + conditioning.unsqueeze(1)

        seq = self.encoder(seq)
        grid = seq.transpose(1, 2).reshape(batch, hidden, t_dim, h_dim, w_dim)
        out = self.unpatch(grid)  # [B, out_channels, T, H, W]
        return out.to(dtype=x_t.dtype)


def _sinusoidal_position_embeddings(length: int, dim: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("Sinusoidal position embeddings require an even feature dimension.")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    embeddings = torch.zeros(length, dim, dtype=torch.float32)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings
