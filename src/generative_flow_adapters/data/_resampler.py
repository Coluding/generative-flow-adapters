"""DynamiCrafter image-projection Resampler.

Standalone vendored copy (no `external_deps` dependency) of the Perceiver-style
resampler DynamiCrafter uses to turn a CLIP-image token sequence into a small
set of cross-attention tokens. Parameter names (``latents``, ``proj_in``,
``proj_out``, ``norm_out``, ``layers.<i>.0/.1.{...}``) are preserved so the
checkpoint's ``image_proj_model.*`` weights load with ``strict=True``.

Originally adapted by DynamiCrafter from open_flamingo / imagen-pytorch /
IP-Adapter; reproduced here so we don't drag the vendored lvdm tree along.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    # [B, L, H*D_h] -> [B, H, L, D_h]
    bs, length, _ = x.shape
    return x.view(bs, length, heads, -1).transpose(1, 2)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q = _split_heads(q, self.heads)
        k = _split_heads(k, self.heads)
        v = _split_heads(v, self.heads)

        # Pre-scale by sqrt(sqrt(d_head)) on both sides for fp16 stability —
        # matches the original DynamiCrafter implementation.
        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embedding_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
        video_length: int | None = None,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.video_length = video_length

        total_queries = num_queries * video_length if video_length is not None else num_queries
        self.latents = nn.Parameter(torch.randn(1, total_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    _feed_forward(dim=dim, mult=ff_mult),
                ]
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
