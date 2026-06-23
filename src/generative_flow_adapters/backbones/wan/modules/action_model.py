"""Small conditioning-agnostic Wan DiT — the Wan analogue of the AVID 11M/34M/145M
adapters.

A structural copy of ``WanModel`` (same patch-embed → AdaLN transformer blocks →
head) at a fraction of the parameters. Conditioning is injected the way AVID
injects actions: added to the timestep embedding so it drives every block's
AdaLN modulation (``e = time_embed(t) + cond_proj(c) + step_embed(d)``).

It is **modality-agnostic**: the ``cond_embedding`` is a single fused vector
``[B, cond_dim]`` produced by an external condition encoder
(``StructuredConditionEncoder`` / ``MultimodalConditionEncoder``), so adding
proprio / goal / language / … later means changing the *encoder*, not this
model. Condition dropout / CFG is handled upstream by that encoder (its
``null_embedding``). The shortcut **step size** keeps its own dedicated path.

Used as the trainable delta on the frozen 1.3B base
(``prediction = base(x_t, t) + tinyWan(x_t, t, cond, d)``); see
``adapters/output/wan.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from generative_flow_adapters.backbones.wan.modules.model import (
    WAN_CROSSATTENTION_CLASSES,
    Head,
    WanAttentionBlock,
    rope_params,
    sinusoidal_embedding_1d,
)


class ActionWanModel(nn.Module):
    """Scaled-down Wan DiT with AdaLN action + step-level conditioning."""

    def __init__(
        self,
        in_dim: int = 16,
        out_dim: int = 16,
        dim: int = 256,
        ffn_dim: int | None = None,
        freq_dim: int = 256,
        text_dim: int = 4096,
        text_len: int = 512,
        num_heads: int = 4,
        num_layers: int = 8,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        cond_dim: int = 256,
        cond_hidden_dim: int | None = None,
        use_step_level: bool = True,
        step_level_transform: str = "log2",
        condition_on_base_outputs: bool = True,
        use_text_context: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0 and (dim // num_heads) % 2 == 0
        ffn_dim = int(ffn_dim if ffn_dim is not None else dim * 4)
        cond_hidden_dim = int(cond_hidden_dim or dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.text_len = text_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.cond_dim = cond_dim
        self.use_step_level = use_step_level
        self.step_level_transform = step_level_transform
        self.condition_on_base_outputs = condition_on_base_outputs
        self.use_text_context = use_text_context

        # The adapter optionally sees the base output concatenated on channels.
        effective_in = in_dim * (2 if condition_on_base_outputs else 1)
        self.patch_embedding = nn.Conv3d(effective_in, dim, kernel_size=patch_size, stride=patch_size)

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Modality-agnostic conditioning: project the fused [B, cond_dim]
        # embedding (from the external condition encoder) -> dim, add to the time
        # embedding (AVID-style). null_cond_emb covers the no-conditioning case.
        self.cond_proj = (
            nn.Sequential(nn.Linear(cond_dim, cond_hidden_dim), nn.SiLU(), nn.Linear(cond_hidden_dim, dim))
            if cond_dim > 0
            else None
        )
        self.null_cond_emb = nn.Parameter(torch.zeros(dim))
        if use_step_level:
            self.step_embed = nn.Sequential(nn.Linear(1, cond_hidden_dim), nn.SiLU(), nn.Linear(cond_hidden_dim, dim))

        self.text_embedding = (
            nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
            if use_text_context
            else None
        )

        cross_attn_type = "t2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        # Zero-init the head so the delta is ~0 at init (identity composition).
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.head.head.bias)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def _conditioning_embedding(self, t: Tensor, cond_embedding: Tensor | None, step_level: Tensor | None) -> Tensor:
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(self.dtype))
        if self.cond_proj is not None:
            if cond_embedding is not None:
                ce = cond_embedding.to(self.dtype)
                if ce.dim() == 3:  # [B, T, cond_dim] -> pool over time
                    ce = ce.mean(dim=1)
                e = e + self.cond_proj(ce)
            else:
                e = e + self.null_cond_emb.to(self.dtype)
        if self.use_step_level and step_level is not None:
            s = step_level.to(self.dtype).reshape(-1, 1)
            if self.step_level_transform == "log2":
                s = torch.log2(s.clamp_min(1e-8))
            e = e + self.step_embed(s)
        return e

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond_embedding: Tensor | None = None,
        step_level: Tensor | None = None,
        base_output: Tensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        """``x``: ``[B, in_dim, T, H, W]`` -> delta ``[B, out_dim, T, H, W]``."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.condition_on_base_outputs:
            ref = base_output if base_output is not None else torch.zeros_like(x)
            x = torch.cat([x, ref.to(dtype=x.dtype)], dim=1)
        x = x.to(self.dtype)

        x = self.patch_embedding(x)  # [B, dim, F', H', W']
        b, _, f, h, w = x.shape
        grid_sizes = torch.tensor([[f, h, w]] * b, dtype=torch.long, device=device)
        seq = x.flatten(2).transpose(1, 2)  # [B, L, dim]
        seq_lens = torch.tensor([seq.shape[1]] * b, dtype=torch.long, device=device)

        # WanAttentionBlock asserts the AdaLN modulation is fp32, so compute the
        # time/action/step embedding with autocast disabled (the vendored
        # WanModel does the same under amp.autocast(float32)).
        with torch.autocast(device_type=device.type, enabled=False):
            e = self._conditioning_embedding(t, cond_embedding, step_level).float()
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()

        if self.text_embedding is not None and context is not None:
            ctx = self.text_embedding(context.to(self.dtype))
        else:
            # Null context: a single zero token; cross-attn becomes a no-op shift.
            ctx = torch.zeros(b, 1, self.dim, device=device, dtype=self.dtype)

        for block in self.blocks:
            seq = block(seq, e0, seq_lens, grid_sizes, self.freqs, ctx, None)
        seq = self.head(seq, e)
        return self._unpatchify(seq, (f, h, w), b)

    def _unpatchify(self, x: Tensor, grid: tuple[int, int, int], batch: int) -> Tensor:
        f, h, w = grid
        pt, ph, pw = self.patch_size
        c = self.out_dim
        x = x.view(batch, f, h, w, pt, ph, pw, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(batch, c, f * pt, h * ph, w * pw)
