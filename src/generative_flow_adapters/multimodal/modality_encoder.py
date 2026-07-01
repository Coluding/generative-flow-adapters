"""Real-backbone coupling primitives for the compositional multimodal adapter.

These two small modules make the bidirectional video↔modality coupling of the
decision note (``50_Decisions/open/multimodal-adapter-broadening.md``, the
"Converged fusion design 2026-06-19") concrete on a *real* DynamiCrafter base —
without any surgery on the frozen UNet:

- :class:`ModalityEncoder` is the **video←m** path. It turns a noised modality
  stream ``z_t^m`` + its timestep ``t_m`` into a short sequence of context
  tokens at the adapter's ``context_dim`` (1024). Those tokens are *appended* to
  the DynamiCrafter adapter's ``context`` so the adapter's existing cross
  attention attends to the modality state and emits a modality-aware video
  correction ``Δ_video``. Appending (rather than inserting) keeps the UNet's
  fixed text/image split (``context[:, :text_context_len]`` vs ``[..., len:]``,
  see ``backbones/dynamicrafter/modules/attention.py``) intact — modality tokens
  ride the *image* cross-attention stream (its own ``to_k_ip``/``to_v_ip``
  projections), so the 77-token text boundary is never shifted.

- :class:`VideoReadout` is the **m←video** path. It pools the frozen base's
  video prediction into a conditioning vector that is added to the shared
  conditioning embedding fed to the per-modality prediction heads, so each
  modality is denoised with knowledge of the video context. Joint training of
  both directions is what stops the modality streams collapsing to noise and
  makes the ``video←m`` path meaningful (decision note, inductive bias).

New params live entirely in these modules + the heads — the 11M AVID adapter and
the frozen base are untouched.
"""

from __future__ import annotations

from math import prod

import torch
from torch import Tensor, nn

from generative_flow_adapters.utils import timestep_embedding


class ModalityEncoder(nn.Module):
    """Encode a noised modality stream into ``context`` tokens (the video←m path).

    Maps ``z_t^m`` of shape ``(B, *feature_shape)`` (one token) or
    ``(B, T, *feature_shape)`` (one token per frame) plus a per-sample timestep
    ``t_m`` of shape ``(B,)`` to ``(B, n_tokens, context_dim)`` tokens ready to be
    concatenated onto the DynamiCrafter adapter's ``context``.
    """

    def __init__(
        self,
        feature_shape: tuple[int, ...],
        context_dim: int,
        *,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_shape = tuple(int(d) for d in feature_shape)
        self.feat = int(prod(self.feature_shape)) if self.feature_shape else 0
        if self.feat <= 0:
            raise ValueError("ModalityEncoder requires a non-empty feature_shape.")
        self.context_dim = int(context_dim)
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.feat + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.context_dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        n_feat_dims = len(self.feature_shape)
        lead = tuple(x_t.shape[: x_t.dim() - n_feat_dims])
        if tuple(x_t.shape[x_t.dim() - n_feat_dims :]) != self.feature_shape:
            raise ValueError(
                f"ModalityEncoder expected trailing shape {self.feature_shape}, "
                f"got tensor of shape {tuple(x_t.shape)}."
            )
        if not lead:
            raise ValueError("ModalityEncoder expects at least a leading batch axis.")
        batch = lead[0]
        n_tokens = int(prod(lead[1:])) if len(lead) > 1 else 1

        x_flat = x_t.reshape(batch, n_tokens, self.feat)
        temb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))  # (B, hidden)
        temb = temb.unsqueeze(1).expand(batch, n_tokens, -1)
        return self.proj(torch.cat([x_flat, temb], dim=-1))  # (B, n_tokens, context_dim)


class VideoReadout(nn.Module):
    """Pool the frozen base video prediction into a conditioning vector (m←video).

    Mean-pools the base output over space (and time) to a per-sample
    ``(B, in_channels)`` summary, then projects to ``cond_dim`` so it can be added
    to the shared conditioning embedding the modality heads consume. Per-frame
    readout is a later refinement; this coarse pool is robust to the video stream
    having a different frame count than the modality streams.
    """

    def __init__(self, in_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.proj = nn.Linear(self.in_channels, int(cond_dim))

    def forward(self, base_output: Tensor) -> Tensor:
        if base_output.dim() >= 3:
            # (B, C, ...) -> mean over every axis after the channel axis.
            pooled = base_output.mean(dim=tuple(range(2, base_output.dim())))
        else:
            pooled = base_output
        if pooled.shape[-1] != self.in_channels:
            raise ValueError(
                f"VideoReadout expected {self.in_channels} channels, got {pooled.shape[-1]}."
            )
        return self.proj(pooled)
