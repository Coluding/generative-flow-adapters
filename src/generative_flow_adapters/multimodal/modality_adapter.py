"""Per-modality prediction heads.

Lightweight denoising heads for the non-video streams (proprio / depth /
tactile). Each head predicts its modality's diffusion target (noise / velocity)
from the noised modality input, its own timestep, and the shared conditioning
embedding (the cross-modal coupling lives in that shared embedding +, in the
compositional fusion, the learned mask — sub-decision 4).

A single MLP head (``ModalityPredictionHead``) covers both ``vector`` and
``map`` kinds by flattening the per-frame ``feature_shape`` and acting
per-frame across any leading ``(B,)`` / ``(B, T)`` axes. Light convs are a
deferred refinement (decision note). The head is also reused as the *video*
adjustment adapter on the dummy base, where the real run uses
``DynamicCrafterOutputAdapter`` instead.

The final projection is zero-initialised so every head starts as the zero
function — the video-adjustment contribution is identity at init (matches the
``Δ→0`` convention of the single-stream adapters).
"""

from __future__ import annotations

from math import prod

import torch
from torch import Tensor, nn

from generative_flow_adapters.utils import timestep_embedding


class ModalityPredictionHead(nn.Module):
    def __init__(
        self,
        feature_shape: tuple[int, ...],
        cond_dim: int | None,
        hidden_dim: int = 128,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_shape = tuple(int(d) for d in feature_shape)
        self.feat = int(prod(self.feature_shape)) if self.feature_shape else 0
        if self.feat <= 0:
            raise ValueError("ModalityPredictionHead requires a non-empty feature_shape.")
        self.cond_dim = int(cond_dim or 0)
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        in_dim = self.feat + hidden_dim + self.cond_dim
        final = nn.Linear(hidden_dim, self.feat)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            final,
        )

    def _lead(self, x_t: Tensor) -> tuple[int, ...]:
        n_feat_dims = len(self.feature_shape)
        lead = tuple(x_t.shape[: x_t.dim() - n_feat_dims])
        if tuple(x_t.shape[x_t.dim() - n_feat_dims :]) != self.feature_shape:
            raise ValueError(
                f"ModalityPredictionHead expected trailing shape {self.feature_shape}, "
                f"got tensor of shape {tuple(x_t.shape)}."
            )
        return lead

    def forward(self, x_t: Tensor, t: Tensor, cond_emb: Tensor | None = None) -> Tensor:
        lead = self._lead(x_t)
        batch = lead[0]
        n = int(prod(lead)) if lead else 1
        repeat = n // batch  # per-sample leading multiplicity (e.g. T); 1 when lead == (B,)

        x_flat = x_t.reshape(n, self.feat)
        temb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))  # (B, hidden)
        parts = [x_flat, temb.repeat_interleave(repeat, dim=0) if repeat > 1 else temb]
        if cond_emb is not None and self.cond_dim > 0:
            parts.append(self._align_cond(cond_emb, lead, n, repeat))
        out = self.net(torch.cat(parts, dim=-1))
        return out.reshape(*lead, *self.feature_shape)

    def _align_cond(self, cond_emb: Tensor, lead: tuple[int, ...], n: int, repeat: int) -> Tensor:
        """Flatten the conditioning embedding to ``(n, cond_dim)`` to match ``x_flat``.

        Two shapes occur: per-sample ``(B, cond_dim)`` (the dummy substrate, and
        action conditioning with no time axis) is broadcast across the leading
        multiplicity via ``repeat_interleave``; per-frame ``(B, T, cond_dim)``
        (real per-frame action conditioning, already aligned with a ``(B, T)``
        stream) is flattened directly. Mixed cases fall back to repeat.
        """
        if cond_emb.dim() == len(lead) + 1 and tuple(cond_emb.shape[: len(lead)]) == lead:
            return cond_emb.reshape(n, self.cond_dim)
        return cond_emb.repeat_interleave(repeat, dim=0) if repeat > 1 else cond_emb
