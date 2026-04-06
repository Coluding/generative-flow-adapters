from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.utils import timestep_embedding


class ContextProjector(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor, cond: Tensor | None) -> Tensor:
        time_features = timestep_embedding(t, self.hidden_dim)
        if cond is None:
            cond = torch.zeros(t.shape[0], self.cond_dim, device=t.device, dtype=time_features.dtype)
        return self.net(torch.cat([time_features, cond], dim=-1))


class TensorBroadcastHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(hidden_dim, feature_dim)
        self.shift = nn.Linear(hidden_dim, feature_dim)

    def forward(self, context: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        scale = self.scale(context)
        shift = self.shift(context)
        while scale.dim() < reference.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return scale, shift


def resolve_condition_embedding(cond: object | None) -> Tensor | None:
    if cond is None or isinstance(cond, Tensor):
        return cond
    if isinstance(cond, Mapping):
        embedding = cond.get("embedding")
        if embedding is None or isinstance(embedding, Tensor):
            return embedding
    raise TypeError("Expected a tensor condition or a mapping containing an 'embedding' tensor.")
