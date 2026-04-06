from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel, infer_prediction_type
from generative_flow_adapters.utils import timestep_embedding


class DummyVectorField(BaseGenerativeModel):
    def __init__(self, model_type: str, feature_dim: int, hidden_dim: int) -> None:
        super().__init__(model_type=model_type, prediction_type=infer_prediction_type(model_type))
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        del cond
        time_features = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        if x_t.dim() != 2:
            raise ValueError("DummyVectorField expects x_t shaped [batch, feature_dim].")
        features = torch.cat([x_t, time_features], dim=-1)
        return self.net(features)
