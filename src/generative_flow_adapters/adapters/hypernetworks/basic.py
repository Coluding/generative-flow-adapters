from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.common import ContextProjector, resolve_condition_embedding


class HyperNetworkAdapter(Adapter):
    def __init__(self, feature_dim: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context = ContextProjector(cond_dim=cond_dim, hidden_dim=hidden_dim)
        self.weight_generator = nn.Linear(hidden_dim, feature_dim * feature_dim)
        self.bias_generator = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        reference = base_output if base_output is not None else x_t
        if reference.dim() != 2:
            raise ValueError("HyperNetworkAdapter currently expects tensor inputs shaped [batch, feature_dim].")
        context = self.context(t, resolve_condition_embedding(cond))
        weight = self.weight_generator(context).view(-1, self.feature_dim, self.feature_dim)
        bias = self.bias_generator(context)
        delta = torch.einsum("bij,bj->bi", weight, reference) + bias
        return delta
