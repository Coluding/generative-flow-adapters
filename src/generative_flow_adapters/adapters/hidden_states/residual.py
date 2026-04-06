from __future__ import annotations

from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.common import ContextProjector, TensorBroadcastHead, resolve_condition_embedding


class ResidualConditioningAdapter(Adapter):
    def __init__(self, feature_dim: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context = ContextProjector(cond_dim=cond_dim, hidden_dim=hidden_dim)
        self.head = TensorBroadcastHead(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        reference = base_output if base_output is not None else x_t
        context = self.context(t, resolve_condition_embedding(cond))
        scale, shift = self.head(context, reference)
        gate = self.gate(context)
        while gate.dim() < reference.dim():
            gate = gate.unsqueeze(-1)
        hidden = (reference.tanh() * scale) + shift
        return gate * hidden
