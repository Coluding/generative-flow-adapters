from __future__ import annotations

from torch import Tensor

from generative_flow_adapters.adapters.common import ContextProjector, TensorBroadcastHead, resolve_condition_embedding
from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult


class AffineOutputAdapter(OutputAdapterInterface):
    def __init__(self, feature_dim: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context = ContextProjector(cond_dim=cond_dim, hidden_dim=hidden_dim)
        self.head = TensorBroadcastHead(feature_dim=feature_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        reference = base_output if base_output is not None else x_t
        context = self.context(t, resolve_condition_embedding(cond))
        scale, shift = self.head(context, reference)
        return OutputAdapterResult(adapter_output=reference * scale + shift, output_kind="delta")


OutputAdapter = AffineOutputAdapter
