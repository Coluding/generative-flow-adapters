from __future__ import annotations

import torch

from generative_flow_adapters.adapters.output_interface import OutputAdapterResult
from generative_flow_adapters.models.adapted_model import AdaptedModel
from generative_flow_adapters.models.base.dummy import DummyVectorField


class FixedResultAdapter(torch.nn.Module):
    def __init__(self, result: OutputAdapterResult | torch.Tensor) -> None:
        super().__init__()
        self.result = result
        self.base_model = None

    def attach_base_model(self, base_model) -> None:
        self.base_model = base_model

    def forward(self, x_t, t, cond, base_output=None):
        del x_t, t, cond, base_output
        return self.result


def main() -> None:
    x_t = torch.randn(2, 4)
    t = torch.rand(2)
    base_model = DummyVectorField(model_type="diffusion", feature_dim=4, hidden_dim=8)

    delta_adapter = FixedResultAdapter(OutputAdapterResult(adapter_output=torch.ones_like(x_t), output_kind="delta"))
    add_model = AdaptedModel(base_model=base_model, adapter=delta_adapter, output_composition="add")
    print("add", add_model(x_t, t).shape)

    pred_adapter = FixedResultAdapter(
        OutputAdapterResult(
            adapter_output=torch.zeros_like(x_t),
            output_kind="prediction",
            gate=torch.zeros(x_t.shape[0], x_t.shape[1]),
        )
    )
    mask_model = AdaptedModel(base_model=base_model, adapter=pred_adapter, output_composition="avid_mask_mix")
    print("mask_mix", mask_model(x_t, t).shape)


if __name__ == "__main__":
    main()
