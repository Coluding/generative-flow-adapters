from __future__ import annotations

from dataclasses import dataclass

import torch

from generative_flow_adapters.adapters.factory import build_adapter
from generative_flow_adapters.conditioning.encoders import build_condition_encoder
from generative_flow_adapters.config import ExperimentConfig
from generative_flow_adapters.losses.registry import LossRegistry
from generative_flow_adapters.models.adapted_model import AdaptedModel
from generative_flow_adapters.models.base.factory import build_base_model


@dataclass(slots=True)
class ExperimentComponents:
    model: AdaptedModel
    optimizer: torch.optim.Optimizer
    loss_fn: object


def build_experiment(config: ExperimentConfig) -> ExperimentComponents:
    base_model = build_base_model(config.model)
    condition_encoder = build_condition_encoder(config.conditioning)
    adapter = build_adapter(config.model, config.adapter, config.conditioning)
    model = AdaptedModel(
        base_model=base_model,
        adapter=adapter,
        condition_encoder=condition_encoder,
        pass_cond_to_base=config.model.pass_cond_to_base,
        condition_drop_prob=config.conditioning.drop_condition_prob,
        output_composition=config.adapter.composition,
        gate_bias=config.adapter.gate_bias,
        include_base_direction=bool(config.conditioning.extra.get("include_base_direction", False)),
        normalize_base_direction=bool(config.conditioning.extra.get("normalize_base_direction", True)),
    )
    loss_key = config.training.loss or config.model.type
    loss_fn = LossRegistry.get_loss(loss_key)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    return ExperimentComponents(model=model, optimizer=optimizer, loss_fn=loss_fn)
