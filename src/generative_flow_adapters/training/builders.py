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
    wandb_logger: object | None = None


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
    wandb_logger = _maybe_build_wandb_logger(config, base_model)
    return ExperimentComponents(model=model, optimizer=optimizer, loss_fn=loss_fn, wandb_logger=wandb_logger)


def _maybe_build_wandb_logger(config: ExperimentConfig, base_model) -> object | None:
    """Build a single WandbLogger that handles both metrics and eval videos.

    Configuration lives under `training.extra.wandb` (preferred) or
    `training.extra.video_logging` (legacy alias kept so older YAMLs still work).
    Set `enable: true` to turn on logging; set `model.extra.load_first_stage_model: true`
    if you also want video panels (a VAE is needed to decode latents to RGB).
    """
    extra = config.training.extra
    cfg = extra.get("wandb") or extra.get("video_logging") or {}
    if not cfg.get("enable", False):
        return None
    decode_fn = getattr(base_model, "decode_first_stage", None)
    has_vae = decode_fn is not None and getattr(base_model, "first_stage_model", None) is not None
    if cfg.get("require_vae", True) and not has_vae:
        # Default: refuse silently mismatched configs. Set `require_vae: false`
        # in the wandb block to opt into metrics-only logging without a VAE.
        raise ValueError(
            "wandb logging requires a VAE on the base model for video panels. "
            "Either set model.extra.load_first_stage_model=true or set "
            "training.extra.wandb.require_vae=false to log scalar metrics only."
        )
    from generative_flow_adapters.training.wandb_logger import WandbLogger
    return WandbLogger(
        decode_fn=decode_fn if has_vae else None,
        num_samples=int(cfg.get("num_samples", 2)),
        fps=int(cfg.get("fps", 4)),
        project=cfg.get("wandb_project") or cfg.get("project"),
        run_name=cfg.get("wandb_run_name") or cfg.get("run_name") or config.name,
        config={"experiment": config.name},
        metrics_prefix=str(cfg.get("metrics_prefix", "train")),
    )
