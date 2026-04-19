from __future__ import annotations

import argparse

import torch

from generative_flow_adapters.config import load_config
from generative_flow_adapters.training.builders import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def build_fake_batch(config, batch_size: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    latent_channels = int(config.model.extra.get("latent_channels", 4))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    temporal_length = int(config.model.extra.get("temporal_length", config.conditioning.extra.get("temporal_length", 8)))
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))
    action_dim = int(config.conditioning.extra.get("action_dim", 7))
    timestep_max = int(config.training.diffusion_timesteps)

    target = torch.randn(batch_size, latent_channels, temporal_length, latent_height, latent_width)
    timesteps = torch.randint(0, timestep_max, (batch_size,), dtype=torch.long)
    cond = {
        "context": torch.randn(batch_size, context_tokens, context_dim),
        "act": torch.randn(batch_size, temporal_length, action_dim),
    }
    return {"target": target, "t": timesteps, "cond": cond}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_hyperalign_fake_action.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)

    print(f"experiment={config.name}")
    print(f"provider={config.model.provider}")
    print(f"adapter_architecture={config.adapter.extra.get('architecture', 'unknown')}")
    print(f"batch_size={args.batch_size}")
    print(f"steps={args.steps}")

    for step in range(1, args.steps + 1):
        batch = build_fake_batch(config=config, batch_size=args.batch_size)
        metrics = trainer.training_step(batch)
        target = batch["target"]
        cond = batch["cond"]
        print(f"step={step} loss={metrics['loss']:.6f}")
        if isinstance(target, torch.Tensor):
            print(f"target_shape={tuple(target.shape)}")
        if isinstance(cond, dict):
            context = cond.get("context")
            act = cond.get("act")
            if isinstance(context, torch.Tensor):
                print(f"context_shape={tuple(context.shape)}")
            if isinstance(act, torch.Tensor):
                print(f"act_shape={tuple(act.shape)}")


if __name__ == "__main__":
    main()
