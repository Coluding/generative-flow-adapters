from __future__ import annotations

import argparse

import torch

from generative_flow_adapters.config import load_config
from generative_flow_adapters.training.builders import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def build_batch(config) -> dict[str, torch.Tensor | object]:
    x_t = torch.randn(4, 32)
    t = torch.rand(4)
    target = torch.randn(4, 32)
    if config.conditioning.type == "multimodal":
        cond = {
            "action": torch.randn(4, 8),
            "goal": torch.randn(4, 8),
            "proprio": torch.randn(4, 16),
        }
        return {"x_t": x_t, "t": t, "target": target, "cond": cond}
    if config.conditioning.include_horizon:
        cond = {"action": torch.randn(4, 8), "horizon": torch.randint(1, 5, (4,), dtype=torch.float32)}
        return {
            "x_t": x_t,
            "t": t,
            "target": target,
            "cond": cond,
            "shortcut_target": torch.randn(4, 32),
            "self_consistency_target": torch.randn(4, 32),
        }
    return {"x_t": x_t, "t": t, "target": target, "cond": torch.randn(4, 8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    metrics = trainer.training_step(build_batch(config))
    print(metrics)


if __name__ == "__main__":
    main()
