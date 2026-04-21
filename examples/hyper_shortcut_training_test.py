from __future__ import annotations

import argparse

from generative_flow_adapters.config import load_config
from generative_flow_adapters.testing import attach_shortcut_targets_from_base, build_fake_dataloader
from generative_flow_adapters.training.builders import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_hyper_shortcut_stepwise.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-length", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    dataloader = build_fake_dataloader(
        config=config,
        batch_size=args.batch_size,
        length=max(args.dataset_length, args.steps * args.batch_size),
    )

    print(f"experiment={config.name}")
    print(f"adapter_type={config.adapter.type}")
    print(f"adapter_architecture={config.adapter.extra.get('architecture', 'unknown')}")
    print(f"step_size_key={config.conditioning.step_size_key}")

    for step, batch in enumerate(dataloader, start=1):
        batch = attach_shortcut_targets_from_base(
            experiment.model,
            batch,
            step_size_key=config.conditioning.step_size_key,
            normalize_base_direction=bool(config.conditioning.extra.get("normalize_base_direction", True)),
        )
        metrics = trainer.training_step(batch)
        print(
            f"step={step} loss={metrics['loss']:.6f} "
            f"shortcut_direction_loss={metrics.get('shortcut_direction_loss')}"
        )
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
