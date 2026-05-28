from __future__ import annotations

import argparse
from pathlib import Path

from generative_flow_adapters.config import load_config
from generative_flow_adapters.testing import build_fake_dataloader
from generative_flow_adapters.training import Trainer, build_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-length", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = config.model.pretrained_model_name_or_path
    if config.model.provider.lower() == "dynamicrafter":
        config.model.extra.setdefault("allow_dummy_concat_condition", True)
        if checkpoint_path and not Path(checkpoint_path).exists():
            config.model.extra.setdefault("allow_missing_checkpoint", True)

    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    dataloader = build_fake_dataloader(
        config=config,
        batch_size=args.batch_size,
        length=max(args.dataset_length, args.steps * args.batch_size),
    )

    print(f"experiment={config.name}")
    print(f"model_type={config.model.type} prediction_type={experiment.model.prediction_type}")
    print(f"provider={config.model.provider}")
    print(f"adapter_architecture={config.adapter.extra.get('architecture', 'affine')}")
    if config.model.extra.get("allow_missing_checkpoint") and checkpoint_path and not Path(checkpoint_path).exists():
        print(f"checkpoint=skipped_missing path={checkpoint_path}")
    if config.model.extra.get("allow_dummy_concat_condition"):
        print("conditioning=dummy_concat_enabled")

    for step, batch in enumerate(dataloader, start=1):
        metrics = trainer.training_step(batch)
        loss = metrics["loss"]
        shortcut_loss = metrics.get("shortcut_direction_loss")
        print(f"step={step} loss={loss:.6f} shortcut_direction_loss={shortcut_loss}")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
