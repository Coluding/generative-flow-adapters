from __future__ import annotations

import argparse
from pathlib import Path

from generative_flow_adapters.config import load_config
from generative_flow_adapters.testing import build_fake_dataloader
from generative_flow_adapters.training.builders import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/opensora_output_adapter.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-length", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.provider.lower() != "opensora":
        raise ValueError(
            f"test_opensora_adapter.py expects an OpenSora config, got provider={config.model.provider!r}."
        )

    checkpoint_path = config.model.pretrained_model_name_or_path
    if checkpoint_path and not Path(checkpoint_path).exists():
        config.model.extra.setdefault("strict_checkpoint", False)
        print(f"checkpoint=missing path={checkpoint_path} (continuing with non-strict load)")

    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    dataloader = build_fake_dataloader(
        config=config,
        batch_size=args.batch_size,
        length=max(args.dataset_length, args.steps * args.batch_size),
    )

    print(f"experiment={config.name}")
    print(f"adapter_type={config.adapter.type}")
    print(f"provider={config.model.provider}")
    print(f"use_stub={bool(config.model.extra.get('use_stub', False))}")

    for step, batch in enumerate(dataloader, start=1):
        metrics = trainer.training_step(batch)
        print(f"step={step} loss={metrics['loss']:.6f}")
        if "generated_samples" in metrics:
            generated = metrics["generated_samples"]
            if hasattr(generated, "shape"):
                print(f"generated_samples_shape={tuple(generated.shape)}")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
