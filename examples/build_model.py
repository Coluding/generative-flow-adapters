from __future__ import annotations

import argparse

from generative_flow_adapters.config import load_config
from generative_flow_adapters.training.builders import build_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = build_experiment(config)

    trainable = sum(parameter.numel() for parameter in experiment.model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in experiment.model.parameters() if not parameter.requires_grad)

    print(f"experiment={config.name}")
    print(f"model_type={experiment.model.model_type}")
    print(f"adapter={config.adapter.type}")
    print(f"trainable_parameters={trainable}")
    print(f"frozen_parameters={frozen}")


if __name__ == "__main__":
    main()
