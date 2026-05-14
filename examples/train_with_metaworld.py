"""Smoke-wire a real MetaWorld HDF5 dataset into a configured model.

This is the boundary the dataset layer guarantees: the translator + dataset
deliver the canonical clip schema (see ``data/schema.py``); turning ``video``
into the trainer-expected ``target`` latent tensor (e.g. via the DynamiCrafter
VAE) is the responsibility of a follow-up encoder step that lives outside the
dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset
from generative_flow_adapters.training import build_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hdf5", required=True, help="Path to MetaWorld HDF5 file")
    parser.add_argument("--window-width", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--caption-mode", choices=["empty", "template"], default="empty")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.provider.lower() == "dynamicrafter":
        config.model.extra.setdefault("allow_dummy_concat_condition", True)
        checkpoint_path = config.model.pretrained_model_name_or_path
        if checkpoint_path and not Path(checkpoint_path).exists():
            config.model.extra.setdefault("allow_missing_checkpoint", True)

    experiment = build_experiment(config)

    torch.manual_seed(args.seed)
    translator = MetaWorldTranslator(args.hdf5, caption_mode=args.caption_mode)
    dataset = TranslatedClipDataset(
        translator,
        window_width=args.window_width,
        frame_stride=args.frame_stride,
        sampling=args.sampling,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.sampling == "exhaustive"),
        num_workers=args.num_workers,
        drop_last=True,
    )

    print(f"experiment={config.name}")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('architecture')}")
    print(f"dataset_size={len(dataset)} (sampling={args.sampling})")

    for step, batch in enumerate(loader, start=1):
        shapes = {
            key: (tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__)
            for key, value in batch.items()
        }
        print(f"batch={step} {shapes}")
        if step >= args.num_batches:
            break

    # NOTE: full trainer.training_step requires `batch["target"]` as latent
    # tensors. That step (encode `video` -> latents, build cond from `act`)
    # belongs to a follow-up wiring once the VAE / first-stage encoder is in
    # place. The dataset returns raw frames by design.


if __name__ == "__main__":
    main()
