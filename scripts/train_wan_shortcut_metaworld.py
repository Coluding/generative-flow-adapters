"""Train an action-conditioned output adapter on a frozen Wan2.1 base over the
MetaWorld HDF5 dataset (flow matching).

The Wan analogue of ``scripts/train_avid_shortcut_metaworld.py``. The difference
is the data layer: MetaWorld pixels are encoded to **16-channel Wan-VAE**
latents (not the 4-channel DynamiCrafter SD-VAE), and the batch is the
rectified-flow triple ``(x_t, t, target=noise-z0)`` built by
``WanBatchPreprocessor`` — see that module for why. The frozen Wan DiT supplies
the base velocity; only the adapter + condition encoder train.

Smoke run (real Wan-VAE + DiT already downloaded):

    python scripts/train_wan_shortcut_metaworld.py \
        --config configs/diffusion_wan_shortcut_metaworld.yaml \
        --hdf5 ds/metaworld_corner2.hdf5 \
        --ckpt-dir ckpts/Wan2.1-T2V-1.3B \
        --steps 5 --batch-size 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    WanBatchPreprocessConfig,
    WanBatchPreprocessor,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_wan_avid_shortcut_metaworld.yaml")
    parser.add_argument(
        "--hdf5",
        default="ds/metaworld_corner2.hdf5",
        help="Path to MetaWorld HDF5 file",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.1-T2V-1.3B",
        help="Wan checkpoint dir (Wan2.1_VAE.pth + DiT safetensors). Loads real weights when present.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    # Latent geometry from the config drives both the VAE resize and the model.
    temporal_length = int(config.model.extra.get("temporal_length", 8))
    latent_height = int(config.model.extra.get("latent_height", 32))
    latent_width = int(config.model.extra.get("latent_width", 32))
    target_height = latent_height * 8  # Wan-VAE spatial stride
    target_width = latent_width * 8

    # Point the frozen base at the real DiT weights when available. Accept both
    # the single-file layout (Wan2.1-1.3B: diffusion_pytorch_model.safetensors)
    # and the HF-sharded layout (Wan2.2-TI2V-5B: diffusion_pytorch_model-0000N-of-*
    # .safetensors); the Wan wrapper's loader globs *.safetensors over the dir.
    ckpt_dir = Path(args.ckpt_dir)
    has_dit_weights = (ckpt_dir / "diffusion_pytorch_model.safetensors").exists() or bool(
        list(ckpt_dir.glob("diffusion_pytorch_model-*.safetensors"))
    )
    if has_dit_weights:
        config.model.pretrained_model_name_or_path = str(ckpt_dir)
        config.model.extra["allow_missing_checkpoint"] = False
    experiment = build_experiment(config)
    model = experiment.model.to(device)
    trainer = Trainer(
        experiment.model,
        experiment.optimizer,
        experiment.loss_fn,
        config.training,
        wandb_logger=getattr(experiment, "wandb_logger", None),
        checkpoint_manager=getattr(experiment, "checkpoint_manager", None),
    )

    # Wan-VAE for the pixel->latent encode.
    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
    from generative_flow_adapters.models.base.wan import make_wan_decode_fn

    vae_pth = ckpt_dir / "Wan2.1_VAE.pth"
    if not vae_pth.exists():
        raise FileNotFoundError(f"Wan-VAE not found at {vae_pth}; pass --ckpt-dir with the downloaded checkpoint.")
    vae = WanVAE(vae_pth=str(vae_pth), device=device)

    # Reuse the same VAE for the eval video grid (latent->pixel decode). The
    # logger is built decoder-less by build_experiment (the frozen Wan DiT has
    # no decode_first_stage), so inject the decoder now that the VAE exists.
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.set_decode_fn(make_wan_decode_fn(vae))

    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    # Single source for the timestep convention: t = sigma * flow_timestep_scale
    # (Wan native = 1000). Shared with the trainer (shortcut + eval sampler).
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    preprocessor = WanBatchPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale
        ),
        condition_keys=condition_keys or ("act",),
    )

    translator, dataset = build_metaworld_clip_dataset(
        config.data,
        default_window_width=temporal_length,
        hdf5=args.hdf5,
        frame_stride=args.frame_stride,
        sampling=args.sampling,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(dataset.sampling == "exhaustive"),
        num_workers=args.num_workers,
        drop_last=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"experiment={config.name}")
    print(f"device={device}  base_dit={'real' if config.model.pretrained_model_name_or_path else 'random-init'}")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('backbone')}")
    print(f"latent geometry: {temporal_length}f -> ({target_height}x{target_width} px) -> 16-ch Wan latent")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"dataset_size={len(dataset)} | steps={args.steps} batch_size={args.batch_size}")

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
