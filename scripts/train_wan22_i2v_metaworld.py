"""Train an action-conditioned output adapter on a frozen Wan2.2-TI2V-5B base
over the MetaWorld HDF5 dataset (flow matching, diffusion forcing).

The Wan2.2 analogue of ``scripts/train_wan_shortcut_metaworld.py``. Differences:
- the base is Wan2.2 TI2V-5B (per-token timestep DiT, ``model2_2``), loaded via
  provider ``wan2.2``;
- pixels are encoded with the **Wan2.2-VAE** (48-ch latents, stride (4,16,16));
- the batch is the **diffusion-forcing** triple built by
  ``Wan22DiffusionForcingPreprocessor``: the leading observation frame(s) are
  held clean (timestep 0) and the future frames are denoised from them, so the
  frozen base does the image conditioning and the adapter does the action.

Smoke run (real Wan2.2-TI2V-5B + VAE downloaded):

    python scripts/train_wan22_i2v_metaworld.py \
        --config configs/diffusion_wan22_i2v_metaworld.yaml \
        --hdf5 ds/metaworld_corner2.hdf5 \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --steps 5 --batch-size 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer

# Wan2.2-VAE spatial stride (vae_stride = (4, 16, 16)).
_WAN22_VAE_SPATIAL_STRIDE = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_wan22_i2v_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="Path to MetaWorld HDF5 file")
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Wan2.2 checkpoint dir (Wan2.2_VAE.pth + DiT safetensors). Loads real weights when present.",
    )
    parser.add_argument("--steps", type=int, default=5)
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
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    target_height = latent_height * _WAN22_VAE_SPATIAL_STRIDE
    target_width = latent_width * _WAN22_VAE_SPATIAL_STRIDE

    # Point the frozen base at the real DiT weights when available. The HF repo
    # ships the DiT sharded (diffusion_pytorch_model-0000N-of-*.safetensors),
    # which the Wan loader merges from the directory; a single-file checkpoint
    # works too. Either form means real weights are present.
    ckpt_dir = Path(args.ckpt_dir)
    if list(ckpt_dir.glob("diffusion_pytorch_model*.safetensors")):
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

    # Wan2.2-VAE for the pixel->latent encode (and the eval grid decode).
    from generative_flow_adapters.backbones.wan.modules.vae2_2 import Wan2_2_VAE
    from generative_flow_adapters.models.base.wan import make_wan_decode_fn

    vae_pth = ckpt_dir / "Wan2.2_VAE.pth"
    if not vae_pth.exists():
        raise FileNotFoundError(f"Wan2.2-VAE not found at {vae_pth}; pass --ckpt-dir with the downloaded checkpoint.")
    vae = Wan2_2_VAE(vae_pth=str(vae_pth), device=device)

    # The decode_fn contract is VAE-version-agnostic (same list-based decode).
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.set_decode_fn(make_wan_decode_fn(vae))

    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    cond_frames = int(config.training.extra.get("cond_frames", 1))
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale
        ),
        condition_keys=condition_keys or ("act",),
        cond_frames=cond_frames,
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
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('backbone')}  cond_frames={cond_frames}")
    print(f"latent geometry: {temporal_length}f -> ({target_height}x{target_width} px) -> 48-ch Wan2.2 latent")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"dataset_size={len(dataset)} | steps={args.steps} batch_size={args.batch_size}")

    trainer.train(loader=loader, max_steps=args.steps, preprocessor=preprocessor, log_every=args.log_every)


if __name__ == "__main__":
    main()
