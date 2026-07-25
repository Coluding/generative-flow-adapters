"""Train the action adapter on a frozen SkyReels-V2-I2V-1.3B base (matrix runs
2 & 3: SkyReels × ACWM push_block da=2 / robot_arm da=7).

Mirrors ``scripts/train_wan22_i2v_metaworld_external.py`` (BaseVideoModel +
AdaptedModel + Trainer, with the ``--dataset acwm_phys --data-dir`` switch), but:

- base provider ``skyreels`` -> :class:`SkyReelsVideoModel` (16-ch Wan2.1 VAE,
  flow/velocity), built from the HF-cached ``Skywork/SkyReels-V2-I2V-1.3B-540P``;
- batch preprocessor :class:`SkyReelsI2VPreprocessor` (classic i2v: obs injected
  via ``y``/``clip_fea``, text via SkyReels' own T5 — NOT the Wan umT5 table);
- VAE spatial stride 8 (Wan2.2 is 16).

The adapter composes at the ``denoise`` seam via ``AdaptedModel`` exactly as for
Wan. Requires CUDA (SkyReels' DiT/VAE/CLIP/T5 load onto GPU).

Smoke run (needs the weights + a GPU):

    python scripts/train_skyreels_acwm.py \
        --config configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml \
        --dataset acwm_phys --data-dir ds/acwm-phys/kinematics/robot_arm/ind_train \
        --steps 5 --batch-size 1 --no-eval-gen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    SkyReelsI2VPreprocessor,
    WanBatchPreprocessConfig,
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer

# SkyReels' Wan2.1 VAE downsamples space 8x (vae_stride = (4, 8, 8)).
_SKYREELS_VAE_SPATIAL_STRIDE = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="acwm_phys")
    parser.add_argument("--data-dir", default=None,
                        help="acwm_phys: split dir (metadata.pt + episode_*.mp4).")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--eval-data-dir", default=None, help="acwm_phys held-out split (ind_test/ood_test).")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--temporal-length", type=int, default=None,
                        help="Override model.extra.temporal_length (pixel-frame window).")
    parser.add_argument("--max-area", type=int, default=None,
                        help="Aspect-preserving resize budget. Overrides config model.extra.max_area.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--num-windows", type=int, default=8,
                        help="K deterministic windows/episode for the latent cache. 0 = unbounded random.")
    parser.add_argument("--latent-cache-dir", default=None,
                        help="Dir for cached 16-ch z0. Default: <data-dir>.skyreels.latents/.")
    parser.add_argument("--no-latent-cache", action="store_true")
    parser.add_argument("--eval-gen", action=argparse.BooleanOptionalAction, default=True,
                        help="Run generation-based eval at the config cadences (--no-eval-gen for a smoke run).")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("train_skyreels_acwm needs a CUDA device (SkyReels DiT/VAE/CLIP/T5 load onto GPU).")
    device = "cuda"
    config = load_config(args.config)
    config.model.provider = "skyreels"  # -> SkyReelsVideoModel (BaseVideoModel)

    if args.eval_every is not None:
        config.training.eval_every_n_steps = args.eval_every
    if not args.eval_gen:
        config.training.inference_every_n_steps = 0
        config.training.quality_metrics = []
        config.training.quality_dist_metrics = []

    # Geometry from the config (SkyReels stride 8).
    if args.temporal_length is not None:
        config.model.extra["temporal_length"] = int(args.temporal_length)
        config.training.extra["inference_frame_num"] = int(args.temporal_length)
    temporal_length = int(config.model.extra.get("temporal_length", 65))
    latent_height = int(config.model.extra.get("latent_height", 48))
    latent_width = int(config.model.extra.get("latent_width", 48))
    stride = _SKYREELS_VAE_SPATIAL_STRIDE
    target_height = latent_height * stride
    target_width = latent_width * stride
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * stride  # patch(2) * stride(8) = 16

    # W&B overrides (CLI wins over YAML).
    if args.wandb_project is not None or args.wandb_run_name is not None:
        wandb_cfg = config.training.extra.get("wandb")
        if not isinstance(wandb_cfg, dict):
            wandb_cfg = {}
            config.training.extra["wandb"] = wandb_cfg
        if args.wandb_project is not None:
            wandb_cfg["project"] = args.wandb_project
        if args.wandb_run_name is not None:
            wandb_cfg["run_name"] = args.wandb_run_name

    # AdaptedModel(SkyReelsVideoModel, adapter) + optimizer/loss/logger.
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
    base = model.base_model  # SkyReelsVideoModel
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.set_decode_fn(lambda z: base.decode(z))

    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    raw_shift = config.training.extra.get("sigma_shift")
    sigma_shift = float(raw_shift) if raw_shift is not None else None
    default_prompt = str(config.model.extra.get("default_prompt", ""))

    default_cache = (
        str(Path(args.data_dir)) + ".skyreels.latents"
        if args.dataset == "acwm_phys" and args.data_dir
        else str(Path(args.hdf5).with_suffix("")) + ".skyreels.latents"
    )
    latent_cache_dir = None if args.no_latent_cache else (args.latent_cache_dir or default_cache)

    preprocessor = SkyReelsI2VPreprocessor(
        model=base,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale,
            max_area=max_area, align_h=align, align_w=align,
            prompt_contexts_path=None,       # SkyReels encodes text live (own T5)
            latent_cache_dir=latent_cache_dir,
            sigma_shift=sigma_shift,
        ),
        condition_keys=condition_keys or ("act",),
        device=device,
        default_prompt=default_prompt,
    )

    num_windows = args.num_windows or None
    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir.")
        translator, dataset = build_acwmphys_clip_dataset(
            config.data, default_window_width=temporal_length, data_dir=args.data_dir,
            frame_stride=args.frame_stride, sampling=args.sampling, num_windows=num_windows,
        )
        print(f"dataset: ACWM-Phys {translator.env_name} ({len(translator.list_episodes())} episodes) from {args.data_dir}")
    else:
        translator, dataset = build_metaworld_clip_dataset(
            config.data, default_window_width=temporal_length, hdf5=args.hdf5,
            frame_stride=args.frame_stride, sampling=args.sampling, num_windows=num_windows,
        )

    eval_dataset = None
    if args.eval_data_dir is not None and args.dataset == "acwm_phys":
        _, eval_dataset = build_acwmphys_clip_dataset(
            config.data, default_window_width=temporal_length, data_dir=args.eval_data_dir,
            frame_stride=args.frame_stride, sampling=args.sampling, num_windows=num_windows,
        )
        print(f"eval dataset: ACWM-Phys split {args.eval_data_dir}")

    _mp_ctx = "spawn" if args.num_workers > 0 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(dataset.sampling == "exhaustive"),
        num_workers=args.num_workers, drop_last=True, multiprocessing_context=_mp_ctx,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size or args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=True, multiprocessing_context=_mp_ctx,
            persistent_workers=args.num_workers > 0,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"experiment={config.name}  base={type(base).__name__} (SkyReels-V2-I2V-1.3B)")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('backbone')} feature_dim={config.adapter.feature_dim} "
          f"action_dim={config.conditioning.input_dim}")
    print(f"latent geometry: {temporal_length}f -> stride-{stride} 16-ch latent | max_area={max_area} align={align}")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"dataset_size={len(dataset)} steps={args.steps} batch_size={args.batch_size} "
          f"eval={'on' if eval_loader is not None else 'off'} gen_eval={'on' if args.eval_gen else 'off'}")

    trainer.train(
        loader=loader, max_steps=args.steps, preprocessor=preprocessor,
        log_every=args.log_every, eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()
