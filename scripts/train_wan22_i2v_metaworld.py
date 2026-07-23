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

Held-out eval (periodic loss + best.pt) is config-driven: it fires when
``training.eval_every_n_steps`` is set and a source is configured —
``data.val_fraction`` (random split of ``data.hdf5``) or ``data.eval_hdf5`` (a
separate leak-free file). The CLI flags ``--eval-every``/``--eval-batches``/
``--val-fraction``/``--eval-hdf5`` override the config only when passed.
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

# Non-DiT weight files that live in the same checkpoint dir (VAE, text encoder,
# …) and must not be mistaken for the DiT when probing for a single-file weight.
_NON_DIT_WEIGHT_MARKERS = ("wan2.2_vae", "vae", "t5", "text_encoder", "clip", "tokenizer")


def _resolve_dit_checkpoint(ckpt_dir: Path) -> Path | None:
    """Locate the Wan2.2 DiT checkpoint in ``ckpt_dir``, or ``None`` if absent.

    Two accepted forms (see ``WanDiTWrapper.load_checkpoint``):
    - the HF sharded layout ``diffusion_pytorch_model*.safetensors`` -> return
      the directory (the loader merges the shards);
    - a single-file ``*.safetensors`` / ``*.pth`` -> return that file, ignoring
      the co-located VAE / text-encoder weights.
    """
    if not ckpt_dir.is_dir():
        return None
    if list(ckpt_dir.glob("diffusion_pytorch_model*.safetensors")):
        return ckpt_dir
    candidates = [
        path
        for path in (*ckpt_dir.glob("*.safetensors"), *ckpt_dir.glob("*.pth"))
        if not any(marker in path.name.lower() for marker in _NON_DIT_WEIGHT_MARKERS)
    ]
    return candidates[0] if len(candidates) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan22/diffusion_wan22_avid_i2v_metaworld.yaml")
    parser.add_argument(
        "--hdf5",
        default="ds/metaworld_corner2.hdf5",
        help="Path to MetaWorld HDF5 file",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Wan2.2 checkpoint dir (Wan2.2_VAE.pth + DiT safetensors). Real weights are required unless --allow-random-base.",
    )
    parser.add_argument(
        "--allow-random-base",
        action="store_true",
        help=(
            "Escape hatch for CPU/CI smoke tests ONLY: proceed with a RANDOM-INIT frozen "
            "base DiT when no checkpoint is found. Never use for real training — the adapter "
            "would train against random base weights."
        ),
    )

    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for the eval loader. Defaults to --batch-size when unset.",
    )
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-hdf5",
        default=None,
        help=(
            "Path to a separate held-out MetaWorld HDF5 for eval. Overrides data.eval_hdf5 "
            "and takes precedence over the val-fraction split (no split of the train file)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of --hdf5 held out for eval via a random window-level split "
            "(ignored when an eval HDF5 is set). 0 disables eval. Overrides data.val_fraction."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Run an eval cycle (held-out loss + best.pt) every N steps. Overrides config.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Number of held-out batches averaged per eval cycle. Overrides config.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    # Eval-cadence overrides (CLI wins over YAML). Eval only runs when a cadence
    # is set *and* a held-out loader exists (a separate --eval-hdf5 or a positive
    # --val-fraction split of the train file).
    if args.eval_every is not None:
        config.training.eval_every_n_steps = args.eval_every
    if args.eval_batches is not None:
        config.training.eval_num_batches = args.eval_batches
    # Eval data source: config is the default, CLI overrides only when passed.
    eval_hdf5 = args.eval_hdf5 or config.data.eval_hdf5
    val_fraction = args.val_fraction if args.val_fraction is not None else config.data.val_fraction
    want_eval = bool(config.training.eval_every_n_steps) and (
        eval_hdf5 is not None or val_fraction > 0.0
    )

    # Latent geometry from the config drives both the VAE resize and the model.
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    target_height = latent_height * _WAN22_VAE_SPATIAL_STRIDE
    target_width = latent_width * _WAN22_VAE_SPATIAL_STRIDE

    # Point the frozen base at the real DiT weights. The HF repo ships the DiT
    # sharded (diffusion_pytorch_model-0000N-of-*.safetensors), which the Wan
    # loader merges from the directory; a single-file .safetensors/.pth works too.
    # Training against a random-init frozen base is almost never intended, so the
    # checkpoint is MANDATORY: fail hard when it is missing unless the caller
    # explicitly opts into a random base with --allow-random-base (smoke tests).
    ckpt_dir = Path(args.ckpt_dir)
    dit_checkpoint = _resolve_dit_checkpoint(ckpt_dir)
    if dit_checkpoint is not None:
        config.model.pretrained_model_name_or_path = str(dit_checkpoint)
        # Force a strict, no-fallback base load: a present-but-broken checkpoint
        # must raise, not silently degrade to random weights.
        config.model.extra["allow_missing_checkpoint"] = False
    elif args.allow_random_base:
        print(
            "WARNING: no Wan2.2 DiT checkpoint found under "
            f"{ckpt_dir} — proceeding with a RANDOM-INIT frozen base "
            "(--allow-random-base). Do NOT use these results."
        )
        config.model.extra["allow_missing_checkpoint"] = True
    else:
        raise FileNotFoundError(
            f"Wan2.2 DiT checkpoint not found under {ckpt_dir} (looked for "
            "diffusion_pytorch_model*.safetensors, *.safetensors, or *.pth). The frozen "
            "base must be loaded for training — pass --ckpt-dir with the downloaded "
            "Wan2.2-TI2V-5B weights, or --allow-random-base for a CPU/CI smoke test only."
        )
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
    cond_frames_dist = config.training.extra.get("cond_frames_dist")
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale
        ),
        condition_keys=condition_keys or ("act",),
        cond_frames=cond_frames,
        cond_frames_dist=cond_frames_dist,
    )

    translator, dataset = build_metaworld_clip_dataset(
        config.data,
        default_window_width=temporal_length,
        hdf5=args.hdf5,
        frame_stride=args.frame_stride,
        sampling=args.sampling,
    )

    # Held-out eval source. A separate --eval-hdf5 is a clean leak-free split;
    # the --val-fraction fallback is a random window-level split of the train
    # file, so adjacent windows from the same episode can land on opposite sides
    # (in-distribution validation, not a leak-free test set).
    eval_dataset = None
    train_dataset = dataset
    if want_eval and eval_hdf5 is not None:
        _, eval_dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=eval_hdf5,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
        )
    elif want_eval:
        val_len = max(args.batch_size, int(len(dataset) * val_fraction))
        val_len = min(val_len, len(dataset) - args.batch_size)
        if val_len >= args.batch_size:
            generator = torch.Generator().manual_seed(args.seed)
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [len(dataset) - val_len, val_len], generator=generator
            )
        else:
            print("  warning: dataset too small for the requested val split; eval disabled.")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(dataset.sampling == "exhaustive"),
        num_workers=args.num_workers,
        drop_last=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
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
    if eval_loader is not None:
        source = f"eval_hdf5={eval_hdf5}" if eval_hdf5 is not None else f"val_split={val_fraction}"
        print(
            f"eval: every={config.training.eval_every_n_steps} "
            f"batches={config.training.eval_num_batches} metric={config.training.eval_metric} "
            f"val_size={len(eval_dataset)} ({source})"
        )
    else:
        print("eval: disabled")

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
        eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()
