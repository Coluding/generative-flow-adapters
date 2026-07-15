"""Train an AVID-style output adapter on MetaWorld with shortcut supervision.

Same data + preprocessor pipeline as ``scripts/train_hyperalign_metaworld.py``
but the adapter is the small action-conditioned DynamiCrafter UNet (AVID 11M)
attached as an *output* adapter on top of the frozen base UNet (the
``avid_mask_mix`` composition: ``base * sigmoid(gate) + adapter * (1 - gate)``).

Default config: ``configs/diffusion_avid_shortcut_metaworld.yaml``. The
trainer:

    1. samples ``step_level`` per batch (Uniform{min, max} from
       ``training.extra.shortcut_step_level_*``);
    2. injects ``step_level`` into the cond mapping (the adapter's
       ``use_step_level_conditioning`` branch consumes the same value);
    3. computes ``shortcut_target = base(x_t, t, cond) * step_level`` against
       the same (x_t, t) used for the adapter forward;
    4. adds the configured ``shortcut_direction`` loss on top of the standard
       diffusion loss.

Run:
    python scripts/train_avid_shortcut_metaworld.py \\
        --config configs/diffusion_avid_shortcut_metaworld.yaml \\
        --hdf5 ds/metaworld_corner2.hdf5 \\
        --steps 200 --batch-size 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    MetaWorldTranslator,
    SD_VAE_DDCONFIG,
    TranslatedClipDataset,
    VideoAutoencoderKL,
    build_metaworld_clip_dataset,
    precompute_null_text_embedding,
)
from generative_flow_adapters.data.clip import (
    OpenCLIPImageEmbedder,
    build_dynamicrafter_resampler_from_checkpoint,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_avid_shortcut_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2_large.hdf5")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for the eval loader. Defaults to --batch-size when unset.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default=None,
                        help="Override data.sampling from the config when set.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "Override data.frame_stride from the config when set. The frame stride "
            "subsamples every k-th frame (longer window + action-SUM); prefer setting "
            "it in the config's `data:` block."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--vae-checkpoint",
        default="ckts/dynami512.ckpt",
        help="Path to a DynamiCrafter / SD VAE checkpoint. Without it, the VAE is random-init (smoke run).",
    )
    parser.add_argument("--uncond-prob", type=float, default=0.05, help="CFG dropout probability per branch.")
    parser.add_argument(
        "--target-height",
        type=int,
        default=320,
        help=(
            "Pixel height the preprocessor resizes clips to. Default 320 matches the "
            "dynamicrafter_512 training resolution (latent 40x64). Set to 0 to disable resize."
        ),
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=512,
        help="Pixel width; see --target-height. Default 512 matches dynamicrafter_512.",
    )
    parser.add_argument(
        "--image-encoder-device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help=(
            "Device for the OpenCLIP image embedder used by the preprocessor's "
            "image cross-attention branch."
        ),
    )
    parser.add_argument(
        "--clip-null-prompt",
        dest="clip_null_prompt",
        action="store_true",
        default=True,
        help=(
            "Run OpenCLIP once on the empty string and feed the cached null-prompt "
            "embedding into cross-attention (matches DynamiCrafter pretraining)."
        ),
    )
    parser.add_argument(
        "--no-clip-null-prompt",
        dest="clip_null_prompt",
        action="store_false",
        help="Disable CLIP null-prompt precompute; cross-attention context will be zeros.",
    )
    parser.add_argument(
        "--shortcut-step-level-max",
        type=int,
        default=None,
        help="Override training.extra.shortcut_step_level_max for quick debugging.",
    )
    parser.add_argument(
        "--shortcut-step-level-min",
        type=int,
        default=None,
        help="Override training.extra.shortcut_step_level_min.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for run artifacts (metrics.jsonl + checkpoints/). Defaults "
            "to training.output_dir from the YAML, else outputs/<config.name>."
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a step-tagged checkpoint every N steps. Overrides config.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=None,
        help="Keep only the most recent K step checkpoints (best/final are kept). Overrides config.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of the dataset held out for eval (random split). 0 disables eval.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint (.pt) to resume trainable weights + optimizer + global_step from.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = load_config(args.config)
    config.model.extra.setdefault("allow_dummy_concat_condition", False)
    # The DynamiCrafter checkpoint bundles UNet + VAE weights, so route the
    # same --vae-checkpoint path into the base-model wrapper. Without this the
    # wrapper's UNet and `first_stage_model` stay random-init, which produced
    # noise rollouts and a broken wandb GT panel even when the standalone
    # preprocessor VAE was loaded correctly.
    if args.vae_checkpoint and not config.model.pretrained_model_name_or_path:
        config.model.pretrained_model_name_or_path = args.vae_checkpoint
    checkpoint_path = config.model.pretrained_model_name_or_path
    if checkpoint_path and not Path(checkpoint_path).exists():
        config.model.extra.setdefault("allow_missing_checkpoint", False)

    if args.shortcut_step_level_max is not None:
        config.training.extra["shortcut_step_level_max"] = int(args.shortcut_step_level_max)
    if args.shortcut_step_level_min is not None:
        config.training.extra["shortcut_step_level_min"] = int(args.shortcut_step_level_min)

    if config.training.shortcut_direction_weight <= 0.0:
        print(
            "WARNING: training.shortcut_direction_weight <= 0; the shortcut loss is OFF. "
            "Set a positive weight in the YAML if you intended to train with shortcut supervision."
        )

    # Run-output overrides (CLI wins over YAML). output_dir defaults to
    # outputs/<config.name> so a plain run still gets metrics.jsonl + checkpoints.
    config.training.output_dir = args.output_dir or config.training.output_dir or f"outputs/{config.name}"
    if args.eval_every is not None:
        config.training.eval_every_n_steps = args.eval_every
    if args.eval_batches is not None:
        config.training.eval_num_batches = args.eval_batches
    if args.checkpoint_every is not None:
        config.training.checkpoint_every_n_steps = args.checkpoint_every
    if args.keep_last_checkpoints is not None:
        config.training.keep_last_checkpoints = args.keep_last_checkpoints
    want_eval = bool(config.training.eval_every_n_steps) and args.val_fraction > 0.0

    temporal_length = int(config.model.extra.get("temporal_length", 8))
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    trainer = Trainer(
        model,
        experiment.optimizer,
        experiment.loss_fn,
        config.training,
        experiment.wandb_logger,
        jsonl_logger=experiment.jsonl_logger,
        checkpoint_manager=experiment.checkpoint_manager,
    )
    if args.resume is not None:
        payload = trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at global_step={trainer.global_step} "
              f"(best_eval_metric={trainer.best_eval_metric:.5f})")

    vae = VideoAutoencoderKL(ddconfig=dict(SD_VAE_DDCONFIG), embed_dim=4).to(device)
    vae_status = "random-init"
    if args.vae_checkpoint is not None:
        if not Path(args.vae_checkpoint).exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
        loaded_keys = vae.load_dynamicrafter_checkpoint(args.vae_checkpoint, strict=True)
        vae_status = f"loaded {len(loaded_keys)} tensors from {args.vae_checkpoint}"
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    caption_encoder = None
    if args.clip_null_prompt:
        print("Precomputing OpenCLIP null-prompt embedding...")
        null_embedding = precompute_null_text_embedding(
            max_length=context_tokens,
            device=device,
            dtype=next(vae.parameters()).dtype,
        )
        if tuple(null_embedding.shape) != (1, context_tokens, context_dim):
            raise ValueError(
                "OpenCLIP null-prompt shape "
                f"{tuple(null_embedding.shape)} does not match config "
                f"(context_tokens={context_tokens}, context_dim={context_dim}). "
                "Disable with --no-clip-null-prompt or adjust the config."
            )
        caption_encoder = CachedNullCaptionEncoder(null_embedding)
        print(f"  cached null prompt: shape={tuple(null_embedding.shape)} device={null_embedding.device}")

    image_encoder = None
    image_resampler = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        device_spec = (
            args.image_encoder_device
            or config.conditioning.extra.get("image_encoder_device")
            or "auto"
        )
        image_encoder_device = device if device_spec == "auto" else torch.device(device_spec)
        print("Loading OpenCLIP image embedder + Resampler for image cross-attention...")
        image_encoder = OpenCLIPImageEmbedder().to(image_encoder_device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            args.vae_checkpoint,
            video_length=None,
            device=device,
        )
        checkpoint_video_length = image_resampler.video_length
        if checkpoint_video_length != temporal_length:
            raise ValueError(
                f"Config has temporal_length={temporal_length} but the DynamiCrafter "
                f"checkpoint was trained at temporal_length={checkpoint_video_length}. "
                "Set `model.extra.temporal_length` and `conditioning.extra.temporal_length` to "
                f"{checkpoint_video_length} in your YAML."
            )
        print(
            f"  image encoder on {image_encoder_device} | Resampler on {device} "
            f"(T={checkpoint_video_length})"
        )

    target_height = args.target_height if args.target_height > 0 else None
    target_width = args.target_width if args.target_width > 0 else None
    condition_keys = tuple(
        spec.key for spec in config.conditioning.conditions if spec.key != "step_level"
    )
    if not condition_keys:
        print("  warning: no dataset-emitted structured conditions; "
              "the adapter's condition encoder will only see trainer-synthesized step_level.")
    preprocessor = DynamiCrafterBatchPreprocessor(
        vae=vae,
        config=BatchPreprocessConfig(
            uncond_prob=args.uncond_prob,
            cond_frame_index=0,
            rand_cond_frame=False,
            context_tokens=context_tokens,
            context_dim=context_dim,
            target_height=target_height,
            target_width=target_width,
            resize_mode="stretch",
            condition_keys=condition_keys,
        ),
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        image_resampler=image_resampler,
    )

    translator, dataset = build_metaworld_clip_dataset(
        config.data,
        default_window_width=temporal_length,
        hdf5=args.hdf5,
        frame_stride=args.frame_stride,
        sampling=args.sampling,
    )

    # Hold out a fraction for the periodic eval cycle. Random window-level split
    # with a fixed generator — note that adjacent windows from the same episode
    # can land on opposite sides, so this is in-distribution validation, not a
    # leak-free held-out test split. See thesis-vault feature note.
    eval_dataset = None
    train_dataset = dataset
    if want_eval:
        val_len = max(args.batch_size, int(len(dataset) * args.val_fraction))
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

    trainable, total = trainable_parameter_count(model)
    vae_params = sum(parameter.numel() for parameter in vae.parameters())
    step_min = int(config.training.extra.get("shortcut_step_level_min", 1))
    step_max = int(config.training.extra.get("shortcut_step_level_max", 1))
    print(f"experiment={config.name}")
    print(f"device={device}")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('architecture')}")
    print(f"composition={config.adapter.composition}")
    print(f"vae={vae_status} params={vae_params:,}")
    print(f"dataset_size={len(dataset)} (sampling={dataset.sampling}, window={dataset.window_width}, "
          f"frame_stride={dataset.frame_stride}, fs={translator.fs_value})")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}% trainable)")
    print(
        f"shortcut: weight={config.training.shortcut_direction_weight} "
        f"method={config.training.shortcut_target_method} "
        f"step_level~U[{step_min}, {step_max}]"
    )
    print(f"steps={args.steps} batch_size={args.batch_size}")
    print(f"output_dir={config.training.output_dir}")
    if eval_loader is not None:
        print(
            f"eval: every={config.training.eval_every_n_steps} "
            f"batches={config.training.eval_num_batches} metric={config.training.eval_metric} "
            f"val_size={len(eval_dataset)}"
        )
    else:
        print("eval: disabled")
    print(
        f"checkpoints: every={config.training.checkpoint_every_n_steps} "
        f"keep_last={config.training.keep_last_checkpoints}"
    )

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
        eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()
