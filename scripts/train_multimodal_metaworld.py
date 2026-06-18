"""Train the multimodal multi-stream world model on MetaWorld.

Real-backbone sibling of ``scripts/train_avid_shortcut_metaworld.py``: same data
+ VAE + DynamiCrafter preprocessor pipeline, but the model predicts a *dict* of
coupled output streams (video + proprio + tactile + ...) instead of one tensor.

The video stream is carried by the frozen DynamiCrafter base + its output
adapter (composed via ``fusion``); every other modality is predicted from
scratch by a per-modality head. The real ``DynamiCrafterBatchPreprocessor``
(VAE-encode + cross-attention conditioning) is plugged in as the *video*
preprocessor; the outer ``MultiModalBatchPreprocessor`` codec-encodes the
remaining streams from the clip batch.

Differences vs. the AVID shortcut trainer (the multimodal trainer is leaner):
    - no shortcut / step-level supervision (the multimodal line shelves it);
    - no periodic eval loop or checkpoint-resume (not implemented on
      ``MultiModalTrainer`` yet) — JSONL metrics + step checkpoints only;
    - ``fusion: compositional`` is dummy-base-only for now, so a real run uses
      ``fusion: trivial`` (see configs/multimodal_dynamicrafter.yaml).

The clip dataset must actually emit the non-video modalities named in the
config's ``output_modalities`` (``proprio`` / ``tactile`` are OPTIONAL MetaWorld
keys — see data/schema.py); trim the config to the streams your HDF5 provides.

Run:
    python scripts/train_multimodal_metaworld.py \\
        --config configs/multimodal_dynamicrafter.yaml \\
        --hdf5 ds/metaworld_corner2.hdf5 \\
        --steps 200 --batch-size 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.data import (
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    SD_VAE_DDCONFIG,
    VideoAutoencoderKL,
    build_metaworld_clip_dataset,
    precompute_null_text_embedding,
)
from generative_flow_adapters.data.clip import (
    OpenCLIPImageEmbedder,
    build_dynamicrafter_resampler_from_checkpoint,
)
from generative_flow_adapters.multimodal.builders import (
    build_codecs,
    build_multimodal_experiment,
)
from generative_flow_adapters.multimodal.config import load_multimodal_config
from generative_flow_adapters.multimodal.preprocessor import MultiModalBatchPreprocessor
from generative_flow_adapters.multimodal.trainer import MultiModalTrainer


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def _build_run_io(config) -> tuple[object | None, object | None]:
    """Mirror training.builders._maybe_build_run_io for the multimodal config.

    Gated on ``training.output_dir``; writes ``metrics.jsonl`` + ``checkpoints/``.
    """
    out = config.base.training.output_dir
    if not out:
        return None, None
    out_path = Path(out)

    jsonl_logger = None
    if config.base.training.log_metrics_jsonl:
        from generative_flow_adapters.training.metrics_logger import JsonlMetricsLogger

        jsonl_logger = JsonlMetricsLogger(out_path / "metrics.jsonl")

    checkpoint_manager = None
    if config.base.training.checkpoint_every_n_steps:
        from generative_flow_adapters.training.checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(
            out_path / "checkpoints",
            keep_last=config.base.training.keep_last_checkpoints,
        )
    return jsonl_logger, checkpoint_manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2_large.hdf5")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
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
        "--output-dir",
        default=None,
        help=(
            "Directory for run artifacts (metrics.jsonl + checkpoints/). Defaults "
            "to training.output_dir from the YAML, else outputs/<config.name>."
        ),
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
        help="Keep only the most recent K step checkpoints (final is kept). Overrides config.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = load_multimodal_config(args.config)
    model_config = config.base.model
    model_config.extra.setdefault("allow_dummy_concat_condition", False)
    # The DynamiCrafter checkpoint bundles UNet + VAE weights, so route the same
    # --vae-checkpoint path into the base-model wrapper (its UNet and
    # first_stage_model would otherwise stay random-init).
    if args.vae_checkpoint and not model_config.pretrained_model_name_or_path:
        model_config.pretrained_model_name_or_path = args.vae_checkpoint
    checkpoint_path = model_config.pretrained_model_name_or_path
    if checkpoint_path and not Path(checkpoint_path).exists():
        model_config.extra.setdefault("allow_missing_checkpoint", False)

    if args.frame_stride is not None:
        config.base.data.frame_stride = args.frame_stride

    # Run-output overrides (CLI wins over YAML). output_dir defaults to
    # outputs/<config.name> so a plain run still gets metrics.jsonl + checkpoints.
    training = config.base.training
    training.output_dir = args.output_dir or training.output_dir or f"outputs/{config.base.name}"
    if args.checkpoint_every is not None:
        training.checkpoint_every_n_steps = args.checkpoint_every
    if args.keep_last_checkpoints is not None:
        training.keep_last_checkpoints = args.keep_last_checkpoints

    temporal_length = int(model_config.extra.get("temporal_length", 8))
    context_tokens = int(config.base.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.base.conditioning.extra.get("context_dim", 512))

    # --- model + multi-stream trainer ------------------------------------- #
    components = build_multimodal_experiment(config)
    model = components.model.to(device)
    jsonl_logger, checkpoint_manager = _build_run_io(config)

    # --- VAE (preprocessor video encoder; also satisfies the video codec) -- #
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

    # Codecs for the NON-video streams. The video codec is registered (so the
    # preprocessor constructor is happy) but never invoked — the video stream is
    # produced by the DynamiCrafter video_preprocessor below.
    codecs = build_codecs(config, vae=vae)

    # --- DynamiCrafter cross-attention conditioning ------------------------ #
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
            or config.base.conditioning.extra.get("image_encoder_device")
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
        spec.key for spec in config.base.conditioning.conditions if spec.key != "step_level"
    )
    video_preprocessor = DynamiCrafterBatchPreprocessor(
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
    preprocessor = MultiModalBatchPreprocessor(
        config.output_modalities,
        codecs,
        video_preprocessor=video_preprocessor,
        condition_keys=condition_keys or ("act",),
    )

    # --- dataset ----------------------------------------------------------- #
    translator, dataset = build_metaworld_clip_dataset(
        config.base.data,
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

    trainer = MultiModalTrainer(
        model,
        components.optimizer,
        training,
        config.output_modalities,
        jsonl_logger=jsonl_logger,
        checkpoint_manager=checkpoint_manager,
    )

    trainable, total = trainable_parameter_count(model)
    vae_params = sum(parameter.numel() for parameter in vae.parameters())
    fusion_kind = str(config.base.adapter.extra.get("fusion", "compositional"))
    print(f"experiment={config.base.name}")
    print(f"device={device}")
    print(f"adapter={config.base.adapter.type}/{config.base.adapter.extra.get('architecture')} fusion={fusion_kind}")
    print(f"streams={[s.name for s in config.output_modalities]} "
          f"weights={{{', '.join(f'{s.name}:{s.loss_weight}' for s in config.output_modalities)}}}")
    print(f"vae={vae_status} params={vae_params:,}")
    print(f"dataset_size={len(dataset)} (sampling={dataset.sampling}, window={dataset.window_width}, "
          f"frame_stride={dataset.frame_stride}, fs={translator.fs_value})")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}% trainable)")
    print(f"steps={args.steps} batch_size={args.batch_size}")
    print(f"output_dir={training.output_dir}")
    print(f"checkpoints: every={training.checkpoint_every_n_steps} keep_last={training.keep_last_checkpoints}")

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
