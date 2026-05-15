"""Train a HyperAlign action adapter on the MetaWorld HDF5 dataset.

Pipeline:
    HDF5 -> MetaWorldTranslator -> TranslatedClipDataset
        -> DynamiCrafterBatchPreprocessor (VideoAutoencoderKL inside)
        -> trainer.training_step

The preprocessor is a first-party port of
``LatentVisualDiffusion.get_batch_input``: it normalizes uint8 video to
[-1, 1], per-frame VAE-encodes to a 4-channel latent (×0.18215 scale
factor), and builds the cond dict with the first-frame latent replicated
across the temporal axis for the channel-concat conditioning channel.

The vendored DynamiCrafter UNet uses the tiny test config (no real
checkpoint) and only the HyperAlign adapter is trained. Pass
``--vae-checkpoint`` to load the SD/DynamiCrafter VAE weights for
meaningful latents; without it the VAE is random-init.

Run:
    python scripts/train_hyperalign_metaworld.py \\
        --config configs/diffusion_hyperalign_metaworld.yaml \\
        --hdf5 data/metaworld_corner2.hdf5 \\
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
    precompute_null_text_embedding,
)
from generative_flow_adapters.data.clip import (
    OpenCLIPImageEmbedder,
    build_dynamicrafter_resampler_from_checkpoint,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer
from generative_flow_adapters.training.wandb_logger import WandbLogger


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_hyperalign_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--frame-stride", type=int, default=1)
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = load_config(args.config)
    config.model.extra.setdefault("allow_dummy_concat_condition", False)
    checkpoint_path = config.model.pretrained_model_name_or_path
    if checkpoint_path and not Path(checkpoint_path).exists():
        config.model.extra.setdefault("allow_missing_checkpoint", False)

    temporal_length = int(config.model.extra.get("temporal_length", 8))
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    trainer = Trainer(model, experiment.optimizer,
                      experiment.loss_fn,
                      config.training,
                      experiment.wandb_logger)

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

    # Build the image cross-attention branch (OpenCLIP vision + Resampler).
    # DynamiCrafter's UNet expects context = [text(77); image(T*16)]; without
    # this branch the cross-attention silently falls through to text-only and
    # generated frames are noticeably blurry. Both encoders are frozen.
    image_encoder = None
    image_resampler = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        print("Loading OpenCLIP image embedder + Resampler for image cross-attention...")
        image_encoder = OpenCLIPImageEmbedder().to(device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            args.vae_checkpoint,
            video_length=temporal_length,
            device=device,
        )
        print(f"  image encoder + Resampler ready on {device}")

    target_height = args.target_height if args.target_height > 0 else None
    target_width = args.target_width if args.target_width > 0 else None
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
        ),
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        image_resampler=image_resampler,
    )

    translator = MetaWorldTranslator(args.hdf5, caption_mode="empty")
    dataset = TranslatedClipDataset(
        translator,
        window_width=temporal_length,
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

    trainable, total = trainable_parameter_count(model)
    vae_params = sum(parameter.numel() for parameter in vae.parameters())
    print(f"experiment={config.name}")
    print(f"device={device}")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('architecture')}")
    print(f"vae={vae_status} params={vae_params:,}")
    print(f"dataset_size={len(dataset)} (sampling={args.sampling}, window={temporal_length})")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}% trainable)")
    print(f"steps={args.steps} batch_size={args.batch_size}")

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
