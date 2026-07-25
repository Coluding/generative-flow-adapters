"""Standalone base-vs-adapted video comparison for the DynamiCrafter backbone.

The DynamiCrafter analogue of ``scripts/generate_wan22_i2v_compare.py``. Loads a
trained adapter checkpoint (from ``training.output_dir``/checkpoints or
``--checkpoint``) on top of the frozen DynamiCrafter UNet, then on the SAME
weights runs two comparisons:

1. **Loss (training seam)** — ``Trainer.evaluate`` over a few preprocessor
   batches, printing the adapted denoise loss, the frozen-base denoise loss,
   and their delta (the ``eval_denoise_adapter_delta`` metric from wandb).
2. **Generation (generation seam)** — the DDIM rollout from a dataset clip's
   conditioning (image cross-attention + first-frame concat latent), once for
   the frozen base and once for the adapted model from a SHARED noise draw, then
   decodes both (and the ground-truth latent) to pixels and writes a
   side-by-side mp4 + frame strip (GT | base | adapted).

Unlike the Wan backbone, DynamiCrafter exposes no native ``generate`` loop, so
generation reuses the Trainer's DDIM ``inference_sampler`` /
``base_inference_sampler`` (exactly the path the periodic wandb rollouts use)
and the wrapper's ``decode_first_stage`` for the latent->pixel decode.

Example:

    python scripts/generate_dynamicrafter_compare.py \\
        --config configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml \\
        --vae-checkpoint ckts/dynami512.ckpt \\
        --hdf5 ds/metaworld_corner2.hdf5 \\
        --checkpoint outputs/diffusion_avid_shortcut_metaworld/checkpoints/step_00000500.pt \\
        --num-steps 50

    # No trained checkpoint yet? Exercise the base-vs-adapted generation +
    # decode plumbing with a perturbed (untrained) adapter:
    python scripts/generate_dynamicrafter_compare.py --random-init
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

from generative_flow_adapters.config import load_config  # noqa: E402
from generative_flow_adapters.data import (  # noqa: E402
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    SD_VAE_DDCONFIG,
    VideoAutoencoderKL,
    build_metaworld_clip_dataset,
    precompute_null_text_embedding,
)
from generative_flow_adapters.data.clip import (  # noqa: E402
    OpenCLIPImageEmbedder,
    build_dynamicrafter_resampler_from_checkpoint,
)
from generative_flow_adapters.training import build_experiment  # noqa: E402
from generative_flow_adapters.training.trainer import (  # noqa: E402
    Trainer,
    _call_preprocessor,
    _strip_adapter_only_keys,
)


def _latest_checkpoint(config) -> Path | None:
    out_dir = getattr(config.training, "output_dir", None)
    if not out_dir:
        return None
    candidates = sorted(Path(out_dir).glob("checkpoints/*.pt"))
    return candidates[-1] if candidates else None


def _decode_to_uint8_frames(base_model, latents: Tensor) -> "list":
    """Decode ``[1, C, T, h, w]`` latents to a list of ``[H, W, 3]`` uint8 frames.

    Matches the WandbLogger normalisation: clamp(-1, 1) -> [0, 255] uint8.
    """
    decoded = base_model.decode_first_stage(latents)  # [1, 3, T, H, W] in ~[-1, 1]
    if decoded.dim() != 5:
        raise ValueError(f"decode_first_stage must return 5D [B, 3, T, H, W]; got {tuple(decoded.shape)}.")
    v = decoded[0].detach().float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)  # [3, T, H, W]
    return [v[:, i].permute(1, 2, 0).cpu().numpy() for i in range(v.shape[1])]


def _save_outputs(out_dir: Path, tag: str, gt: "list", base: "list", adapted: "list", fps: int) -> None:
    import numpy as np
    import imageio.v2 as imageio

    n = min(len(gt), len(base), len(adapted))
    panel = [np.concatenate([gt[i], base[i], adapted[i]], axis=1) for i in range(n)]
    video_path = out_dir / f"{tag}_gt_base_adapted.mp4"
    imageio.mimwrite(video_path, panel, fps=fps, codec="h264", quality=8)
    strip_idx = list(range(0, n, max(1, n // 6)))[:6]
    strip = np.concatenate([panel[i] for i in strip_idx], axis=0)
    strip_path = out_dir / f"{tag}_strip.png"
    imageio.imwrite(strip_path, strip)
    print(f"wrote {video_path}  (panels: GT | base | adapted)")
    print(f"wrote {strip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter checkpoint (.pt). Default: newest in <output_dir>/checkpoints/.")
    parser.add_argument("--random-init", action="store_true",
                        help="Skip the checkpoint and randomly perturb the adapter-only params instead "
                        "(excludes base-aliased params so the frozen UNet is not corrupted). Exercises the "
                        "base-vs-adapted generation + decode path without trained weights.")
    parser.add_argument("--vae-checkpoint", default="ckts/dynami512.ckpt",
                        help="DynamiCrafter checkpoint (UNet + VAE + Resampler). Also routed into the base "
                        "wrapper so its UNet / first_stage_model load (not random-init).")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--clip-index", type=int, default=0, help="Dataset episode index for the rollout clip.")
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM solver steps for the rollout.")
    parser.add_argument("--fs", type=int, default=None,
                        help="Override the fps/frame-stride conditioning value fed to the base UNet for the "
                        "rollout (cond['fs']). AVID's MetaWorld eval uses default_fs=10; the data pipeline "
                        "here defaults fs_value=1. Set to probe whether fps conditioning drives the blur.")
    parser.add_argument("--loss-batches", type=int, default=4, help="Batches for the loss comparison (0 skips).")
    parser.add_argument("--loss-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--uncond-prob", type=float, default=0.0,
                        help="CFG dropout probability in the preprocessor. 0 for a deterministic eval.")
    parser.add_argument("--target-height", type=int, default=320,
                        help="Pixel height the preprocessor resizes to (dynamicrafter_512 default). 0 disables.")
    parser.add_argument("--target-width", type=int, default=512, help="Pixel width; see --target-height.")
    parser.add_argument("--image-encoder-device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument("--clip-null-prompt", dest="clip_null_prompt", action="store_true", default=True,
                        help="Feed OpenCLIP's null-prompt embedding into cross-attention (matches pretraining).")
    parser.add_argument("--no-clip-null-prompt", dest="clip_null_prompt", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/dynamicrafter_compare")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)

    # Route the DynamiCrafter checkpoint into the base wrapper so its UNet +
    # first_stage_model load from real weights (else noise rollouts / broken GT).
    if args.vae_checkpoint and not config.model.pretrained_model_name_or_path:
        config.model.pretrained_model_name_or_path = args.vae_checkpoint
    # We decode with the wrapper's first_stage_model, so it must be built.
    config.model.extra["load_first_stage_model"] = True
    wandb_cfg = config.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False  # debug script: never log to wandb

    temporal_length = int(config.model.extra.get("temporal_length", 16))
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))

    # Resolve the checkpoint BEFORE the expensive build so a bad path fails fast.
    payload = None
    ckpt_path = None
    if not args.random_init:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No adapter checkpoint found (looked for --checkpoint / <output_dir>/checkpoints). "
                f"Got: {ckpt_path}. Pass --random-init to run without trained weights."
            )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    model.eval()
    if payload is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"checkpoint: {ckpt_path}  (global_step={payload.get('global_step')})")
        print(f"  loaded {len(payload['model'])} trainable tensors  (missing={len(missing)} frozen-base keys "
              f"expected, unexpected={len(unexpected)})")
        if unexpected:
            raise RuntimeError(f"checkpoint has {len(unexpected)} keys the model doesn't: {unexpected[:5]} ...")
    else:
        # RANDOM-INIT: the AVID adapter with condition_on_base_outputs ALIASES
        # frozen-base params — perturbing them corrupts the base and collapses
        # every rollout to noise. Perturb ONLY adapter-exclusive params.
        with torch.no_grad():
            base_ptrs = {p.data_ptr() for p in model.base_model.parameters()}
            adapter_only = [p for p in model.adapter.parameters() if p.data_ptr() not in base_ptrs]
            for p in adapter_only:
                p.add_(0.02 * torch.randn_like(p))
        print(f"RANDOM-INIT MODE: perturbed {len(adapter_only)} adapter-only params "
              f"(excluded base-aliased params to avoid corrupting the frozen UNet).")

    # --- preprocessor: same pipeline as scripts/train_avid_shortcut_metaworld.py
    vae = VideoAutoencoderKL(ddconfig=dict(SD_VAE_DDCONFIG), embed_dim=4).to(device)
    if args.vae_checkpoint is not None:
        if not Path(args.vae_checkpoint).exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
        loaded = vae.load_dynamicrafter_checkpoint(args.vae_checkpoint, strict=True)
        print(f"preprocessor VAE: loaded {len(loaded)} tensors from {args.vae_checkpoint}")
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    caption_encoder = None
    if args.clip_null_prompt:
        null_embedding = precompute_null_text_embedding(
            max_length=context_tokens, device=device, dtype=next(vae.parameters()).dtype
        )
        if tuple(null_embedding.shape) != (1, context_tokens, context_dim):
            raise ValueError(
                f"OpenCLIP null-prompt shape {tuple(null_embedding.shape)} != config "
                f"(context_tokens={context_tokens}, context_dim={context_dim})."
            )
        caption_encoder = CachedNullCaptionEncoder(null_embedding)

    image_encoder = None
    image_resampler = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        device_spec = args.image_encoder_device or config.conditioning.extra.get("image_encoder_device") or "auto"
        image_encoder_device = device if device_spec == "auto" else torch.device(device_spec)
        image_encoder = OpenCLIPImageEmbedder().to(image_encoder_device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            args.vae_checkpoint, video_length=None, device=device
        )
        if image_resampler.video_length != temporal_length:
            raise ValueError(
                f"Config temporal_length={temporal_length} but the checkpoint Resampler was trained at "
                f"temporal_length={image_resampler.video_length}. Align model.extra.temporal_length."
            )
        print(f"image cross-attention: encoder on {image_encoder_device}, Resampler on {device} "
              f"(T={image_resampler.video_length})")

    target_height = args.target_height if args.target_height > 0 else None
    target_width = args.target_width if args.target_width > 0 else None
    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
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
            condition_keys=condition_keys or ("act",),
        ),
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        image_resampler=image_resampler,
    )

    _, dataset = build_metaworld_clip_dataset(
        config.data,
        default_window_width=temporal_length,
        hdf5=args.hdf5,
        frame_stride=int(config.data.frame_stride or 1),
        sampling="random",
    )
    if not (0 <= args.clip_index < len(dataset)):
        raise ValueError(f"--clip-index {args.clip_index} out of range (dataset len {len(dataset)}).")

    trainer = Trainer(model, experiment.optimizer, experiment.loss_fn, config.training)
    # The Trainer only builds base_inference_sampler when a wandb_logger is
    # attached (its only in-training consumer). We pass none, so build the
    # frozen-base sampler explicitly with the same family/config as the adapted
    # one — it drives the "no-adapter" row of the comparison.
    base_sampler = trainer.base_inference_sampler or trainer._build_inference_sampler(model.base_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) loss comparison at the training seam --------------------------
    if args.loss_batches > 0:
        loader = DataLoader(dataset, batch_size=args.loss_batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)
        stats = trainer.evaluate(loader, max_batches=args.loss_batches, preprocessor=preprocessor)
        print(f"\n=== Denoise loss over {args.loss_batches} batches (training seam) ===")
        print(f"  adapted denoise loss : {stats.get('eval_base_loss', float('nan')):.5f}")
        print(f"  base    denoise loss : {stats.get('eval_denoise_base_only', float('nan')):.5f}")
        print(f"  delta (base-adapted) : {stats.get('eval_denoise_adapter_delta', float('nan')):+.5f}  "
              f"(>0 = adapter better)")
        rel = stats.get("eval_adapter_rel_contribution")
        if rel is not None:
            print(f"  adapter_rel_contribution: {rel:.4f}  (|pred-base|/|base|)")
        total = stats.get("eval_loss")
        if total is not None:
            print(f"  total eval loss (incl. shortcut terms): {total:.5f}")

    # ---- 2) DDIM rollout: base vs adapted from a shared noise draw ---------
    raw_batch = next(iter(DataLoader(Subset(dataset, [args.clip_index]), batch_size=1)))
    batch = _call_preprocessor(preprocessor, raw_batch, train=False)
    target = batch["target"]  # clean latent z0 (DynamiCrafter target IS the latent)
    if not isinstance(target, Tensor):
        raise RuntimeError("preprocessed batch has no tensor 'target' latent to sample against.")

    cond = batch.get("cond")
    if args.fs is not None and isinstance(cond, dict):
        fs_val = cond.get("fs")
        new_fs = torch.full_like(fs_val, int(args.fs)) if isinstance(fs_val, Tensor) \
            else torch.full((target.shape[0],), int(args.fs), device=target.device, dtype=torch.long)
        cond["fs"] = new_fs
        print(f"fs override: cond['fs'] set to {int(args.fs)} (was {fs_val.tolist() if isinstance(fs_val, Tensor) else fs_val})")

    print(f"\n=== Rollout: clip {args.clip_index}, {args.num_steps} DDIM steps, "
          f"T={target.shape[2]}, latent {tuple(target.shape)} ===")

    # DynamiCrafter i2v grounding = concat conditioning + first-frame anchoring:
    # frame 0 of the latent is pinned to the clean observation latent at every
    # DDIM step (mirrors AVID's `mask=cond_mask, x0=z, clean_cond=True`, lvdm
    # DDIMSampler.ddim_sampling). Without it the rollout drifts to a blurry mean.
    anchor_mask = torch.zeros_like(target)
    anchor_mask[:, :, 0, :, :] = 1.0

    with torch.no_grad():
        shared_noise = torch.randn_like(target)
        adapted = trainer.inference_sampler.sample_from_batch(
            batch=batch, num_inference_steps=args.num_steps, initial_sample=shared_noise,
            anchor_mask=anchor_mask, anchor_latent=target,
        )
        base_cond = _strip_adapter_only_keys(batch.get("cond"))
        base = base_sampler.sample_from_batch(
            batch={"target": target, "cond": base_cond},
            num_inference_steps=args.num_steps, initial_sample=shared_noise,
            anchor_mask=anchor_mask, anchor_latent=target,
        )
        gt_frames = _decode_to_uint8_frames(model.base_model, target)
        base_frames = _decode_to_uint8_frames(model.base_model, base)
        adapted_frames = _decode_to_uint8_frames(model.base_model, adapted)

    fps = int((config.training.extra.get("wandb") or {}).get("fps", 5))
    tag = f"clip{args.clip_index}_s{args.num_steps}"
    _save_outputs(out_dir, tag, gt_frames, base_frames, adapted_frames, fps)


if __name__ == "__main__":
    main()
