"""Action-sensitivity probe on a trained adapter checkpoint (thesis requirement R1).

Answers one question: **does perturbing the action change the prediction?**

A world model whose prediction is invariant to the action is action-blind. It
can still show clean convergence and a healthy gate — the AVID/DynamiCrafter
reference run (wandb ``pg3x72uc``) shows exactly that — while being useless for
planning, because every candidate action sequence yields the same rollout. This
script separates the two.

Scope: **DynamiCrafter-family bases** (the AVID output adapter). Wan2.2 already
has an equivalent probe behind ``scripts/generate_wan22_i2v_compare.py
--sigma-sweep --action-probe``; the measurement core here
(``generative_flow_adapters.evaluation.action_sensitivity``) is backbone-agnostic,
so wiring Wan/SkyReels onto it is a follow-up, not a rewrite.

NOTE on the upstream AVID checkpoint: ``pg3x72uc`` was trained with the *real
upstream* AVID code (``external_repos/avid/...``), so its state dict does not
match a ``generative_flow_adapters`` model. Point this script at a repo-side
DynamiCrafter/AVID run, or convert the upstream checkpoint first.

Run:
    python scripts/eval_action_sensitivity.py \
        --config configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml \
        --checkpoint outputs/<run>/checkpoints/step_00010000.pt \
        --hdf5 ds/metaworld_corner2.hdf5 \
        --num-batches 8 --num-draws 4

    # ACWM-Phys (action-informative; the contrast against redundant MetaWorld):
    python scripts/eval_action_sensitivity.py \
        --config configs/dynamicrafter/diffusion_dc_acwm_pushblock.yaml \
        --checkpoint outputs/<run>/checkpoints/latest.pt \
        --dataset acwm_phys --data-dir ds/acwm-phys/rigid_body/push_block/ind_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    SD_VAE_DDCONFIG,
    VideoAutoencoderKL,
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
    precompute_null_text_embedding,
)
from generative_flow_adapters.data.clip import (
    OpenCLIPImageEmbedder,
    build_dynamicrafter_resampler_from_checkpoint,
)
from generative_flow_adapters.evaluation import (
    VARIANTS,
    format_report,
    result_to_dict,
    run_action_sensitivity,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer, _call_preprocessor


def _latest_checkpoint(config) -> Path | None:
    out_dir = getattr(config.training, "output_dir", None)
    if not out_dir:
        return None
    candidates = sorted(Path(out_dir).glob("checkpoints/*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter checkpoint (.pt). Default: newest in <output_dir>/checkpoints/.")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="metaworld")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--data-dir", default=None, help="acwm_phys only: split directory.")
    parser.add_argument("--vae-checkpoint", default="ckts/dynami512.ckpt")
    parser.add_argument("--num-batches", type=int, default=8,
                        help="Preprocessed batches to probe. Each also donates its actions to "
                             "the next batch's 'shuffle' variant, so >= 2 is required for a "
                             "meaningful shuffle at batch size 1.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-draws", type=int, default=4,
                        help="Fresh noise draws per batch. Drives the bootstrap CI width.")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"Comma-separated subset of: {','.join(VARIANTS)}")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Relative prediction change below which the model is called "
                             "action-blind. Deliberately generous — 1%% is far below anything "
                             "that could drive a planner.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 by default: decord worker processes deadlock after CUDA init "
                             "on ACWM, and a short diagnostic gains nothing from workers.")
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None,
                        help="Where to write action_sensitivity.json. Default: <output_dir>/eval/.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)
    config.model.extra.setdefault("allow_dummy_concat_condition", False)
    if args.vae_checkpoint and not config.model.pretrained_model_name_or_path:
        config.model.pretrained_model_name_or_path = args.vae_checkpoint

    # Never log a diagnostic run to wandb as if it were training.
    wandb_cfg = config.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False

    # Resolve the checkpoint before the expensive base-model build so a bad path
    # fails in a second rather than after the UNet + VAE + CLIP load.
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No adapter checkpoint found (--checkpoint or <output_dir>/checkpoints/). Got: {ckpt_path}"
        )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    temporal_length = int(config.model.extra.get("temporal_length", 8))
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    model.eval()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    print(f"checkpoint: {ckpt_path}  (global_step={payload.get('global_step')})")
    print(f"  loaded {len(payload['model'])} tensors "
          f"(missing={len(missing)} frozen-base keys expected, unexpected={len(unexpected)})")
    if unexpected:
        raise RuntimeError(
            f"checkpoint has {len(unexpected)} keys the model does not: {unexpected[:5]} ... "
            "Wrong config for this checkpoint, or an upstream-AVID checkpoint that needs conversion."
        )

    # ---- preprocessor stack (mirrors scripts/train_avid_shortcut_metaworld.py) ----
    vae = VideoAutoencoderKL(ddconfig=dict(SD_VAE_DDCONFIG), embed_dim=4).to(device)
    if args.vae_checkpoint:
        if not Path(args.vae_checkpoint).exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
        vae.load_dynamicrafter_checkpoint(args.vae_checkpoint, strict=True)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    null_embedding = precompute_null_text_embedding(
        max_length=context_tokens, device=device, dtype=next(vae.parameters()).dtype
    )
    caption_encoder = CachedNullCaptionEncoder(null_embedding)

    image_encoder = image_resampler = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        image_encoder = OpenCLIPImageEmbedder().to(device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            args.vae_checkpoint, video_length=None, device=device
        )

    condition_keys = tuple(
        spec.key for spec in config.conditioning.conditions if spec.key != "step_level"
    )
    if not any(key in ("act", "action", "action_seq") for key in condition_keys):
        raise SystemExit(
            f"config emits no action condition (conditions={condition_keys!r}). "
            "An action-sensitivity probe on an action-free model is meaningless."
        )
    preprocessor = DynamiCrafterBatchPreprocessor(
        vae=vae,
        config=BatchPreprocessConfig(
            # CFG condition dropout is a training augmentation; at 0 every probe
            # forward sees the real conditioning, which is the whole point.
            uncond_prob=0.0,
            cond_frame_index=0,
            rand_cond_frame=False,
            context_tokens=context_tokens,
            context_dim=context_dim,
            target_height=args.target_height if args.target_height > 0 else None,
            target_width=args.target_width if args.target_width > 0 else None,
            resize_mode="stretch",
            condition_keys=condition_keys,
        ),
        caption_encoder=caption_encoder,
        image_encoder=image_encoder,
        image_resampler=image_resampler,
    )

    if args.dataset == "acwm_phys":
        if not args.data_dir:
            parser.error("--dataset acwm_phys requires --data-dir.")
        _translator, dataset = build_acwmphys_clip_dataset(
            config.data, default_window_width=temporal_length, data_dir=args.data_dir,
            frame_stride=args.frame_stride, sampling=args.sampling,
        )
    else:
        _translator, dataset = build_metaworld_clip_dataset(
            config.data, default_window_width=temporal_length, hdf5=args.hdf5,
            frame_stride=args.frame_stride, sampling=args.sampling,
        )
    print(f"dataset: {args.dataset} size={len(dataset)} window={dataset.window_width}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=True)
    batches = []
    for raw in loader:
        batches.append(_call_preprocessor(preprocessor, raw, train=False))
        if len(batches) >= args.num_batches:
            break
    if len(batches) < 2:
        print("WARNING: fewer than 2 batches — the 'shuffle' variant has no donor clip "
              "and degenerates. Increase --num-batches or use a larger dataset.")

    trainer = Trainer(model, experiment.optimizer, experiment.loss_fn, config.training)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())

    print(f"\nprobing {len(batches)} batches x {args.num_draws} draws x {len(variants)} variants ...")
    result = run_action_sensitivity(
        trainer=trainer, model=model, batches=batches,
        variants=variants, num_draws=args.num_draws, seed=args.seed,
    )

    print()
    print(format_report(result, threshold=args.threshold))

    out_dir = Path(args.out_dir or (Path(config.training.output_dir or "outputs") / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result_to_dict(result)
    summary["meta"] = {
        "config": args.config,
        "checkpoint": str(ckpt_path),
        "global_step": payload.get("global_step"),
        "dataset": args.dataset,
        "data_path": args.data_dir if args.dataset == "acwm_phys" else args.hdf5,
        "num_batches": len(batches),
        "batch_size": args.batch_size,
        "num_draws": args.num_draws,
        "threshold": args.threshold,
        "seed": args.seed,
    }
    json_path = out_dir / "action_sensitivity.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {json_path}")

    # Non-zero exit on a harness error so a job script fails loudly instead of
    # logging numbers that must not be reported.
    if result.base_null_violation > 1e-6:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
