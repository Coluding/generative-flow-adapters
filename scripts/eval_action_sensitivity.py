"""Action-sensitivity probe on a trained adapter checkpoint (thesis requirement R1).

Answers one question: **does perturbing the action change the prediction?**

A world model whose prediction is invariant to the action is action-blind. It
can still show clean convergence and a healthy gate — the AVID/DynamiCrafter
reference run (wandb ``pg3x72uc``) shows exactly that — while being useless for
planning, because every candidate action sequence yields the same rollout. This
script separates the two.

Backbones are dispatched on ``config.model.provider``:

    dynamicrafter* -> SD-VAE + OpenCLIP + DynamiCrafterBatchPreprocessor
    wan2.2*        -> Wan22DiffusionForcingPreprocessor + latent cache
    skyreels       -> SkyReelsI2VPreprocessor

The measurement core (``generative_flow_adapters.evaluation.action_sensitivity``)
is backbone-agnostic; only the data stack differs per family.

NOTE on the upstream AVID checkpoint: ``pg3x72uc`` was trained with the *real
upstream* AVID code (``external_repos/avid/...``), so its state dict does not
match a ``generative_flow_adapters`` model. Point this script at a repo-side
DynamiCrafter/AVID run, or convert the upstream checkpoint first.

Run:
    # DynamiCrafter / AVID on MetaWorld
    python scripts/eval_action_sensitivity.py \
        --config configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml \
        --checkpoint outputs/<run>/checkpoints/step_00010000.pt \
        --hdf5 ds/metaworld_corner2.hdf5

    # Wan2.2 on ACWM Push Cube
    python scripts/eval_action_sensitivity.py \
        --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_metaworld.yaml \
        --checkpoint outputs/<run>/checkpoints/latest.pt \
        --dataset acwm_phys --data-dir ds/acwm-phys/rigid_body/push_block/ind_train \
        --wan-ckpt-dir ckpts/Wan2.2-TI2V-5B --max-area 589824

    # Non-default conditioning key names
    python scripts/eval_action_sensitivity.py ... --action-keys action,action_seq
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before CUDA init: the 5B DiT + VAE brush the 24 GB ceiling, and
# without expandable segments, allocator fragmentation turns nominally free VRAM
# into an OOM (observed on the 3090).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from generative_flow_adapters.config import load_config  # noqa: E402
from generative_flow_adapters.data import (  # noqa: E402
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    SD_VAE_DDCONFIG,
    SkyReelsI2VPreprocessor,
    VideoAutoencoderKL,
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
    precompute_null_text_embedding,
)
from generative_flow_adapters.evaluation import (  # noqa: E402
    ACTION_KEYS,
    VARIANTS,
    format_report,
    result_to_dict,
    run_action_sensitivity,
)
from generative_flow_adapters.training import build_experiment  # noqa: E402
from generative_flow_adapters.training.trainer import Trainer, _call_preprocessor  # noqa: E402

_WAN_VAE_SPATIAL_STRIDE = 16


def _family(provider: str | None) -> str:
    """Map a config provider onto a data-stack family."""
    p = (provider or "").lower()
    if p.startswith("dynamicrafter"):
        return "dynamicrafter"
    if p.startswith("wan2.2") or p.startswith("wan22"):
        return "wan2.2"
    if p.startswith("wan2.1") or p.startswith("wan21"):
        return "wan2.1"
    if p.startswith("skyreels"):
        return "skyreels"
    raise SystemExit(
        f"no action-sensitivity data stack for provider {provider!r}. "
        "Supported: dynamicrafter*, wan2.2*, wan2.1*, skyreels. The measurement core is "
        "backbone-agnostic — add a stack builder here to extend it."
    )


def _latest_checkpoint(config) -> Path | None:
    out_dir = getattr(config.training, "output_dir", None)
    if not out_dir:
        return None
    candidates = sorted(Path(out_dir).glob("checkpoints/*.pt"))
    return candidates[-1] if candidates else None


def _build_dataset(args, config, temporal_length: int, **extra):
    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir (a split dir of the HF release).")
        return build_acwmphys_clip_dataset(
            config.data, default_window_width=temporal_length, data_dir=args.data_dir,
            frame_stride=args.frame_stride, sampling=args.sampling, **extra,
        )
    return build_metaworld_clip_dataset(
        config.data, default_window_width=temporal_length, hdf5=args.hdf5,
        frame_stride=args.frame_stride, sampling=args.sampling, **extra,
    )


# --------------------------------------------------------------------------
# DynamiCrafter family
# --------------------------------------------------------------------------
def _build_dynamicrafter_stack(args, config, model, device, condition_keys, temporal_length):
    context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
    context_dim = int(config.conditioning.extra.get("context_dim", 512))

    vae = VideoAutoencoderKL(ddconfig=dict(SD_VAE_DDCONFIG), embed_dim=4).to(device)
    if args.vae_checkpoint:
        if not Path(args.vae_checkpoint).exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
        vae.load_dynamicrafter_checkpoint(args.vae_checkpoint, strict=True)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    caption_encoder = CachedNullCaptionEncoder(
        precompute_null_text_embedding(
            max_length=context_tokens, device=device, dtype=next(vae.parameters()).dtype
        )
    )

    image_encoder = image_resampler = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        from generative_flow_adapters.data.clip import (  # noqa: PLC0415 — heavy import
            OpenCLIPImageEmbedder,
            build_dynamicrafter_resampler_from_checkpoint,
        )

        image_encoder = OpenCLIPImageEmbedder().to(device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            args.vae_checkpoint, video_length=None, device=device
        )

    preprocessor = DynamiCrafterBatchPreprocessor(
        vae=vae,
        config=BatchPreprocessConfig(
            # CFG condition dropout is a *training* augmentation. Leaving it on
            # would blank the action on a random subset of forwards and blur the
            # variants together — the probe must see the real conditioning.
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
    _translator, dataset = _build_dataset(args, config, temporal_length)
    return preprocessor, dataset


# --------------------------------------------------------------------------
# Wan family
# --------------------------------------------------------------------------
def _wan_preprocess_config(args, config, temporal_length: int) -> WanBatchPreprocessConfig:
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    align = 2 * _WAN_VAE_SPATIAL_STRIDE
    action_per_frame = bool(config.training.extra.get("action_per_frame", False))
    latent_frames = 1 + (temporal_length - 1) // 4

    prompt_contexts_path = None
    prompts_file = config.model.extra.get("text_prompts_file")
    if prompts_file:
        candidate = Path(prompts_file).with_suffix(".contexts.pt")
        if candidate.exists():
            prompt_contexts_path = str(candidate)

    default_cache = (
        str(Path(args.data_dir)) + ".latents"
        if args.dataset == "acwm_phys" and args.data_dir
        else str(Path(args.hdf5).with_suffix("")) + ".latents"
    )
    return WanBatchPreprocessConfig(
        target_height=latent_height * _WAN_VAE_SPATIAL_STRIDE,
        target_width=latent_width * _WAN_VAE_SPATIAL_STRIDE,
        timestep_scale=float(config.training.extra.get("flow_timestep_scale", 1000.0)),
        max_area=int(max_area) if max_area is not None else None,
        align_h=align,
        align_w=align,
        prompt_contexts_path=prompt_contexts_path,
        latent_cache_dir=args.latent_cache_dir or default_cache,
        action_per_frame=action_per_frame,
        action_seq_len=(latent_frames if action_per_frame else None),
    )


def _pre_encode_wan_windows(dataset, args, pre_cfg, condition_keys, cond_frames: int) -> bool:
    """Warm the latent cache BEFORE the big DiT is resident.

    A native-resolution window encode needs ~15 GiB — trivial on an empty GPU,
    impossible next to a resident 5B. Loads a standalone VAE, encodes only the
    windows this probe will read (cache hits are skipped), frees it.
    """
    if getattr(dataset, "num_windows", None) is None or not torch.cuda.is_available():
        return False
    needed = set(range(min(len(dataset), args.num_batches * args.batch_size)))
    enum = dataset.fixed_window_enumeration()
    order = {id(ep): i for i, ep in enumerate(dataset.episodes)}
    idxs = [i for i, (ep, _s) in enumerate(enum._pairs) if order[id(ep)] in needed]
    if not idxs:
        return False

    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from precompute_latents import _load_vae_only  # noqa: PLC0415 — sibling script, heavy import

    vae0 = _load_vae_only(Path(args.wan_ckpt_dir), "cuda")
    vae0.dtype = torch.bfloat16
    pre0 = Wan22DiffusionForcingPreprocessor(
        vae=vae0, config=pre_cfg, condition_keys=condition_keys or ("act",), cond_frames=cond_frames
    )
    newly, ok = 0, False
    try:
        for raw in DataLoader(Subset(enum, idxs), batch_size=1):
            _bs, encoded = pre0.precompute(raw)
            newly += encoded
        ok = True
        print(f"pre-encode: {len(idxs)} windows ready ({newly} newly encoded) -> {pre_cfg.latent_cache_dir}")
    except torch.OutOfMemoryError:
        print("pre-encode: GPU OOM — the preprocessor will encode on demand (slower, may OOM).")
    finally:
        del pre0, vae0
        torch.cuda.empty_cache()
    return ok


def _build_wan_stack(args, config, model, device, condition_keys, temporal_length):
    pre_cfg = _wan_preprocess_config(args, config, temporal_length)
    cond_frames = int(config.training.extra.get("cond_frames", 1))
    _translator, dataset = _build_dataset(
        args, config, temporal_length,
        num_windows=args.num_windows or None,
        **({"letterbox_aspect": (1280, 704)} if (args.letterbox and args.dataset == "acwm_phys") else {}),
    )
    _pre_encode_wan_windows(dataset, args, pre_cfg, condition_keys, cond_frames)

    vae = model.base_model.wan.vae
    # bf16 matches training-time preprocessing. The probe never generates, so
    # the native-fp32 restore the rollout paths need does not apply here.
    vae.dtype = torch.bfloat16
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=pre_cfg,
        condition_keys=condition_keys or ("act",),
        cond_frames=cond_frames,
        cond_frames_dist=config.training.extra.get("cond_frames_dist"),
    )
    return preprocessor, dataset


def _build_skyreels_stack(args, config, model, device, condition_keys, temporal_length):
    pre_cfg = _wan_preprocess_config(args, config, temporal_length)
    _translator, dataset = _build_dataset(
        args, config, temporal_length, num_windows=args.num_windows or None
    )
    preprocessor = SkyReelsI2VPreprocessor(
        model=model.base_model,
        config=pre_cfg,
        condition_keys=condition_keys or ("act",),
        device=str(device),
    )
    return preprocessor, dataset


_STACKS = {
    "dynamicrafter": _build_dynamicrafter_stack,
    "wan2.2": _build_wan_stack,
    "wan2.1": _build_wan_stack,
    "skyreels": _build_skyreels_stack,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter checkpoint (.pt). Default: newest in <output_dir>/checkpoints/.")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="metaworld")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--data-dir", default=None, help="acwm_phys only: split directory.")

    parser.add_argument("--action-keys", default=None,
                        help="Comma-separated cond keys holding the action, e.g. 'action,action_seq'. "
                             f"Default: {','.join(ACTION_KEYS)}. When given explicitly, EVERY named key "
                             "must exist in the batch or the run aborts — a typo that silently narrows "
                             "the perturbation under-reports sensitivity.")

    parser.add_argument("--num-batches", type=int, default=8,
                        help="Preprocessed batches to probe. Each donates its actions to the next "
                             "batch's 'shuffle' variant, so >= 2 is required.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-draws", type=int, default=4,
                        help="Fresh noise draws per batch. Drives the bootstrap CI width.")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"Comma-separated subset of: {','.join(VARIANTS)}")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Relative prediction change below which the model is called action-blind. "
                             "Deliberately generous — 1%% is far below anything that could drive a planner.")

    # DynamiCrafter-family
    parser.add_argument("--vae-checkpoint", default="ckts/dynami512.ckpt")
    parser.add_argument("--target-height", type=int, default=320)
    parser.add_argument("--target-width", type=int, default=512)
    # Wan/SkyReels-family
    parser.add_argument("--wan-ckpt-dir", default="ckpts/Wan2.2-TI2V-5B")
    parser.add_argument("--latent-cache-dir", default=None)
    parser.add_argument("--max-area", type=int, default=None,
                        help="Must match training or the latent cache misses and a full VAE encode runs.")
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--letterbox", action="store_true",
                        help="ACWM only: pad frames to Wan's native 1280:704 aspect.")

    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default=None)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 by default: decord workers deadlock after CUDA init on ACWM, and a "
                             "short diagnostic gains nothing from them.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None,
                        help="Where to write action_sensitivity.json. Default: <output_dir>/eval/.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)
    family = _family(config.model.provider)

    config.model.extra.setdefault("allow_dummy_concat_condition", False)
    if family == "dynamicrafter" and args.vae_checkpoint and not config.model.pretrained_model_name_or_path:
        config.model.pretrained_model_name_or_path = args.vae_checkpoint
    if family in ("wan2.2", "wan2.1"):
        config.model.pretrained_model_name_or_path = str(args.wan_ckpt_dir)
        config.model.extra["offload_model"] = False

    wandb_cfg = config.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False  # a diagnostic must never log as if it were training

    # Resolve the checkpoint before the expensive base build so a bad path fails
    # in a second rather than after a multi-GB load.
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No adapter checkpoint found (--checkpoint or <output_dir>/checkpoints/). Got: {ckpt_path}"
        )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    temporal_length = int(config.model.extra.get("temporal_length", 8))
    condition_keys = tuple(
        spec.key for spec in config.conditioning.conditions if spec.key != "step_level"
    )
    if not condition_keys:
        raise SystemExit(
            "config declares no structured conditions — an action-sensitivity probe on an "
            "action-free model is meaningless."
        )

    print(f"provider={config.model.provider} -> data stack '{family}'")
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
            "Wrong config for this checkpoint, or an upstream-AVID checkpoint needing conversion."
        )

    preprocessor, dataset = _STACKS[family](
        args, config, model, device, condition_keys, temporal_length
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
        print("WARNING: fewer than 2 batches — 'shuffle' has no donor clip and degenerates.")

    if args.action_keys:
        action_keys = tuple(k.strip() for k in args.action_keys.split(",") if k.strip())
        require_all = True
    else:
        action_keys, require_all = ACTION_KEYS, False

    trainer = Trainer(model, experiment.optimizer, experiment.loss_fn, config.training)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())

    print(f"\nprobing {len(batches)} batches x {args.num_draws} draws x {len(variants)} variants")
    print(f"action keys: {action_keys}{' (strict: all must be present)' if require_all else ''}")
    result = run_action_sensitivity(
        trainer=trainer, model=model, batches=batches, variants=variants,
        num_draws=args.num_draws, seed=args.seed,
        action_keys=action_keys, require_all_keys=require_all,
    )

    print()
    print(format_report(result, threshold=args.threshold))

    out_dir = Path(args.out_dir or (Path(config.training.output_dir or "outputs") / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result_to_dict(result)
    summary["meta"] = {
        "config": args.config,
        "provider": config.model.provider,
        "family": family,
        "checkpoint": str(ckpt_path),
        "global_step": payload.get("global_step"),
        "dataset": args.dataset,
        "data_path": args.data_dir if args.dataset == "acwm_phys" else args.hdf5,
        "action_keys": list(action_keys),
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

    # Non-zero exit on a harness error, so a job script fails loudly rather than
    # logging numbers that must not be reported.
    if result.base_null_violation > 1e-6:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
