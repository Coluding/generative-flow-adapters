"""Train the AVID action adapter on a frozen Wan2.2-TI2V-5B base — **new
plug-and-play building approach** (``BaseVideoModel`` + ``AdaptedModel``).

Same experiment as ``scripts/train_wan22_i2v_metaworld.py`` (same
``configs/wan22/diffusion_wan22_avid_i2v_metaworld.yaml``, same AVID adapter, same
diffusion-forcing preprocessor and loss). The ONLY difference is how the frozen
base is built:

- old: provider ``wan2.2`` -> the vendored ``Wan22DiTWrapper`` (a hand-copied
  fork of the Wan code, with our own forward/sampling wiring);
- new: provider ``wan2.2_external`` -> ``WanTI2VVideoModel``, which wraps the
  upstream ``wan.WanTI2V`` from ``external_repos/Wan2.2`` and exposes the strict
  ``BaseVideoModel`` interface (``encode``/``decode``/``denoise``/``generate``).
  The adapter composes at the ``denoise`` seam via ``AdaptedModel`` exactly as
  before — training is untouched; only the base object changed.

The base's own Wan2.2-VAE is reused for the pixel->latent encode (no second VAE
load). Generation-based eval (the native Wan step-size grid + FID/FVD) runs the
frozen base's own ``generate`` loop and is ON by default, driven by the config
cadences; held-out **loss** eval is unaffected. Pass ``--no-eval-gen`` to skip
generation for a fast smoke run.

Requires a CUDA device (the upstream WanTI2V pins ``cuda``).

Smoke run:

    python scripts/train_wan22_i2v_metaworld_external.py \
        --config configs/wan22/diffusion_wan22_avid_i2v_metaworld.yaml \
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
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer

# Wan2.2-VAE spatial stride (vae_stride = (4, 16, 16)).
_WAN22_VAE_SPATIAL_STRIDE = 16

# Batches per epoch for --overfit-index runs. The clip is the same either way;
# this only sets how often the trainer hits an epoch boundary (and so how often
# any per-epoch loader cost is paid). Also keeps the printed epoch counter from
# incrementing once per micro-step.
EPOCH_BATCH_TARGET = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan22/diffusion_wan22_avid_xattn_replace_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="Path to MetaWorld HDF5 file")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="metaworld",
                        help="Dataset family. acwm_phys reads a split dir of the HF t1an/ACWM-Phys release "
                             "(--data-dir) instead of --hdf5.")
    parser.add_argument("--data-dir", default=None,
                        help="acwm_phys only: split directory (metadata.pt + episode_*.mp4), e.g. "
                             "/scratch-shared/$USER/acwm-phys/rigid_dynamics/push_block/ind_train")
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Wan2.2-TI2V-5B checkpoint DIR (DiT safetensors + Wan2.2_VAE.pth + uncond_context.pt).",
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for the eval loader. Defaults to --batch-size when unset.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--temporal-length", type=int, default=None,
                        help="Override model.extra.temporal_length (pixel-frame window). Must match "
                             "the latent cache's window length to hit it (local cache: 41).")
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Pixel-area budget for the Wan-native aspect-preserving resize (best_output_size). "
        "Overrides config model.extra.max_area. Native Wan = 901120 (704*1280); lower it for less VRAM.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--num-windows", type=int, default=16,
                        help="random mode: draw each clip from a fixed pool of K deterministic windows "
                             "per episode (enables latent caching). 0 = unbounded random (cache can't hit).")
    parser.add_argument("--overfit-index", type=int, default=None,
                        help="Single-clip overfit sanity check: pin training to ONE dataset entry "
                             "(this episode index) repeated every step, so the target is a constant "
                             "clip the adapter only has to memorize. Pair with --num-windows 1 so the "
                             "window start is fixed at 0 (byte-identical clip each draw). Disables the "
                             "train/val split. See thesis-vault exp-single-clip-overfit.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-gen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run generation-based eval (the native Wan step-size grid + quality/FID/FVD metrics) "
        "at the config's cadences. ON by default now that eval uses the native `generate` loop "
        "(not the retired sampler). Pass --no-eval-gen to skip it (e.g. a fast smoke run); "
        "held-out loss eval always runs.",
    )
    parser.add_argument("--eval-hdf5", default=None, help="Separate held-out MetaWorld HDF5 for loss eval.")
    parser.add_argument("--eval-data-dir", default=None,
                        help="acwm_phys only: held-out split dir for loss eval (ind_test or ood_test).")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of --hdf5 held out for loss eval via a random window split (0 disables). Overrides config.",
    )
    parser.add_argument("--eval-every", type=int, default=None, help="Held-out loss eval cadence (steps). Overrides config.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Batches averaged per loss-eval cycle. Overrides config.")
    parser.add_argument("--latent-cache-dir", default=None,
                        help="Dir for cached VAE latents (skips the ~3s/step encode). Default: <hdf5>.latents/.")
    parser.add_argument("--no-latent-cache", action="store_true",
                        help="Disable the latent cache (always run the VAE encode).")
    parser.add_argument("--precompute-latents", action="store_true",
                        help="Encode every clip to the latent cache and exit (no training). Run once before training.")
    parser.add_argument("--wandb-project", default=None,
                        help="Override training.extra.wandb.project (the W&B project). CLI wins over YAML.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Override the W&B run name (defaults to the wandb block's run_name, else config.name).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA device: the upstream WanTI2V base pins cuda.")
    device = "cuda"
    config = load_config(args.config)

    # --- NEW building approach: point the base provider at the external repo ---
    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "Wan2.2_VAE.pth").exists():
        raise FileNotFoundError(f"Wan2.2_VAE.pth not found in {ckpt_dir}; pass --ckpt-dir with the full checkpoint.")
    if not list(ckpt_dir.glob("diffusion_pytorch_model*.safetensors")):
        raise FileNotFoundError(
            f"No Wan2.2 DiT safetensors in {ckpt_dir}. The external WanTI2V loads real weights via "
            "from_pretrained(ckpt_dir); a random base is not supported here."
        )
    config.model.provider = "wan2.2_external"        # -> WanTI2VVideoModel (BaseVideoModel)
    config.model.pretrained_model_name_or_path = str(ckpt_dir)   # WanTI2V takes the DIR
    config.model.extra["offload_model"] = False       # keep the DiT resident on GPU for training

    # Eval-cadence overrides (CLI wins over YAML). Loss eval only runs when a
    # cadence is set *and* a held-out loader exists.
    if args.eval_every is not None:
        config.training.eval_every_n_steps = args.eval_every
    if args.eval_batches is not None:
        config.training.eval_num_batches = args.eval_batches
    eval_hdf5 = args.eval_hdf5 or config.data.eval_hdf5
    val_fraction = args.val_fraction if args.val_fraction is not None else config.data.val_fraction
    want_eval = bool(config.training.eval_every_n_steps) and (eval_hdf5 is not None or val_fraction > 0.0)

    # Generation-based eval runs the native Wan `generate` loop now, so it is
    # driven by the config cadences by default. --no-eval-gen zeros them for a
    # fast smoke run. Held-out loss eval (above) is unaffected either way.
    if not args.eval_gen:
        config.training.inference_every_n_steps = 0
        config.training.quality_metrics = []
        config.training.quality_dist_metrics = []

    # Latent geometry from the config drives both the VAE resize and the model.
    # --temporal-length overrides the config (e.g. local 3090 runs at 41-frame
    # windows to hit the 41-frame latent cache, while the cluster configs say 97).
    if args.temporal_length is not None:
        config.model.extra["temporal_length"] = int(args.temporal_length)
        config.training.extra["inference_frame_num"] = int(args.temporal_length)
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    target_height = latent_height * _WAN22_VAE_SPATIAL_STRIDE
    target_width = latent_width * _WAN22_VAE_SPATIAL_STRIDE
    # Wan-native aspect-preserving resize budget (CLI wins over config). When set,
    # frames are sized via best_output_size to the base's native regime and the
    # fixed target_height/width above are ignored. Alignment grid = patch(2)*stride(16).
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE  # = 32

    # Optional text conditioning for the frozen base. When model.extra.
    # text_prompts_file is set, load its precomputed <stem>.contexts.pt table
    # (from precompute_prompt_contexts.py) and condition the base on each clip's
    # task prompt. null -> frame-only training (base uses its uncond context).
    prompt_contexts_path = None
    prompts_file = config.model.extra.get("text_prompts_file")
    if prompts_file:
        prompt_contexts_path = str(Path(prompts_file).with_suffix(".contexts.pt"))
        if not Path(prompt_contexts_path).exists():
            raise FileNotFoundError(
                f"text_prompts_file={prompts_file} is set but {prompt_contexts_path} is missing. "
                f"Run: python external_repos/Wan2.2/precompute_prompt_contexts.py "
                f"--ckpt_dir {ckpt_dir} --prompts_file {prompts_file}"
            )

    # Feed the native eval grid/metrics the geometry + prompt table it needs
    # (the Trainer only sees config.training). guide_scale / shift / use_prompt /
    # frame_num are already user-facing in training.extra.
    # setdefault so a smaller eval resolution set in the YAML (training.extra.
    # inference_max_area) wins — lets eval generate at a lighter res than training
    # to fit VRAM, without touching the training resolution.
    config.training.extra.setdefault("inference_max_area", max_area)
    config.training.extra["inference_temporal_length"] = temporal_length
    config.training.extra["inference_prompt_contexts_path"] = prompt_contexts_path
    config.training.extra.setdefault("inference_use_prompt", prompt_contexts_path is not None)

    # W&B project / run-name overrides (CLI wins over YAML). The logger is built
    # inside build_experiment from training.extra.wandb, so inject here first.
    # setdefault keeps `enable`/`require_vae`/etc. from the YAML block intact.
    if args.wandb_project is not None or args.wandb_run_name is not None:
        wandb_cfg = config.training.extra.get("wandb")
        if not isinstance(wandb_cfg, dict):
            wandb_cfg = {}
            config.training.extra["wandb"] = wandb_cfg
        if args.wandb_project is not None:
            wandb_cfg["project"] = args.wandb_project
        if args.wandb_run_name is not None:
            wandb_cfg["run_name"] = args.wandb_run_name

    # build_experiment now constructs AdaptedModel(WanTI2VVideoModel, AVID adapter).
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

    # Reuse the base's own Wan2.2-VAE for the pixel->latent encode and eval decode
    # (no second VAE load).
    from generative_flow_adapters.models.base.wan import make_wan_decode_fn

    vae = model.base_model.wan.vae
    # VAE encode is the per-step bottleneck (profiler: ~4.4s at fp32). The Wan VAE
    # runs encode/decode under `amp.autocast(dtype=self.dtype)`, defaulting to fp32.
    # Switching to bf16 ~halves the encode (and speeds the eval decode) — the latent
    # is still returned as fp32, so downstream is unchanged. Override with
    # training.extra.vae_dtype: float32 to keep full precision.
    vae_dtype_str = str(config.training.extra.get("vae_dtype", "bf16")).lower()
    _vae_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                  "fp16": torch.float16, "float16": torch.float16,
                  "fp32": torch.float32, "float32": torch.float32}.get(vae_dtype_str)
    if _vae_dtype is not None and getattr(vae, "dtype", None) != _vae_dtype:
        vae.dtype = _vae_dtype
        print(f"VAE encode/decode dtype -> {vae_dtype_str} (was fp32; speeds the per-step VAE encode)")
    if trainer.wandb_logger is not None:
        trainer.wandb_logger.set_decode_fn(make_wan_decode_fn(vae))

    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    raw_sigma_shift = config.training.extra.get("sigma_shift")
    sigma_shift = float(raw_sigma_shift) if raw_sigma_shift is not None else None
    cond_frames = int(config.training.extra.get("cond_frames", 1))
    cond_frames_dist = config.training.extra.get("cond_frames_dist")
    # AVID-style per-frame action conditioning: feed the adapter's action encoder a
    # per-frame sequence binned to the latent temporal grid rather than one
    # aggregated vector. action_seq_len is pinned to the latent frame count so the
    # per-frame embedding aligns with the UNet's [(b t)] layout.
    action_per_frame = bool(config.training.extra.get("action_per_frame", False))
    latent_frames = 1 + (temporal_length - 1) // 4
    # Latent cache: skip the frozen-VAE encode (per-step bottleneck) by caching z0
    # per clip. Default dir derives from the hdf5 (resolution is in each key, so one
    # dir is safe across max_area). Disabled with --no-latent-cache.
    default_cache = (
        str(Path(args.data_dir)) + ".latents"
        if args.dataset == "acwm_phys" and args.data_dir
        else str(Path(args.hdf5).with_suffix("")) + ".latents"
    )
    latent_cache_dir = None if args.no_latent_cache else (args.latent_cache_dir or default_cache)
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale,
            max_area=max_area, align_h=align, align_w=align,
            prompt_contexts_path=prompt_contexts_path,
            latent_cache_dir=latent_cache_dir,
            action_per_frame=action_per_frame,
            action_seq_len=(latent_frames if action_per_frame else None),
            sigma_shift=sigma_shift,
        ),
        condition_keys=condition_keys or ("act",),
        cond_frames=cond_frames,
        cond_frames_dist=cond_frames_dist,
    )
    if prompt_contexts_path is not None:
        n_tasks = len(preprocessor.prompt_contexts._positive) - 1  # minus __default__
        print(f"text conditioning: ON — base conditioned on task prompts "
              f"({n_tasks} tasks) from {prompt_contexts_path}")
    else:
        print("text conditioning: OFF — frame-only (base uses its unconditional context)")
    if sigma_shift is not None and sigma_shift != 1.0:
        print(f"sigma shift: {sigma_shift} — training noise levels concentrated toward high sigma "
              f"(median sigma {sigma_shift * 0.5 / (1 + (sigma_shift - 1) * 0.5):.3f}); eval stays U(0,1)")
    else:
        print("sigma shift: OFF — training sigma ~ U(0,1)")

    num_windows = args.num_windows or None  # 0 -> None (unbounded random, cache can't hit)
    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir (a split dir of the HF release).")
        translator, dataset = build_acwmphys_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            data_dir=args.data_dir,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
        )
        print(f"dataset: ACWM-Phys {translator.env_name} ({len(translator.list_episodes())} episodes) from {args.data_dir}")
    else:
        translator, dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=args.hdf5,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
        )

    eval_dataset = None
    train_dataset = dataset
    if args.overfit_index is not None:
        # Single-clip overfit: alias every access to one entry so each step trains
        # on the SAME clip. Repeat the index >= batch_size times or drop_last empties
        # the loader (Subset of len 1 < batch_size -> zero batches). With
        # --num-windows 1 the window start is pinned to 0, so the clip is identical
        # every draw.
        #
        # Eval targets that SAME clip (eval_dataset = train_dataset), not a held-out
        # split: the native generation grid + quality metrics all read their
        # conditioning batch from eval_loader (see Trainer), so pointing it at the
        # overfit clip is what makes the inference grid run — and the ideal overfit
        # signal is watching the model regenerate the exact clip it memorized.
        # `want_eval = False` only skips the held-out split branches below; the
        # eval loader is still built from eval_dataset further down. Gen eval still
        # needs --eval-gen (default on) + a nonzero inference_every_n_steps.
        if not (0 <= args.overfit_index < len(dataset)):
            raise ValueError(
                f"--overfit-index {args.overfit_index} out of range for dataset of "
                f"length {len(dataset)}."
            )
        if num_windows != 1:
            print(f"  warning: --overfit-index set but --num-windows={num_windows} "
                  f"(not 1); window start will still jitter within episode "
                  f"{args.overfit_index}. Pass --num-windows 1 for a fixed clip.")
        repeat = max(args.batch_size, 8)
        # Epoch length is a performance knob, not just bookkeeping: the trainer
        # recreates the loader iterator once per epoch, so a subset of exactly
        # one batch makes EVERY micro-step an epoch boundary and re-spawns the
        # DataLoader workers each time (~26 s under the spawn context, measured
        # 2026-07-23). Span many batches per epoch so that cost is amortized.
        train_repeat = repeat * EPOCH_BATCH_TARGET
        train_dataset = torch.utils.data.Subset(dataset, [args.overfit_index] * train_repeat)
        # Eval stays at the small subset: config.eval_num_batches (default 8)
        # bounds the loss eval, so sharing the enlarged subset would silently
        # grow each eval cycle from 1 batch to 8.
        eval_dataset = torch.utils.data.Subset(dataset, [args.overfit_index] * repeat)
        want_eval = False                 # skip the held-out split branches below
        print(f"OVERFIT MODE: training on dataset[{args.overfit_index}] repeated "
              f"{train_repeat}x (single-clip, {EPOCH_BATCH_TARGET} batches/epoch). "
              f"Eval/generation targets the same clip; "
              f"pass --eval-gen (default) to see the inference grid regenerate it.")
    if want_eval and args.eval_data_dir is not None:
        # ACWM-Phys: eval on a real held-out split dir (ind_test / ood_test).
        _, eval_dataset = build_acwmphys_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            data_dir=args.eval_data_dir,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
        )
        print(f"eval dataset: ACWM-Phys split {args.eval_data_dir}")
    elif want_eval and eval_hdf5 is not None:
        _, eval_dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=eval_hdf5,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
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

    # decord/FFmpeg (and h5py) handles are not fork-safe: the geometry probe below
    # touches the translator in the parent, and fork-after-decord deadlocks the
    # first worker get_batch(). Spawn workers re-open their readers lazily.
    _mp_ctx = "spawn" if args.num_workers > 0 else None
    # persistent_workers keeps the spawned workers alive across epoch boundaries.
    # Without it, every epoch tears down and re-spawns all `num_workers`
    # processes, each re-importing torch/decord/h5py under the spawn context —
    # ~26 s per epoch, measured 2026-07-23. That is invisible on long epochs but
    # dominated single-clip overfit runs, where an epoch is a handful of batches.
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(dataset.sampling == "exhaustive"),
        num_workers=args.num_workers,
        drop_last=True,
        multiprocessing_context=_mp_ctx,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            multiprocessing_context=_mp_ctx,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"experiment={config.name}  (NEW BaseVideoModel building approach)")
    print(f"device={device}  base={type(model.base_model).__name__} (external wan.WanTI2V)")
    print(f"adapter={config.adapter.type}/{config.adapter.extra.get('backbone')}  cond_frames={cond_frames}  "
          f"action={'per-frame[B,%d,A]' % latent_frames if action_per_frame else 'aggregated[B,A]'}")
    if max_area is not None:
        from generative_flow_adapters.data.wan_batch_preprocessor import best_output_size

        vid = dataset[0]["video"]  # [T, H, W, C]
        dataset.translator.close()  # probe opened a parent-process reader; workers open their own
        src_h, src_w = int(vid.shape[1]), int(vid.shape[2])
        ow, oh = best_output_size(src_w, src_h, align, align, max_area)
        lat_f = 1 + (temporal_length - 1) // 4
        tok = (oh // align) * (ow // align)  # patch tokens per frame (align == patch*stride)
        print(f"resize: source {src_w}x{src_h} -> Wan-native {ow}x{oh} px (max_area={max_area}, aligned to {align})")
        print(f"latent geometry: {temporal_length}f -> {lat_f}x{oh // _WAN22_VAE_SPATIAL_STRIDE}x{ow // _WAN22_VAE_SPATIAL_STRIDE}"
              f" (48-ch) | seq_len={lat_f * tok} tokens/clip")
    else:
        print(f"latent geometry: {temporal_length}f -> ({target_height}x{target_width} px, fixed) -> 48-ch Wan2.2 latent")
    print(f"params trainable={trainable:,} total={total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"dataset_size={len(dataset)} | steps={args.steps} batch_size={args.batch_size}")
    print(f"eval: {'loss@every=' + str(config.training.eval_every_n_steps) if eval_loader is not None else 'disabled'}"
          f"  gen_eval={'on' if args.eval_gen else 'off'}")

    # Offline latent precompute: encode EVERY clip (train + eval) into the cache and
    # exit. Run once; subsequent training reads latents and skips the VAE (~3s/step).
    if args.precompute_latents:
        if preprocessor.latent_cache is None:
            raise SystemExit("--precompute-latents needs the latent cache (drop --no-latent-cache).")
        import time as _time

        # Enumerate EXACTLY the windows the random sampler can draw. With a fixed
        # K-window pool that's the deterministic (episode, start) set; otherwise
        # (unbounded random / exhaustive) the dataset itself is the enumeration —
        # though unbounded random can't be fully precomputed (infinite windows).
        if num_windows is not None and dataset.sampling == "random":
            precompute_set = dataset.fixed_window_enumeration()
        else:
            if dataset.sampling == "random":
                print("WARNING: --num-windows 0 (unbounded random) — precompute only covers one "
                      "random draw per episode; training will still miss most windows. Use K>0.")
            precompute_set = dataset
        full_loader = DataLoader(precompute_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False,
                                 multiprocessing_context=_mp_ctx)
        print(f"precomputing latents -> {latent_cache_dir}  ({len(precompute_set)} windows)")
        total = len(precompute_set)
        done, encoded, t0 = 0, 0, _time.time()
        for raw_batch in full_loader:
            bs, enc = preprocessor.precompute(raw_batch)
            done += bs
            encoded += enc
            if done % 25 == 0 or done >= total:
                print(f"  {done}/{total} windows  (encoded {encoded}, cached {done - encoded}) "
                      f"{(_time.time() - t0) / max(done, 1):.2f}s/clip", flush=True)
        print(f"done: encoded {encoded} new, {done - encoded} already cached "
              f"-> {preprocessor.latent_cache.num_files()} files in {latent_cache_dir}")
        return

    trainer.train(
        loader=loader,
        max_steps=args.steps,
        preprocessor=preprocessor,
        log_every=args.log_every,
        eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()
