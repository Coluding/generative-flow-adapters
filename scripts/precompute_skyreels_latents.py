"""Standalone SkyReels z0-latent precompute (cache prewarm) for a clip dataset.

SkyReels' 16-ch Wan2.1 VAE encode is a per-step training cost. This script fills
the ``.skyreels.latents`` cache ahead of time so ``scripts/train_skyreels_acwm.py``
reads z0 from disk instead of re-encoding on the first pass.

Why not ``scripts/precompute_latents.py``? That script hard-uses the Wan2.2
diffusion-forcing preprocessor (48-ch, stride 16) — the wrong VAE convention AND
cache-key scheme for SkyReels. This script instead builds the SAME
:class:`SkyReelsI2VPreprocessor` as training and calls its inherited
``precompute()`` (``WanBatchPreprocessor.precompute`` -> the SkyReels-overridden
``_encode_z0`` + inherited ``_latent_keys``), so the cached keys are IDENTICAL to
what training hits — by construction, as long as ``--config`` and ``--num-windows``
match the training run.

Only z0 is cached; the i2v side channels (``y`` / ``clip_fea`` / text) are built
live per batch during training either way, so they are not precomputed here.

Example (MetaWorld):

    python scripts/precompute_skyreels_latents.py \
        --dataset metaworld --hdf5 ds/metaworld_corner2.hdf5 \
        --config configs/skyreels/diffusion_skyreels_xattn_metaworld.yaml \
        --num-windows 8 --batch-size 4

Example (ACWM-Phys, one split — the job loops splits into a shared cache):

    python scripts/precompute_skyreels_latents.py \
        --dataset acwm_phys --data-dir ds/acwm-phys/.../push_block/ind_train \
        --config configs/skyreels/diffusion_skyreels_xattn_acwm_pushblock.yaml \
        --latent-cache-dir ds/acwm-phys/.../push_block/skyreels.latents.shared \
        --num-windows 8 --batch-size 4
"""

from __future__ import annotations

import argparse
import time
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
from generative_flow_adapters.models.base.factory import build_base_model

# SkyReels' Wan2.1 VAE downsamples space 8x (vae_stride = (4, 8, 8)) — must match
# scripts/train_skyreels_acwm.py so the resized geometry (and thus the cache keys)
# are identical.
_SKYREELS_VAE_SPATIAL_STRIDE = 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml",
                        help="MUST be the SAME config the training run uses (geometry drives the cache keys).")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="acwm_phys")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--data-dir", default=None,
                        help="acwm_phys: split dir (metadata.pt + episode_*.mp4).")
    parser.add_argument("--num-windows", type=int, default=8,
                        help="K deterministic windows/episode. MUST match the training --num-windows.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--temporal-length", type=int, default=None,
                        help="Override model.extra.temporal_length (must match training).")
    parser.add_argument("--max-area", type=int, default=None,
                        help="Override model.extra.max_area (must match training).")
    parser.add_argument("--latent-cache-dir", default=None,
                        help="Default: <data-dir>.skyreels.latents / <hdf5 stem>.skyreels.latents "
                             "(identical to training).")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of windows (debugging).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("precompute_skyreels_latents needs a CUDA device (SkyReels VAE loads onto GPU).")
    device = "cuda"

    config = load_config(args.config)
    config.model.provider = "skyreels"  # -> SkyReelsVideoModel (BaseVideoModel)

    # --- geometry, EXACTLY as train_skyreels_acwm.py derives it -------------
    if args.temporal_length is not None:
        config.model.extra["temporal_length"] = int(args.temporal_length)
    temporal_length = int(config.model.extra.get("temporal_length", 65))
    latent_height = int(config.model.extra.get("latent_height", 48))
    latent_width = int(config.model.extra.get("latent_width", 48))
    stride = _SKYREELS_VAE_SPATIAL_STRIDE
    target_height = latent_height * stride
    target_width = latent_width * stride
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * stride  # patch(2) * stride(8) = 16
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    raw_shift = config.training.extra.get("sigma_shift")
    sigma_shift = float(raw_shift) if raw_shift is not None else None
    default_prompt = str(config.model.extra.get("default_prompt", ""))
    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")

    # --- cache dir: identical default to training --------------------------
    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir.")
        default_cache = str(Path(args.data_dir)) + ".skyreels.latents"
    else:
        default_cache = str(Path(args.hdf5).with_suffix("")) + ".skyreels.latents"
    latent_cache_dir = args.latent_cache_dir or default_cache

    # --- SkyReels base (VAE lives on it) — base only, no adapter/optimizer --
    print(f"building SkyReels base ({config.model.extra.get('model_id')}) ...", flush=True)
    base = build_base_model(config.model)
    base = base.to(device)

    preprocessor = SkyReelsI2VPreprocessor(
        model=base,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width, timestep_scale=timestep_scale,
            max_area=max_area, align_h=align, align_w=align,
            prompt_contexts_path=None,          # z0 encode needs no text/clip
            latent_cache_dir=latent_cache_dir,
            sigma_shift=sigma_shift,
        ),
        condition_keys=condition_keys or ("act",),
        device=device,
        default_prompt=default_prompt,
    )
    if preprocessor.latent_cache is None:
        raise SystemExit("latent cache is disabled — pass a --latent-cache-dir.")

    # --- dataset: SAME build + num-windows as training ---------------------
    num_windows = args.num_windows or None
    if args.dataset == "acwm_phys":
        translator, dataset = build_acwmphys_clip_dataset(
            config.data, default_window_width=temporal_length, data_dir=args.data_dir,
            frame_stride=args.frame_stride, sampling=args.sampling, num_windows=num_windows,
        )
    else:
        translator, dataset = build_metaworld_clip_dataset(
            config.data, default_window_width=temporal_length, hdf5=args.hdf5,
            frame_stride=args.frame_stride, sampling=args.sampling, num_windows=num_windows,
        )

    # Enumerate EXACTLY the windows training samples (K per episode).
    if num_windows is not None and dataset.sampling == "random":
        precompute_set = dataset.fixed_window_enumeration()
    elif dataset.sampling == "exhaustive":
        precompute_set = dataset
    else:
        raise SystemExit(
            "--num-windows 0 with --sampling random has infinitely many windows and cannot be "
            "precomputed. Use --num-windows K>0 (recommended, and matching training) or --sampling exhaustive."
        )

    total = len(precompute_set) if args.limit is None else min(args.limit, len(precompute_set))
    print(f"latent geometry: {temporal_length}f -> stride-{stride} 16-ch latent | "
          f"target {target_height}x{target_width} | max_area={max_area} align={align}")
    print(f"precomputing SkyReels z0 -> {latent_cache_dir}  ({total} windows, num_windows={num_windows})", flush=True)

    # decord/h5py handles are not fork-safe; close the parent reader before spawn workers.
    dataset.translator.close()
    loader = DataLoader(
        precompute_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        drop_last=False, multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    done, encoded, t0, last_report = 0, 0, time.time(), 0
    for raw_batch in loader:
        if args.limit is not None and done >= args.limit:
            break
        bs, enc = preprocessor.precompute(raw_batch)
        done += bs
        encoded += enc
        if done - last_report >= 25 or done >= total:
            last_report = done
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{total} windows ({encoded} newly encoded, {done - encoded} cache hits) "
                  f"{rate:.1f} win/s", flush=True)

    print(f"done: encoded {encoded} new / {done} windows -> {latent_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
