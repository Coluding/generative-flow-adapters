"""Standalone VAE-latent precompute for a MetaWorld clip dataset.

The frozen Wan2.2 VAE encode is the per-step training bottleneck (~1.8s/clip on an
A100, ~3.3s on a 3090) and it is pure recomputation. This script encodes every clip
the trainer will sample **once** into the on-disk latent cache
(``generative_flow_adapters.data.latent_cache``), so training reads latents and skips
the VAE entirely.

Unlike ``train_wan22_i2v_metaworld_external.py --precompute-latents``, this loads
**only the VAE** — not the 5B DiT, the adapter, the optimizer, or wandb — so it starts
fast and uses little memory. It reuses the exact same dataset build, resize, VAE dtype,
and ``_encode_z0`` cache logic as training, so the latents (and their cache keys) are
identical: the trainer will hit every one.

Windows: with ``--sampling random`` and ``--num-windows K`` (the training default),
each episode exposes a fixed pool of K deterministic evenly-spaced start indices, and
this script enumerates *exactly* that pool (K x num_episodes windows). Use the SAME K
here and in training, or the keys won't match.

Example:
    python scripts/precompute_latents.py \
        --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --num-windows 16 --max-area 589824

Then train normally (latent cache is on by default):
    python scripts/train_wan22_i2v_metaworld_external.py --num-windows 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.models.base.wan_ti2v import _ensure_wan_importable

# Wan2.2-VAE spatial stride (vae_stride = (4, 16, 16)); align grid = patch(2)*stride(16).
_WAN22_VAE_SPATIAL_STRIDE = 16

# Per-provider VAE spatial stride. Wan2.2 downsamples 16x, SkyReels' Wan2.1 VAE 8x.
# Drives target_height/width and the align grid so a provider's latent geometry
# matches its VAE. Default (unlisted providers) -> the Wan2.2 stride, so the
# Wan path is byte-identical to before.
_PROVIDER_VAE_SPATIAL_STRIDE = {
    "wan2.2": 16, "wan": 16, "wan2.1": 16,
    "skyreels": 8,
}


def _vae_spatial_stride(provider: str | None) -> int:
    return _PROVIDER_VAE_SPATIAL_STRIDE.get((provider or "").lower(), _WAN22_VAE_SPATIAL_STRIDE)

_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def _make_bar(total: int, mode: str):
    """A tqdm bar over *windows* (not batches), or ``None`` to use plain periodic
    prints. ``auto`` picks the bar only on a TTY: under Slurm/nohup the output is a
    file, where a bar's carriage returns turn the log into one unreadable line."""
    if mode == "plain":
        return None
    if mode == "auto" and not sys.stderr.isatty():
        return None
    try:
        from tqdm.auto import tqdm  # noqa: PLC0415 — optional dependency
    except ImportError:
        if mode == "bar":
            print("warning: --progress bar requested but tqdm is not installed; using plain prints.")
        return None
    return tqdm(total=total, unit="win", desc="encoding", dynamic_ncols=True, smoothing=0.05)


def _load_vae_only(ckpt_dir: Path, device: str, provider: str = "wan2.2", model_path: str | None = None):
    """Instantiate ONLY the provider's VAE (skips the DiT).

    - ``wan2.2`` / ``wan`` (default): the 48-ch Wan2.2 VAE, mirroring
      ``WanTI2V.__init__``'s construction exactly (byte-identical to before).
    - ``skyreels``: the 16-ch Wan2.1 VAE (``vae_stride=(4,8,8)``) from the
      vendored SkyReels repo. ``model_path`` (a local snapshot) or the HF cache is
      searched for ``Wan2.1_VAE.pth``.

    NOTE (skyreels): swapping the VAE is necessary but NOT sufficient for a real
    SkyReels precompute — SkyReels-I2V is i2v-conditioned (needs ``y``/``clip_fea``
    per window), so the Wan22DiffusionForcingPreprocessor below does not produce a
    training-complete SkyReels batch. A SkyReels preprocessor is the remaining
    piece; this branch exists so the VAE-selection choke point is provider-aware.
    """
    if provider.lower() == "skyreels":
        from generative_flow_adapters.models.base.skyreels_video import _ensure_skyreels_importable  # noqa: PLC0415
        _ensure_skyreels_importable()
        from skyreels_v2_infer.modules import download_model, get_vae  # noqa: PLC0415
        import os  # noqa: PLC0415

        snap = model_path or download_model("Skywork/SkyReels-V2-I2V-1.3B-540P")
        vae_pth = os.path.join(snap, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_pth):
            raise FileNotFoundError(f"Wan2.1_VAE.pth not found in {snap}; pass model.extra.model_path.")
        return get_vae(vae_pth, device=device, weight_dtype=torch.float32)

    _ensure_wan_importable()
    from wan.configs import WAN_CONFIGS  # noqa: PLC0415
    from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: PLC0415

    config = WAN_CONFIGS["ti2v-5B"]
    vae_pth = ckpt_dir / config.vae_checkpoint
    if not vae_pth.exists():
        raise FileNotFoundError(f"{config.vae_checkpoint} not found in {ckpt_dir}; pass --ckpt-dir.")
    return Wan2_2_VAE(vae_pth=str(vae_pth), device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/wan22/diffusion_wan22_avid_i2v_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="MetaWorld HDF5 to encode.")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="metaworld",
                        help="Dataset family. acwm_phys reads a split dir of the HF t1an/ACWM-Phys release.")
    parser.add_argument("--data-dir", default=None,
                        help="acwm_phys only: split directory (contains metadata.pt + episode_*.mp4), "
                             "e.g. /scratch-shared/$USER/acwm-phys/rigid_dynamics/push_block/ind_train")
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.2-TI2V-5B", help="Dir holding Wan2.2_VAE.pth.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--temporal-length", type=int, default=None,
                        help="Override config model.extra.temporal_length (must match the consumer's window).")
    parser.add_argument("--sampling", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--num-windows", type=int, default=16,
                        help="K deterministic windows/episode to cache (must match training's --num-windows). "
                             "0 = unbounded random (not precomputable; errors unless --sampling exhaustive).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-area", type=int, default=None,
                        help="Wan-native resize budget (default: config model.extra.max_area). MUST match training.")
    parser.add_argument("--vae-dtype", default="bf16", choices=sorted(_DTYPES), help="VAE encode precision.")
    parser.add_argument("--latent-cache-dir", default=None,
                        help="Where to write latents. Default: <hdf5>.latents/ (same default as training).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Encode only the first N windows then stop (partial cache; safe to resume later).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the window pool across N independent workers (e.g. a Slurm job array) so a "
                             "40-GPU-hour pass runs as N shorter jobs. Cache keys are content-derived and writes "
                             "are atomic, so shards never collide. Billing is per GPU-hour, so this is free.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process owns, 0 <= index < --num-shards.")
    parser.add_argument("--progress", choices=["auto", "bar", "plain"], default="auto",
                        help="auto: tqdm bar on a TTY, periodic lines when redirected to a log (default).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    # Geometry, exactly as train_wan22_i2v_metaworld_external.py derives it.
    if args.temporal_length is not None:
        config.model.extra["temporal_length"] = int(args.temporal_length)
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    provider = str(config.model.provider).lower()
    vae_stride = _vae_spatial_stride(provider)  # 16 for wan2.2, 8 for skyreels
    target_height = latent_height * vae_stride
    target_width = latent_width * vae_stride
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * vae_stride  # patch(2) * stride (= 32 for wan2.2, 16 for skyreels)
    num_windows = args.num_windows or None  # 0 -> None
    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir (see --help).")
        default_cache = str(Path(args.data_dir)) + ".latents"
    else:
        default_cache = str(Path(args.hdf5).with_suffix("")) + ".latents"
    cache_dir = args.latent_cache_dir or default_cache

    # VAE only — no DiT, no adapter, no trainer. Provider-aware: wan2.2 -> 48-ch
    # Wan2.2 VAE (stride 16); skyreels -> 16-ch Wan2.1 VAE (stride 8).
    vae = _load_vae_only(
        Path(args.ckpt_dir), device, provider=provider, model_path=config.model.extra.get("model_path")
    )
    vae.dtype = _DTYPES[args.vae_dtype]
    print(f"loaded {provider} VAE (only) on {device}, stride={vae_stride}, encode dtype={args.vae_dtype}")

    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(
            target_height=target_height, target_width=target_width,
            max_area=max_area, align_h=align, align_w=align,
            prompt_contexts_path=None,          # cond/text not needed to encode z0
            latent_cache_dir=cache_dir,
        ),
        condition_keys=("act",),
    )

    if args.dataset == "acwm_phys":
        _, dataset = build_acwmphys_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            data_dir=args.data_dir,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
        )
    else:
        _, dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=args.hdf5,
            frame_stride=args.frame_stride,
            sampling=args.sampling,
            num_windows=num_windows,
        )

    # Enumerate exactly the windows training will sample.
    if num_windows is not None and dataset.sampling == "random":
        precompute_set = dataset.fixed_window_enumeration()
    elif dataset.sampling == "exhaustive":
        precompute_set = dataset
    else:
        raise SystemExit(
            "--num-windows 0 with --sampling random has infinitely many windows and cannot be "
            "precomputed. Use --num-windows K>0 (recommended) or --sampling exhaustive."
        )

    # Fan the pool out across independent processes. STRIDED, not contiguous, so
    # every shard draws uniformly from the whole dataset — a contiguous split
    # would hand one shard all the long episodes if the manifest is ordered.
    # Safe to run concurrently: the cache key is derived from clip identity +
    # geometry (data/latent_cache.py:27), so two shards never target the same
    # file, and LatentCache.put writes atomically and skips existing files.
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}")
        from torch.utils.data import Subset  # noqa: PLC0415
        idxs = list(range(args.shard_index, len(precompute_set), args.num_shards))
        print(f"shard {args.shard_index}/{args.num_shards}: {len(idxs)} of {len(precompute_set)} windows")
        precompute_set = Subset(precompute_set, idxs)

    total = len(precompute_set) if args.limit is None else min(args.limit, len(precompute_set))
    if max_area is not None:
        vid = dataset[0]["video"]
        print(f"resize budget max_area={max_area} (align {align}) | source {int(vid.shape[2])}x{int(vid.shape[1])}px"
              f" | {temporal_length}f windows")
    # The probe above opened a video/HDF5 reader in the parent; drop it and use
    # spawn workers — decord/FFmpeg (and h5py) handles are not fork-safe, and
    # fork-after-decord deadlocks the first worker get_batch().
    dataset.translator.close()
    loader = DataLoader(precompute_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False,
                        multiprocessing_context="spawn" if args.num_workers > 0 else None)
    print(f"precomputing latents -> {cache_dir}  ({total} windows"
          + (f", capped from {len(precompute_set)} by --limit" if args.limit is not None else "") + ")")

    bar = _make_bar(total, args.progress)
    done, encoded, t0, last_report = 0, 0, time.time(), 0
    for raw_batch in loader:
        bs, enc = preprocessor.precompute(raw_batch)
        done += bs
        encoded += enc
        if bar is not None:
            bar.update(bs)
            bar.set_postfix(encoded=encoded, cached=done - encoded, refresh=False)
        # Plain fallback (log files / non-tty): report every ~25 windows *since the
        # last report*, not on `done % 25` — with --batch-size 4 the latter only
        # aligns every 100 windows (and never at all for batch sizes coprime with
        # 25), which reads as a hang.
        elif done - last_report >= 25 or done >= total:
            last_report = done
            rate = (time.time() - t0) / max(done, 1)
            eta = rate * (total - done) / 60.0
            print(f"  {done}/{total} windows  (encoded {encoded}, cached {done - encoded}) "
                  f"{rate:.2f}s/clip  eta {eta:.0f}min", flush=True)
        if done >= total:
            break
    if bar is not None:
        bar.close()
    print(f"done: encoded {encoded} new, {done - encoded} already cached "
          f"-> {preprocessor.latent_cache.num_files()} files in {cache_dir}")


if __name__ == "__main__":
    main()
