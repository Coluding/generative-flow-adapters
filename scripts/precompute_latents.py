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
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.models.base.wan_ti2v import _ensure_wan_importable

# Wan2.2-VAE spatial stride (vae_stride = (4, 16, 16)); align grid = patch(2)*stride(16).
_WAN22_VAE_SPATIAL_STRIDE = 16

_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def _load_vae_only(ckpt_dir: Path, device: str):
    """Instantiate ONLY the Wan2.2 VAE (skips the 5B DiT that ``wan.WanTI2V`` also
    builds). Mirrors ``WanTI2V.__init__``'s VAE construction exactly."""
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
    parser.add_argument("--config", default="configs/diffusion_wan22_avid_i2v_metaworld.yaml")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="MetaWorld HDF5 to encode.")
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.2-TI2V-5B", help="Dir holding Wan2.2_VAE.pth.")
    parser.add_argument("--frame-stride", type=int, default=1)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    # Geometry, exactly as train_wan22_i2v_metaworld_external.py derives it.
    temporal_length = int(config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    target_height = latent_height * _WAN22_VAE_SPATIAL_STRIDE
    target_width = latent_width * _WAN22_VAE_SPATIAL_STRIDE
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE  # = 32
    num_windows = args.num_windows or None  # 0 -> None
    cache_dir = args.latent_cache_dir or (str(Path(args.hdf5).with_suffix("")) + ".latents")

    # VAE only — no DiT, no adapter, no trainer.
    vae = _load_vae_only(Path(args.ckpt_dir), device)
    vae.dtype = _DTYPES[args.vae_dtype]
    print(f"loaded Wan2.2 VAE (only) on {device}, encode dtype={args.vae_dtype}")

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

    total = len(precompute_set) if args.limit is None else min(args.limit, len(precompute_set))
    loader = DataLoader(precompute_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    if max_area is not None:
        vid = dataset[0]["video"]
        print(f"resize budget max_area={max_area} (align {align}) | source {int(vid.shape[2])}x{int(vid.shape[1])}px"
              f" | {temporal_length}f windows")
    print(f"precomputing latents -> {cache_dir}  ({total} windows"
          + (f", capped from {len(precompute_set)} by --limit" if args.limit is not None else "") + ")")

    done, encoded, t0 = 0, 0, time.time()
    for raw_batch in loader:
        bs, enc = preprocessor.precompute(raw_batch)
        done += bs
        encoded += enc
        if done % 25 == 0 or done >= total:
            print(f"  {done}/{total} windows  (encoded {encoded}, cached {done - encoded}) "
                  f"{(time.time() - t0) / max(done, 1):.2f}s/clip", flush=True)
        if done >= total:
            break
    print(f"done: encoded {encoded} new, {done - encoded} already cached "
          f"-> {preprocessor.latent_cache.num_files()} files in {cache_dir}")


if __name__ == "__main__":
    main()
