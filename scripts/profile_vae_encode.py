"""Profile ONLINE VAE encoding — is pre-encoding actually needed at 768^2?

Pre-encoding was a workaround for the native-1280x704 OOM. At 768^2 (max_area
589824) the transient is far smaller, so online encoding may be fine and the
whole precompute step could be dropped. This script measures, per batch, over
a real dataset:

  - wall time: total preprocess, VAE-encode-only, and the rest (CPU LANCZOS
    resize + host->device), so you can see where the time goes;
  - peak GPU memory (allocated + reserved) during a batch, and the transient
    of a SINGLE-clip encode (the number that decides whether encode can
    coexist with the resident 5B);
  - throughput (clips/s, ms/clip) so you can compare against your training
    step time and decide if the encode would bottleneck the loop.

VAE only — no DiT/adapter/optimizer loaded. The latent cache is DISABLED so
every batch genuinely encodes.

Examples (run on the H100):

    # ACWM push_block, the real training geometry (768^2, 65 frames)
    python scripts/profile_vae_encode.py --dataset acwm_phys \
        --data-dir ds/acwm-phys/rigid_dynamics/push_block/ind_train \
        --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
        --max-area 589824 --temporal-length 65 --batch-size 1 2 4 --num-batches 12

    # MetaWorld, for comparison
    python scripts/profile_vae_encode.py --hdf5 ds/metaworld_corner2.hdf5 \
        --max-area 589824 --temporal-length 65 --batch-size 1 2 4
"""

from __future__ import annotations

import argparse
import statistics
import time
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
from generative_flow_adapters.data.wan_batch_preprocessor import best_output_size

_WAN22_VAE_SPATIAL_STRIDE = 16
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _gb(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{m * 1e3:7.1f} ±{s * 1e3:5.1f} ms  (min {min(xs) * 1e3:.0f}, max {max(xs) * 1e3:.0f})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml")
    p.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="acwm_phys")
    p.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    p.add_argument("--data-dir", default="ds/acwm-phys/rigid_dynamics/push_block/ind_train")
    p.add_argument("--ckpt-dir", default="ckpts/Wan2.2-TI2V-5B")
    p.add_argument("--temporal-length", type=int, default=None, help="Override config temporal_length.")
    p.add_argument("--max-area", type=int, default=None, help="Override config max_area (589824 = 768^2).")
    p.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4], help="Batch sizes to sweep.")
    p.add_argument("--num-batches", type=int, default=10, help="Timed batches per size (after warmup).")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--num-windows", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0: decord+CUDA-safe).")
    p.add_argument("--vae-dtype", default="bf16", choices=sorted(_DTYPES))
    p.add_argument("--with-dit", action="store_true",
                   help="Also load the 5B DiT+adapter first, so peak-memory numbers reflect encoding "
                        "ALONGSIDE a resident training model (the realistic coexistence test).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    config = load_config(args.config)

    if args.temporal_length is not None:
        config.model.extra["temporal_length"] = int(args.temporal_length)
    temporal_length = int(config.model.extra.get("temporal_length", 65))
    latent_h = int(config.model.extra.get("latent_height", 16))
    latent_w = int(config.model.extra.get("latent_width", 16))
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE

    resident_gb = 0.0
    if args.with_dit:
        # Load the full training model first so peak memory reflects coexistence.
        from generative_flow_adapters.training import build_experiment  # noqa: PLC0415

        config.model.provider = "wan2.2_external"
        config.model.pretrained_model_name_or_path = args.ckpt_dir
        config.model.extra["offload_model"] = False
        wc = config.training.extra.get("wandb")
        if isinstance(wc, dict):
            wc["enable"] = False
        model = build_experiment(config).model.to(device)
        model.eval()
        vae = model.base_model.wan.vae
        vae.dtype = _DTYPES[args.vae_dtype]
        resident_gb = _gb(torch.cuda.memory_allocated())
        print(f"resident model (DiT+adapter) on GPU: {resident_gb:.2f} GB allocated")
    else:
        from scripts.precompute_latents import _load_vae_only  # noqa: PLC0415

        vae = _load_vae_only(Path(args.ckpt_dir), device)
        vae.dtype = _DTYPES[args.vae_dtype]

    # Time the VAE encode in isolation by wrapping it.
    enc_times: list[float] = []
    _orig_encode = vae.encode

    def timed_encode(vs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _orig_encode(vs)
        torch.cuda.synchronize()
        enc_times.append(time.perf_counter() - t0)
        return out

    vae.encode = timed_encode

    # Report the resized geometry the VAE actually sees.
    if max_area is not None:
        ow, oh = best_output_size(1024, 1024, align, align, max_area)  # square source (ACWM); MW similar
        latent_frames = 1 + (temporal_length - 1) // 4
        print(f"geometry: {temporal_length}f pixel window -> resized ~{ow}x{oh} -> "
              f"latent {latent_frames}x{oh // _WAN22_VAE_SPATIAL_STRIDE}x{ow // _WAN22_VAE_SPATIAL_STRIDE} "
              f"(max_area {max_area}, vae dtype {args.vae_dtype})")

    def make_dataset():
        pre = Wan22DiffusionForcingPreprocessor(
            vae=vae,
            config=WanBatchPreprocessConfig(
                target_height=latent_h * _WAN22_VAE_SPATIAL_STRIDE,
                target_width=latent_w * _WAN22_VAE_SPATIAL_STRIDE,
                max_area=max_area, align_h=align, align_w=align,
                latent_cache_dir=None,  # DISABLED — every batch encodes
            ),
            condition_keys=("act",), cond_frames=1,
        )
        if args.dataset == "acwm_phys":
            _, ds = build_acwmphys_clip_dataset(
                config.data, default_window_width=temporal_length, data_dir=args.data_dir,
                num_windows=args.num_windows, sampling="random")
        else:
            _, ds = build_metaworld_clip_dataset(
                config.data, default_window_width=temporal_length, hdf5=args.hdf5,
                num_windows=args.num_windows, sampling="random")
        return pre, ds

    pre, dataset = make_dataset()

    print(f"\n{'bs':>3} {'total/batch':>26} {'encode/batch':>26} {'resize+h2d/batch':>26} "
          f"{'ms/clip':>8} {'clips/s':>8} {'peak alloc':>11} {'peak resv':>10}")
    print("-" * 130)

    for bs in args.batch_size:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
        it = iter(loader)
        totals: list[float] = []
        encodes: list[float] = []
        peak_alloc = 0.0
        peak_resv = 0.0
        seen = 0
        for i in range(args.warmup + args.num_batches):
            try:
                raw = next(it)
            except StopIteration:
                it = iter(loader)
                raw = next(it)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            enc_times.clear()
            t0 = time.perf_counter()
            pre(raw, train=True)  # full online preprocess: resize + encode + build batch
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                totals.append(dt)
                encodes.append(sum(enc_times))
                peak_alloc = max(peak_alloc, _gb(torch.cuda.max_memory_allocated()))
                peak_resv = max(peak_resv, _gb(torch.cuda.max_memory_reserved()))
                seen += bs
        other = [t - e for t, e in zip(totals, encodes)]
        ms_per_clip = statistics.mean(totals) / bs * 1e3
        clips_s = bs / statistics.mean(totals)
        print(f"{bs:>3} {_fmt(totals):>26} {_fmt(encodes):>26} {_fmt(other):>26} "
              f"{ms_per_clip:>8.1f} {clips_s:>8.2f} {peak_alloc:>9.2f}GB {peak_resv:>8.2f}GB")

    print("\nVerdict inputs:")
    print("  - 'encode/batch' is the GPU VAE cost; 'resize+h2d' is CPU LANCZOS + transfer (workers hide it).")
    print("  - 'peak alloc' is the encode transient; add your resident training memory (or use --with-dit)")
    print("    to check it fits the card alongside the 5B.")
    print("  - Compare 'ms/clip' to your training step time: if encode << step, online encoding is free")
    print("    with a couple of dataloader workers and pre-encoding can be dropped.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
