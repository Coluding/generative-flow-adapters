"""Standalone base-vs-adapted comparison for the Wan2.2 replace investigation.

Loads a trained adapter checkpoint (from ``training.output_dir``/checkpoints),
then on the SAME weights:

1. **Loss (training seam)** — runs ``Trainer.evaluate`` on a few preprocessor
   batches and prints the adapted denoise loss, the frozen-base denoise loss,
   and their delta (the exact ``eval_denoise_adapter_delta`` metric from wandb).
2. **Generation (generation seam)** — runs the native Wan i2v loop from a
   dataset observation frame, once for the frozen base and once for the adapted
   model, and writes a side-by-side mp4 + frame strip (GT | base | adapted).
3. **Seam instrumentation** — hooks ``AdaptedModel._compose_with_adapter`` so
   every adapter invocation (at either seam) records ``t``, ‖base‖, ‖out‖ and
   cos(out, base). Under ``composition: replace`` the composed output IS the
   adapter output, so a healthy adapter should track the base closely
   (cos → 1); cos ≈ 0 at the generation seam but not at the loss seam is the
   smoking gun for out-of-distribution generation-time inputs.

Debugger-friendly: the first adapter call of each phase is dumped to
``<out-dir>/seam_{loss,gen}_first_call.pt`` for offline diffing.

Example (3090: 41 frames instead of the training 97 to fit VRAM):

    python scripts/generate_wan22_i2v_compare.py \
        --config configs/wan22/diffusion_wan22_avid_xattn_replace_metaworld.yaml \
        --checkpoint outputs/replace-metaworld-run/checkpoints/step_00000500.pt \
        --frame-num 41 --num-steps 50

    # CFG isolation (the ×5 prompt-guidance confound):
    python scripts/generate_wan22_i2v_compare.py --guide-scale 1.0 --no-use-prompt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before CUDA init: the 5B DiT + VAE + a cache-miss encode brush the
# 24 GB ceiling, and without expandable segments allocator fragmentation turns
# ~7 GB of nominally free VRAM into an OOM (observed on the 3090).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import (
    Wan22DiffusionForcingPreprocessor,
    WanBatchPreprocessConfig,
    build_acwmphys_clip_dataset,
    build_metaworld_clip_dataset,
)
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer, _call_preprocessor

_WAN22_VAE_SPATIAL_STRIDE = 16


def _latest_checkpoint(config) -> Path | None:
    out_dir = getattr(config.training, "output_dir", None)
    if not out_dir:
        return None
    candidates = sorted(Path(out_dir).glob("checkpoints/*.pt"))
    return candidates[-1] if candidates else None


def _seam_hook(model, records: list[dict], phase: dict, dump_dir: Path):
    """Wrap ``model._compose_with_adapter`` to record per-call stats. Fires at
    BOTH seams: ``forward()`` (loss eval) and ``generate()``'s ``_compose_step``."""
    orig = model._compose_with_adapter

    def hooked(x_t: Tensor, t, cond, base_output: Tensor) -> Tensor:
        out = orig(x_t, t, cond, base_output)
        with torch.no_grad():
            t_tensor = torch.as_tensor(t).float()
            b, o = base_output.float().flatten(), out.float().flatten()
            rec = {
                "phase": phase["name"],
                "t_max": float(t_tensor.max()),
                "t_min": float(t_tensor.min()),
                "t_shape": tuple(torch.as_tensor(t).shape),
                "x_std": float(x_t.float().std()),
                "base_norm": float(b.norm()),
                "out_norm": float(o.norm()),
                "cos_out_base": float(F.cosine_similarity(o, b, dim=0)),
                "rel_diff": float((o - b).norm() / b.norm().clamp_min(1e-8)),
            }
            records.append(rec)
            if not phase.get(f"dumped_{phase['name']}"):
                phase[f"dumped_{phase['name']}"] = True
                action = cond.get("action") if isinstance(cond, dict) else None
                torch.save(
                    {"x_t": x_t.cpu(), "t": torch.as_tensor(t).cpu(),
                     "action": action.cpu() if isinstance(action, Tensor) else None,
                     "base_output": base_output.cpu(), "composed": out.detach().cpu()},
                    dump_dir / f"seam_{phase['name']}_first_call.pt",
                )
        return out

    model._compose_with_adapter = hooked


def _print_seam_summary(records: list[dict]) -> None:
    phases = sorted({r["phase"] for r in records})
    print("\n=== Seam stats (adapter invocations) ===")
    print(f"{'phase':10} {'calls':>5} {'t range':>16} {'|base|':>9} {'|out|':>9} {'cos(out,base)':>14} {'rel_diff':>9}")
    for ph in phases:
        rows = [r for r in records if r["phase"] == ph]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        t_lo, t_hi = min(r["t_min"] for r in rows), max(r["t_max"] for r in rows)
        print(f"{ph:10} {len(rows):>5} {f'{t_lo:.0f}..{t_hi:.0f}':>16} {mean('base_norm'):>9.2f} "
              f"{mean('out_norm'):>9.2f} {mean('cos_out_base'):>14.3f} {mean('rel_diff'):>9.3f}")


def _print_gen_calls(records: list[dict]) -> None:
    rows = [r for r in records if r["phase"] == "gen"]
    if not rows:
        return
    keep = rows if len(rows) <= 24 else rows[:8] + rows[8 :: max(1, len(rows) // 12)]
    print("\n--- generation-seam calls (per DiT invocation; CFG runs 2/step) ---")
    print(f"{'#':>4} {'t_max':>8} {'x_std':>7} {'|base|':>9} {'|out|':>9} {'cos':>7} {'rel':>7}")
    for r in keep:
        idx = rows.index(r)
        print(f"{idx:>4} {r['t_max']:>8.1f} {r['x_std']:>7.3f} {r['base_norm']:>9.2f} "
              f"{r['out_norm']:>9.2f} {r['cos_out_base']:>7.3f} {r['rel_diff']:>7.3f}")


def _masked_mse(pred: Tensor, target: Tensor, frame_mask: Tensor) -> float:
    """Trainer._frame_masked_mse, standalone: MSE over predicted frames only."""
    m = frame_mask.to(device=pred.device, dtype=pred.dtype)
    m = m.view(m.shape[0], 1, m.shape[1], 1, 1).expand_as(pred)
    sq = (pred.float() - target.float()) ** 2
    return float(((sq * m).sum() / m.sum().clamp_min(1.0)).cpu())


def _sigma_sweep(model, preprocessor, dataset, args, config, timestep_scale: float, out_dir: Path) -> None:
    """Per-σ loss breakdown: adapted vs frozen-base denoise loss at fixed noise
    levels, on the same clips/noise. The training objective samples σ~U(0,1), so
    the scalar eval loss averages over noise levels; this sweep un-averages it.
    Reads ``x0``/``frame_mask``/``cond`` from preprocessed eval batches and
    rebuilds ``x_t``/``t`` at each σ exactly as the Wan2.2 diffusion-forcing
    preprocessor does (obs frames clean at t=0, future frames at σ·scale)."""
    sigmas = [float(s) for s in args.sweep_sigmas.split(",")]
    amp = str(config.training.extra.get("amp_dtype", "")).lower()
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp in ("bf16", "bfloat16")
        else torch.amp.autocast(device_type="cuda", enabled=False)
    )

    loader = DataLoader(dataset, batch_size=args.loss_batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=True)
    batches: list[dict] = []
    for raw in loader:
        b = _call_preprocessor(preprocessor, raw, train=False)
        x0, fm, cond = b["x0"], b["frame_mask"], b["cond"]
        if not (isinstance(x0, Tensor) and isinstance(fm, Tensor)):
            raise RuntimeError("sigma sweep needs the Wan2.2 preprocessor's x0/frame_mask outputs.")
        batches.append({"x0": x0, "frame_mask": fm, "cond": cond})
        if len(batches) >= args.sweep_batches:
            break
    print(f"\n=== σ sweep: {len(sigmas)} levels x {len(batches)} batches x {args.sweep_draws} noise draws ===")

    # Action-override variants. "true" = the clip's own actions; "shuffle" =
    # actions from a DIFFERENT clip (realistic but mismatched — the cleanest
    # test of whether the model *uses* actions); "zero" = null actions. The
    # frozen base ignores actions, so its loss is identical across variants —
    # any adapted-loss gap between variants is pure action sensitivity.
    _ACTION_KEYS = ("action", "action_seq")
    variants = ["true", "shuffle", "zero"] if args.action_probe else ["true"]

    def _variant_cond(cond: object, idx: int, variant: str) -> object:
        if variant == "true" or not isinstance(cond, dict):
            return cond
        out = dict(cond)
        for key in _ACTION_KEYS:
            v = out.get(key)
            if not isinstance(v, Tensor):
                continue
            if variant == "zero":
                out[key] = torch.zeros_like(v)
            else:  # shuffle: take the same key from the next clip in the pool
                donor = batches[(idx + 1) % len(batches)]["cond"]
                out[key] = donor[key] if isinstance(donor, dict) and isinstance(donor.get(key), Tensor) else torch.zeros_like(v)
        return out

    rows: list[dict] = []
    gen = torch.Generator(device=x0.device).manual_seed(int(args.seed))
    for sigma in sigmas:
        vals: dict[str, list[float]] = {}
        for i, b in enumerate(batches):
            x0, fm, cond = b["x0"], b["frame_mask"], b["cond"]
            fm5 = fm.view(fm.shape[0], 1, fm.shape[1], 1, 1).to(x0.dtype)
            for _ in range(args.sweep_draws):
                # One noise draw shared by all variants -> paired comparison.
                noise = torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
                target = noise - x0
                x_noised = (1.0 - sigma) * x0 + sigma * noise
                x_t = (1.0 - fm5) * x0 + fm5 * x_noised
                t = (fm * (sigma * timestep_scale)).to(dtype=x_t.dtype)
                for variant in variants:
                    with torch.no_grad(), autocast:
                        pred, base = model(x_t, t, _variant_cond(cond, i, variant), return_base=True)
                    pred, base = pred.float(), base.float()
                    p = "" if variant == "true" else f"{variant}_"
                    vals.setdefault(f"{p}adapted", []).append(_masked_mse(pred, target, fm))
                    if variant == "true":
                        vals.setdefault("base", []).append(_masked_mse(base, target, fm))
                        m = fm5.expand_as(pred).bool()
                        pm, bm = pred[m], base[m]
                        vals.setdefault("rel", []).append(float(((pm - bm).norm() / bm.norm().clamp_min(1e-8)).cpu()))
                        vals.setdefault("cos", []).append(float(F.cosine_similarity(pm.flatten(), bm.flatten(), dim=0).cpu()))
        row = {"sigma": sigma}
        for k, v in vals.items():
            tv = torch.tensor(v)
            row[k] = float(tv.mean())
            row[f"{k}_std"] = float(tv.std()) if len(v) > 1 else 0.0
        row["delta"] = row["base"] - row["adapted"]
        rows.append(row)
        line = (f"  σ={sigma:0.2f}  adapted {row['adapted']:8.5f} ±{row['adapted_std']:0.5f}"
                f"   base {row['base']:8.5f} ±{row['base_std']:0.5f}"
                f"   Δ(base-adapted) {row['delta']:+8.5f}"
                f"   rel {row['rel']:0.4f}   cos {row['cos']:0.4f}")
        if args.action_probe:
            row["shuffle_gap"] = row["shuffle_adapted"] - row["adapted"]
            row["zero_gap"] = row["zero_adapted"] - row["adapted"]
            line += (f"   | shuffle {row['shuffle_adapted']:8.5f} (gap {row['shuffle_gap']:+.5f})"
                     f"   zero {row['zero_adapted']:8.5f} (gap {row['zero_gap']:+.5f})")
        print(line)

    csv_path = out_dir / "sigma_sweep.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6g}" for k in keys) + "\n")
    print(f"  wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipped plot)")
        return
    xs = [r["sigma"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].errorbar(xs, [r["adapted"] for r in rows], yerr=[r["adapted_std"] for r in rows],
                     label="adapted", marker="o")
    axes[0].errorbar(xs, [r["base"] for r in rows], yerr=[r["base_std"] for r in rows],
                     label="frozen base", marker="s")
    axes[0].set(xlabel="σ", ylabel="masked denoise MSE", title="loss vs noise level")
    axes[0].legend()
    axes[1].axhline(0.0, color="gray", lw=0.8)
    axes[1].errorbar(xs, [r["delta"] for r in rows], marker="o", color="tab:green", label="base − adapted")
    if "shuffle_gap" in rows[0]:
        axes[1].plot(xs, [r["shuffle_gap"] for r in rows], marker="^", color="tab:red",
                     label="action gap (shuffled − true)")
        axes[1].plot(xs, [r["zero_gap"] for r in rows], marker="v", color="tab:orange",
                     label="action gap (zeroed − true)")
    axes[1].legend()
    axes[1].set(xlabel="σ", ylabel="loss difference", title="adapter advantage & action sensitivity")
    axes[2].plot(xs, [r["rel"] for r in rows], marker="o", label="rel |pred−base|/|base|")
    axes[2].plot(xs, [r["cos"] for r in rows], marker="s", label="cos(pred, base)")
    axes[2].set(xlabel="σ", title="how far from a base-copy")
    axes[2].legend()
    fig.tight_layout()
    png_path = out_dir / "sigma_sweep.png"
    fig.savefig(png_path, dpi=140)
    print(f"  wrote {png_path}")


def _pre_encode_windows(dataset, args, pre_cfg: WanBatchPreprocessConfig) -> None:
    """GPU-encode the windows this run will read into the latent cache BEFORE
    the 5B is built: a native-res (1280x704) window encode needs ~15 GiB —
    trivial on an empty GPU, impossible next to the resident DiT. Loads a
    standalone VAE (~seconds), encodes only the needed windows (already-cached
    ones are skipped by the cache layer), frees it. Requires the fixed window
    pool (--num-windows K>0); unbounded random sampling can't be pre-encoded
    (the CPU phase-1 shim at rollout time remains the fallback there)."""
    if dataset.num_windows is None or not torch.cuda.is_available():
        return
    needed = {args.clip_index}
    if args.loss_batches > 0:
        needed |= set(range(min(len(dataset), args.loss_batches * args.loss_batch_size)))
    if getattr(args, "sigma_sweep", False):
        # The sweep draws a random window from each of the first sweep_batches
        # episodes' pools — cover all their fixed starts.
        needed |= set(range(min(len(dataset), args.sweep_batches * args.loss_batch_size)))
    enum = dataset.fixed_window_enumeration()
    order = {id(ep): i for i, ep in enumerate(dataset.episodes)}
    idxs = [i for i, (ep, _s) in enumerate(enum._pairs) if order[id(ep)] in needed]
    if not idxs:
        return
    from precompute_latents import _load_vae_only  # noqa: PLC0415 — sibling script, heavy import

    vae0 = _load_vae_only(Path(args.ckpt_dir), "cuda")
    vae0.dtype = torch.bfloat16
    pre0 = Wan22DiffusionForcingPreprocessor(vae=vae0, config=pre_cfg, condition_keys=("act",), cond_frames=1)
    newly = 0
    try:
        for raw in DataLoader(Subset(enum, idxs), batch_size=1):
            _bs, enc = pre0.precompute(raw)
            newly += enc
        print(f"pre-encode (GPU, before the 5B build): {len(idxs)} windows ready "
              f"({newly} newly encoded) -> {pre_cfg.latent_cache_dir}")
    except torch.OutOfMemoryError:
        print("pre-encode: GPU OOM — falling back to the CPU encode shim at rollout time")
    finally:
        del pre0, vae0
        torch.cuda.empty_cache()


def _to_uint8_frames(px: Tensor) -> "list":
    """``[3, N, H, W]`` in [-1,1] -> list of ``[H, W, 3]`` uint8 frames."""
    v = px.detach().float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    return [v[:, i].permute(1, 2, 0).cpu().numpy() for i in range(v.shape[1])]


def _gt_frames(clip: Tensor, out_h: int, out_w: int, n: int) -> "list":
    v = clip[:n].float().permute(0, 3, 1, 2)  # [n, 3, H, W]
    v = F.interpolate(v, size=(out_h, out_w), mode="bilinear", align_corners=False)
    v = v.round().clamp(0, 255).to(torch.uint8)
    return [v[i].permute(1, 2, 0).cpu().numpy() for i in range(v.shape[0])]


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
    parser.add_argument("--config", default="configs/wan22/diffusion_wan22_avid_xattn_replace_metaworld.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter checkpoint (.pt). Default: newest in <output_dir>/checkpoints/.")
    parser.add_argument("--random-init", action="store_true",
                        help="Skip the checkpoint and randomly perturb the adapter instead. The zero-init "
                        "head outputs exactly 0, which under `replace` freezes the rollout at noise by "
                        "construction — the perturbation makes outputs non-zero so the seam input dumps / "
                        "path plumbing can be exercised without trained weights.")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--latent-cache-dir", default=None,
                        help="Latent cache dir (default: <data-dir>.latents / <hdf5>.latents). Point at the "
                             "shared training cache to reuse its latents; encodes+caches on miss either way.")
    parser.add_argument("--dataset", choices=["metaworld", "acwm_phys"], default="metaworld",
                        help="Dataset family. acwm_phys reads a split dir of the HF t1an/ACWM-Phys release.")
    parser.add_argument("--data-dir", default=None,
                        help="acwm_phys only: split directory (metadata.pt + episode_*.mp4).")
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.2-TI2V-5B")
    parser.add_argument("--clip-index", type=int, default=0, help="Dataset episode index for the rollout clip.")
    parser.add_argument("--num-steps", type=int, default=50, help="Solver steps for the rollout.")
    parser.add_argument("--frame-num", type=int, default=None,
                        help="Pixel frames to generate. Default: config inference_frame_num. "
                        "41 fits the 3090 (training used the config temporal_length).")
    parser.add_argument("--guide-scale", type=float, default=None, help="Override inference_guide_scale.")
    parser.add_argument("--use-prompt", action=argparse.BooleanOptionalAction, default=None,
                        help="Text-CFG on the base loop. Default: config inference_use_prompt.")
    parser.add_argument("--max-area", type=int, default=None,
                        help="Override the pixel-area budget (preprocessor + generation). Must match "
                        "training or the latent cache misses and a full VAE encode runs (OOM risk).")
    parser.add_argument("--temporal-length", type=int, default=None,
                        help="Pixel-frame window width (dataset + preprocessor). Must match training "
                        "or the latent cache misses. Default: config model.extra.temporal_length.")
    parser.add_argument("--loss-batches", type=int, default=4, help="Batches for the loss comparison (0 skips).")
    parser.add_argument("--loss-batch-size", type=int, default=1)
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for loss/sweep loaders. Default 0: worker processes "
                             "deadlock with decord after CUDA init (known ACWM issue) and buy nothing "
                             "for a diagnostic script.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-level", type=float, default=None,
                        help="Optional step_level injected into the adapter cond (matches the eval grid).")
    parser.add_argument("--decode-cpu", action=argparse.BooleanOptionalAction, default=None,
                        help="Run the final VAE decode on CPU. The 768x768 x41f decode needs >24 GB "
                        "GPU (measured: OOM on the 3090 with NOTHING else resident), so this is "
                        "auto-enabled on <32 GB cards at max_area >= 400k. Slow (minutes) but keeps "
                        "the solver at the trained resolution instead of degrading it.")
    parser.add_argument("--out-dir", default="outputs/replace_debug")
    parser.add_argument("--sigma-sweep", action="store_true",
                        help="Per-σ loss breakdown (adapted vs base at fixed noise levels), then exit — no rollout.")
    parser.add_argument("--sweep-sigmas", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    parser.add_argument("--sweep-batches", type=int, default=6)
    parser.add_argument("--sweep-draws", type=int, default=2, help="Fresh noise draws per (σ, batch).")
    parser.add_argument("--action-probe", action="store_true",
                        help="With --sigma-sweep: also evaluate adapted loss under shuffled (other-clip) "
                             "and zeroed actions, same noise. Gap ≈ 0 at every σ ⇒ the model ignores actions "
                             "(or the data makes them redundant).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (upstream WanTI2V pins cuda).")
    device = "cuda"
    config = load_config(args.config)

    # --- build exactly like scripts/train_wan22_i2v_metaworld_external.py ---
    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "Wan2.2_VAE.pth").exists():
        raise FileNotFoundError(f"Wan2.2_VAE.pth not found in {ckpt_dir}.")
    config.model.provider = "wan2.2_external"
    config.model.pretrained_model_name_or_path = str(ckpt_dir)
    config.model.extra["offload_model"] = False
    wandb_cfg = config.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False  # debug script: never log to wandb

    temporal_length = int(args.temporal_length or config.model.extra.get("temporal_length", 17))
    latent_height = int(config.model.extra.get("latent_height", 16))
    latent_width = int(config.model.extra.get("latent_width", 16))
    max_area = args.max_area if args.max_area is not None else config.model.extra.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE

    prompt_contexts_path = None
    prompts_file = config.model.extra.get("text_prompts_file")
    if prompts_file:
        p = Path(prompts_file).with_suffix(".contexts.pt")
        if p.exists():
            prompt_contexts_path = str(p)

    # Resolve the checkpoint BEFORE the expensive 5B build so a bad path fails fast.
    payload = None
    if not args.random_init:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No adapter checkpoint found (looked for --checkpoint / <output_dir>/checkpoints). "
                f"Got: {ckpt_path}. Pass --random-init to run without trained weights."
            )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ---- dataset + GPU pre-encode BEFORE the 5B build ---------------------
    # (order matters: window encodes need the GPU to themselves at native res)
    condition_keys = tuple(spec.key for spec in config.conditioning.conditions if spec.key != "step_level")
    timestep_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    action_per_frame = bool(config.training.extra.get("action_per_frame", False))
    latent_frames = 1 + (temporal_length - 1) // 4
    default_cache = (
        str(Path(args.data_dir)) + ".latents"
        if args.dataset == "acwm_phys" and args.data_dir
        else str(Path(args.hdf5).with_suffix("")) + ".latents"
    )
    latent_cache_dir = args.latent_cache_dir or default_cache
    preprocess_cfg = WanBatchPreprocessConfig(
        target_height=latent_height * _WAN22_VAE_SPATIAL_STRIDE,
        target_width=latent_width * _WAN22_VAE_SPATIAL_STRIDE,
        timestep_scale=timestep_scale,
        max_area=max_area, align_h=align, align_w=align,
        prompt_contexts_path=prompt_contexts_path,
        latent_cache_dir=latent_cache_dir,
        action_per_frame=action_per_frame,
        action_seq_len=(latent_frames if action_per_frame else None),
    )

    if args.dataset == "acwm_phys":
        if not args.data_dir:
            raise SystemExit("--dataset acwm_phys requires --data-dir (a split dir of the HF release).")
        _, dataset = build_acwmphys_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            data_dir=args.data_dir,
            frame_stride=int(config.data.frame_stride or 1),
            sampling="random",
            num_windows=args.num_windows or None,
        )
    else:
        _, dataset = build_metaworld_clip_dataset(
            config.data,
            default_window_width=temporal_length,
            hdf5=args.hdf5,
            frame_stride=int(config.data.frame_stride or 1),
            sampling="random",
            num_windows=args.num_windows or None,
        )
    if not (0 <= args.clip_index < len(dataset)):
        raise ValueError(f"--clip-index {args.clip_index} out of range (dataset len {len(dataset)}).")
    _pre_encode_windows(dataset, args, preprocess_cfg)

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    model.eval()
    if payload is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        loaded = len(payload["model"])
        print(f"checkpoint: {ckpt_path}  (global_step={payload.get('global_step')})")
        print(f"  loaded {loaded} trainable tensors  (missing={len(missing)} frozen-base keys expected, "
              f"unexpected={len(unexpected)})")
        if unexpected:
            raise RuntimeError(f"checkpoint has {len(unexpected)} keys the model doesn't: {unexpected[:5]} ...")
    else:
        with torch.no_grad():
            for p in model.adapter.parameters():
                p.add_(0.02 * torch.randn_like(p))
        print("RANDOM-INIT MODE: no checkpoint loaded; adapter perturbed (outputs non-zero but untrained).")

    vae = model.base_model.wan.vae
    vae.dtype = torch.bfloat16  # match training (speeds encode/decode; latents return fp32)

    # Same geometry/cache config as the pre-encode pass -> guaranteed key hits.
    preprocessor = Wan22DiffusionForcingPreprocessor(
        vae=vae,
        config=preprocess_cfg,
        condition_keys=condition_keys or ("act",),
        cond_frames=int(config.training.extra.get("cond_frames", 1)),
        cond_frames_dist=config.training.extra.get("cond_frames_dist"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    phase = {"name": "loss"}
    _seam_hook(model, records, phase, out_dir)

    # ---- 1) loss comparison at the training seam --------------------------
    if args.loss_batches > 0:
        trainer = Trainer(model, experiment.optimizer, experiment.loss_fn, config.training)
        loader = DataLoader(dataset, batch_size=args.loss_batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)
        stats = trainer.evaluate(loader, max_batches=args.loss_batches, preprocessor=preprocessor)
        print(f"\n=== Denoise loss over {args.loss_batches} batches (training seam) ===")
        print(f"  adapted denoise loss : {stats.get('eval_base_loss', float('nan')):.5f}")
        print(f"  base    denoise loss : {stats.get('eval_denoise_base_only', float('nan')):.5f}")
        print(f"  delta (base-adapted) : {stats.get('eval_denoise_adapter_delta', float('nan')):+.5f}  (>0 = adapter better)")
        rel = stats.get("eval_adapter_rel_contribution")
        if rel is not None:
            print(f"  adapter_rel_contribution: {rel:.4f}  (|pred-base|/|base|; replace: 0 = clone of base)")
        total = stats.get("eval_loss")
        if total is not None:
            print(f"  total eval loss (incl. shortcut terms): {total:.5f}")

    # ---- 1b) per-σ loss breakdown (exits before the rollout machinery) ----
    if args.sigma_sweep:
        phase["name"] = "sweep"
        _sigma_sweep(model, preprocessor, dataset, args, config, timestep_scale, out_dir)
        return

    # ---- 2) native i2v rollout: base vs adapted ---------------------------
    gen_max_area = int(args.max_area or config.training.extra.get("inference_max_area") or max_area or 704 * 1280)
    decode_cpu = args.decode_cpu
    if decode_cpu is None:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        decode_cpu = total_gb < 32 and gen_max_area >= 400_000
    # SCOPED CPU-VAE handling (2026-07-23, bisection finding): the base
    # generates coherent video only when the VAE is in its native GPU/bf16
    # state DURING the rollout — a whole-run CPU shim (device/dtype attrs
    # reassigned before generate) broke the i2v conditioning path and gave
    # pure-noise rollouts at native ACWM geometry, while the identical call
    # with an untouched VAE worked (jobs/experiments/local_probe_acwm_wrapper.py).
    # So: phase 1 CPU-encode (preprocessing only, then FULLY restore), phase 2
    # lazy decode-only shim that flips state after sampling finishes.
    wan_vae = model.base_model.wan.vae
    if decode_cpu:
        # Phase 1 — preprocessing under a temporary CPU VAE: a latent-cache
        # MISS at native res needs ~15 GiB transient, impossible next to the
        # resident 5B; on CPU it's fp32 and slow (minutes/window) but fits.
        saved = (wan_vae.model, list(wan_vae.scale), wan_vae.dtype, wan_vae.device,
                 wan_vae.decode, wan_vae.encode)
        wan_vae.model = wan_vae.model.to("cpu")
        wan_vae.scale = [s.cpu() if isinstance(s, Tensor) else s for s in wan_vae.scale]
        wan_vae.dtype = torch.float32
        wan_vae.device = torch.device("cpu")
        _enc = saved[5]
        wan_vae.encode = lambda vs: [
            z.to(device) for z in _enc([v.detach().float().cpu() for v in vs])
        ]
        print("preprocessing: VAE temporarily on CPU (cache miss encodes fp32, slow; hits skip)")

    # Preprocess the rollout clip (cache hit -> instant).
    raw_batch = next(iter(DataLoader(Subset(dataset, [args.clip_index]), batch_size=1)))
    batch = _call_preprocessor(preprocessor, raw_batch, train=False)

    if decode_cpu:
        # Restore the native GPU/bf16 VAE state for the rollouts — the
        # conditioning-frame encode (1 frame) fits on GPU, and generation
        # must see the VAE exactly as upstream built it.
        wan_vae.model = saved[0].to(device).to(saved[2])
        wan_vae.scale = [s.to(device) if isinstance(s, Tensor) else s for s in saved[1]]
        wan_vae.dtype = saved[2]
        wan_vae.device = saved[3]
        wan_vae.decode, wan_vae.encode = saved[4], saved[5]

        # Phase 2 — decode-only CPU shim (probe-validated): per decode call,
        # flip to CPU fp32, decode, then RESTORE GPU/bf16 — the script runs
        # TWO rollouts (base, adapted) and the second one's conditioning
        # encode + generation need the native GPU state back.
        _gpu_decode = wan_vae.decode

        def _cpu_roundtrip_decode(zs):
            print("decode: CPU fp32 (VAE restored to GPU/bf16 afterwards)", flush=True)
            wan_vae.model = wan_vae.model.to("cpu").float()
            wan_vae.scale = [s.cpu() if isinstance(s, Tensor) else s for s in wan_vae.scale]
            wan_vae.dtype = torch.float32
            try:
                return _gpu_decode([z.detach().float().cpu() for z in zs])
            finally:
                wan_vae.model = wan_vae.model.to(device).to(saved[2])
                wan_vae.scale = [s.to(device) if isinstance(s, Tensor) else s for s in wan_vae.scale]
                wan_vae.dtype = saved[2]

        wan_vae.decode = _cpu_roundtrip_decode
        print("rollouts: VAE on GPU/bf16; each final decode round-trips through CPU")
    cond = batch.get("cond") if isinstance(batch.get("cond"), dict) else {}
    actions = cond.get("action")
    e = config.training.extra
    use_prompt = args.use_prompt if args.use_prompt is not None else bool(e.get("inference_use_prompt", False))
    context = context_null = None
    ctx_val = cond.get("context")
    # The preprocessor emits context as a list of per-sample [L, C] embeddings
    # (variable length) OR a stacked tensor — accept both.
    if use_prompt and isinstance(ctx_val, (Tensor, list, tuple)) and len(ctx_val) > 0:
        context = ctx_val[0]
        if prompt_contexts_path:
            table = torch.load(prompt_contexts_path, map_location="cpu", weights_only=False)
            neg = table.get("negative")
            context_null = neg if isinstance(neg, Tensor) else None
    elif use_prompt:
        print("WARNING: prompt requested but the preprocessed cond has no 'context' — running prompt-free. "
              "(Remote runs used real CFG at guide 5.0; prompt-free is NOT a faithful reproduction.)")

    video = raw_batch["video"]  # [1, T, H, W, 3] raw pixels
    frame = video[0, 0]
    adapted_cond = None
    if isinstance(actions, Tensor):
        adapted_cond = {"action": actions[0:1].to(device)}
        # ROOT-CAUSE FIX (2026-07-20): the xattn adapter trains on per-frame
        # `action_seq` tokens; omitting them here triggers the aggregated-action
        # fallback (one summed token, values ~25) and collapses the adapter
        # output (cos 0.997 -> 0.63 measured). Pass the sequence like training.
        aseq = cond.get("action_seq")
        if isinstance(aseq, Tensor):
            adapted_cond["action_seq"] = aseq[0:1].to(device)
        if args.step_level is not None:
            key = str(config.training.extra.get("shortcut_step_level_key", "step_level"))
            adapted_cond[key] = torch.full((1,), float(args.step_level), device=device)
    else:
        print("WARNING: no action in the preprocessed cond — adapter runs unconditioned.")

    # Explicit --temporal-length also drives the rollout length (in-distribution);
    # --frame-num still wins for deliberate off-length probes.
    frame_num = args.frame_num or (args.temporal_length if args.temporal_length else None) \
        or int(e.get("inference_frame_num") or temporal_length)
    if frame_num != temporal_length:
        print(f"NOTE: generating {frame_num} frames but training used {temporal_length} "
              f"(off-distribution length; base row is the control).")
    kw: dict[str, object] = {
        "max_area": gen_max_area,
        "frame_num": frame_num,
        "sampling_steps": int(args.num_steps),
        "shift": float(e.get("inference_shift", 5.0)),
        "guide_scale": float(args.guide_scale if args.guide_scale is not None else e.get("inference_guide_scale", 5.0)),
        "seed": args.seed,
        "offload_model": True,  # DiT -> CPU before the VAE decode (24 GB card)
    }
    if context is not None:
        kw["context"] = context.to(device)
        kw["context_null"] = context_null.to(device) if context_null is not None else None
    print(f"\n=== Rollout: clip {args.clip_index}, {args.num_steps} steps, {frame_num} frames, "
          f"guide_scale={kw['guide_scale']}, prompt={'on' if context is not None else 'off'} ===")

    with torch.no_grad():
        base_px = model.base_model.generate(frame, **kw).cpu()
        torch.cuda.empty_cache()
        phase["name"] = "gen"
        adapted_px = model.generate(frame, cond=adapted_cond, **kw).cpu()
        torch.cuda.empty_cache()

    _print_gen_calls(records)
    _print_seam_summary(records)
    with (out_dir / "seam_records.json").open("w") as fh:
        json.dump(records, fh, indent=1, default=str)

    out_h, out_w = int(base_px.shape[-2]), int(base_px.shape[-1])
    n_frames = min(int(base_px.shape[1]), int(video.shape[1]))
    fps = int((config.training.extra.get("wandb") or {}).get("fps", 5))
    tag = f"clip{args.clip_index}_s{args.num_steps}_g{kw['guide_scale']}"
    _save_outputs(
        out_dir, tag,
        _gt_frames(video[0], out_h, out_w, n_frames),
        _to_uint8_frames(base_px)[:n_frames],
        _to_uint8_frames(adapted_px)[:n_frames],
        fps,
    )
    print(f"\nseam dumps: {out_dir}/seam_loss_first_call.pt, {out_dir}/seam_gen_first_call.pt "
          f"(x_t, t, action, base_output, composed) — load both and diff input stats.")


if __name__ == "__main__":
    main()
