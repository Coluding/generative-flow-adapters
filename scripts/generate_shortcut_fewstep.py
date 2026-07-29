#!/usr/bin/env python
"""Offline few-step video generation from a trained SHORTCUT checkpoint.

The standalone, reproducible twin of the in-training ``eval_step_grid`` videos:
load a shortcut checkpoint (Wan / SkyReels / DynamiCrafter, via the shared
BaseVideoModel interface), and for a few eval clips render a **gt | base |
adapted** grid with **rows = N in the step schedule** — saved to disk as mp4.
Use it for thesis figures and to inspect existing runs (e.g. pzmc2orq Wan,
t4bp8nki DC) WITHOUT restarting them.

It reuses the trainer's own `_native_eval_grid` generation (so the rollout is
identical to training) by swapping the wandb logger for a capturing shim, then
assembles the grid via the shared `assemble_step_size_grids` helper.

`--stepsize-perturb` additionally renders at a WRONG fixed step_level (e.g. 50
DDIM steps but the adapter is told it's a 1-step big jump). A step-BLIND adapter
produces an identical video — the visual twin of `eval_stepsize_effect_rel`.

Example:
    python scripts/generate_shortcut_fewstep.py \
      --config configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml \
      --checkpoint outputs/wan-shortcut-actionfree-robotarm/checkpoints/best.pt \
      --dataset acwm_phys --data-dir ds/acwm-phys/kinematics/robot_arm/ind_train \
      --num-clips 3 --stepsize-perturb --out-dir outputs/fewstep_videos

NOTE: full generation needs a GPU (shortcut rollout does 2-3x base forwards) and
is provider-specific; the Wan path is the primary one. Run remotely via
jobs/experiments_cluster/shortcut_eval/submit_generate_fewstep.sh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.config import load_config
from generative_flow_adapters.training import build_experiment
from generative_flow_adapters.training.trainer import Trainer
from generative_flow_adapters.training.wandb_logger import assemble_step_size_grids

_WAN22_VAE_SPATIAL_STRIDE = 16


class _CaptureShim:
    """Stands in for the wandb logger inside ``_native_eval_grid``: captures the
    step-grid pixel tensors instead of logging them. The eval-grid path only
    reads ``.num_samples`` and calls ``.log_step_size_grid_pixels(...)``; any other
    logger method it might touch is a harmless no-op."""

    def __init__(self, num_samples: int, fps: int) -> None:
        self.num_samples = int(num_samples)
        self.fps = int(fps)
        self.captured: tuple | None = None

    def log_step_size_grid_pixels(self, *, target_pixels, adapted_by_steps, base_by_steps, cond, step) -> None:
        self.captured = (target_pixels, adapted_by_steps, base_by_steps, cond)

    def __getattr__(self, _name):  # log_metrics / log_histogram / log_step_size_grid / ...
        return lambda *a, **k: None


def _schedule_from_arg(spec: str) -> list[dict]:
    """'1,2,4,8,25,50' -> [{num_steps, step_level}], step_level = 1/N (few-step
    convention: N=1 is the biggest jump, step_level 1.0)."""
    out = []
    for tok in spec.split(","):
        n = int(tok.strip())
        out.append({"num_steps": n, "step_level": round(1.0 / n, 6)})
    return out


def _save_grids(captured, out_dir: Path, prefix: str, fps: int) -> list[Path]:
    target_pixels, adapted_by_steps, base_by_steps, _cond = captured
    grids = assemble_step_size_grids(target_pixels, adapted_by_steps, base_by_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio  # noqa: PLC0415 — heavy, import on demand

    paths = []
    for i, (grid, cols, order) in enumerate(grids):
        # grid: uint8 [T, C, H, W] -> [T, H, W, C]
        vid = grid.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        p = out_dir / f"{prefix}_sample{i}.mp4"
        imageio.mimwrite(str(p), vid, fps=fps, quality=8, output_params=["-loglevel", "error"])
        paths.append(p)
        print(f"  saved {p}  (rows: {order} | cols: {cols})")
    return paths


def _build_preprocessor(config, model, args):
    """Provider-aware preprocessor + geometry (mirrors the train scripts)."""
    provider = str(config.model.provider).lower()
    e = config.model.extra
    temporal_length = int(e.get("temporal_length", 17))
    latent_h = int(e.get("latent_height", 16))
    latent_w = int(e.get("latent_width", 16))
    max_area = args.max_area if args.max_area is not None else e.get("max_area")
    max_area = int(max_area) if max_area is not None else None
    align = 2 * _WAN22_VAE_SPATIAL_STRIDE
    ts_scale = float(config.training.extra.get("flow_timestep_scale", 1000.0))
    prompts_file = e.get("text_prompts_file")
    ctx_path = str(Path(prompts_file).with_suffix(".contexts.pt")) if prompts_file else None
    latent_frames = 1 + (temporal_length - 1) // 4

    from generative_flow_adapters.data import WanBatchPreprocessConfig  # noqa: PLC0415

    pp_cfg = WanBatchPreprocessConfig(
        target_height=latent_h * _WAN22_VAE_SPATIAL_STRIDE,
        target_width=latent_w * _WAN22_VAE_SPATIAL_STRIDE,
        timestep_scale=ts_scale,
        max_area=max_area, align_h=align, align_w=align,
        prompt_contexts_path=ctx_path,
        latent_cache_dir=None,
        action_per_frame=bool(config.training.extra.get("action_per_frame", False)),
        action_seq_len=(latent_frames if config.training.extra.get("action_per_frame") else None),
        sigma_shift=None,
    )
    if provider in ("wan2.2", "wan", "wan2.1"):
        from generative_flow_adapters.data import Wan22DiffusionForcingPreprocessor  # noqa: PLC0415
        return Wan22DiffusionForcingPreprocessor(vae=model.base_model.wan.vae, config=pp_cfg)
    if provider == "skyreels":
        from generative_flow_adapters.data import SkyReelsI2VPreprocessor  # noqa: PLC0415
        return SkyReelsI2VPreprocessor(vae=model.base_model.vae, config=pp_cfg)
    if provider == "dynamicrafter_video":
        # DC's grid (_native_dc_eval_grid) drives base.encode/generate on the raw
        # batch; the preprocessor is passed but barely used.
        from generative_flow_adapters.data import DynamiCrafterBatchPreprocessor  # noqa: PLC0415
        return DynamiCrafterBatchPreprocessor(model.base_model, config=pp_cfg)
    raise SystemExit(f"unsupported provider for few-step generation: {provider}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=["acwm_phys", "rt1", "openvid", "metaworld"], default="acwm_phys")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--hdf5", default=None, help="metaworld only")
    ap.add_argument("--num-clips", type=int, default=3)
    ap.add_argument("--out-dir", default="outputs/fewstep_videos")
    ap.add_argument("--step-schedule", default="1,2,4,8,25,50")
    ap.add_argument("--stepsize-perturb", action="store_true",
                    help="Also render at a WRONG fixed step_level (visual step-blindness check).")
    ap.add_argument("--max-area", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    config = load_config(args.config)
    temporal_length = int(config.model.extra.get("temporal_length", 17))

    # eval schedule (rows of the grid) + inference plumbing, mirroring the train script.
    config.training.extra["eval_step_schedule"] = _schedule_from_arg(args.step_schedule)
    e = config.model.extra
    prompts_file = e.get("text_prompts_file")
    ctx_path = str(Path(prompts_file).with_suffix(".contexts.pt")) if prompts_file else None
    config.training.extra["inference_prompt_contexts_path"] = ctx_path
    config.training.extra.setdefault("inference_use_prompt", ctx_path is not None)
    if args.max_area is not None:
        config.training.extra["inference_max_area"] = int(args.max_area)

    experiment = build_experiment(config)
    model = experiment.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    trainer = Trainer(model, experiment.optimizer, experiment.loss_fn, config.training,
                      wandb_logger=_CaptureShim(num_samples=args.num_clips, fps=args.fps))
    payload = trainer.load_checkpoint(args.checkpoint)
    print(f"[fewstep] loaded {args.checkpoint} @ global_step={trainer.global_step}")

    # dataset + eval loader (batch 1: generation is per-clip)
    from generative_flow_adapters.data import (  # noqa: PLC0415
        build_acwmphys_clip_dataset, build_rt1_clip_dataset, build_openvid_clip_dataset,
    )
    _builder = {"rt1": build_rt1_clip_dataset, "openvid": build_openvid_clip_dataset}.get(
        args.dataset, build_acwmphys_clip_dataset)
    translator, dataset = _builder(
        config.data, default_window_width=temporal_length, data_dir=args.data_dir,
        frame_stride=args.frame_stride, sampling="random", num_windows=1,
    )
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    preprocessor = _build_preprocessor(config, model, args)
    out_dir = Path(args.out_dir)

    # 1) the few-step grid (rows = N schedule)
    print(f"[fewstep] rendering N={args.step_schedule} grid for {args.num_clips} clips ...")
    trainer._native_eval_grid(eval_loader, preprocessor=preprocessor)
    if trainer.wandb_logger.captured is None:
        raise SystemExit("no grid captured — check the base supports native generate() + a step schedule.")
    _save_grids(trainer.wandb_logger.captured, out_dir, "fewstep", args.fps)

    # 2) step-size perturbation: render 50 DDIM steps but feed WRONG step_levels.
    if args.stepsize_perturb:
        print("[fewstep] step-size perturbation (50 steps @ wrong step_level; identical rows ⇒ step-blind) ...")
        config.training.extra["eval_step_schedule"] = [
            {"num_steps": 50, "step_level": 1.0},   # told: 1-step big jump
            {"num_steps": 50, "step_level": 0.02},  # told: 50-step fine (correct)
        ]
        trainer.wandb_logger.captured = None
        trainer._native_eval_grid(eval_loader, preprocessor=preprocessor)
        if trainer.wandb_logger.captured is not None:
            _save_grids(trainer.wandb_logger.captured, out_dir, "stepsize_perturb", args.fps)

    # index
    (out_dir / "README.txt").write_text(
        f"few-step videos for {args.checkpoint} (step {trainer.global_step})\n"
        f"fewstep_sample*.mp4 : rows N={args.step_schedule}, cols gt|base|adapted. "
        "adapted@small-N should approach base@50 if few-step works.\n"
        "stepsize_perturb_sample*.mp4 : both rows are 50 DDIM steps but with "
        "different step_level fed to the adapter; identical rows => step-size-blind.\n"
    )
    print(f"[fewstep] done -> {out_dir}")


if __name__ == "__main__":
    main()
