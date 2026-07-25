"""Base-vs-adapted video comparison for the NATIVE DynamiCrafter backbone.

Builds the AVID adapter on top of the frozen **native** DynamiCrafter
(``dynamicrafter_video`` provider — the real vendored lvdm ``LatentVisualDiffusion``
driven by its own ``DDIMSampler``), then renders a ``GT | base | adapted`` strip:

* **base**    — ``model.base_model.generate(batch)`` (no adapter);
* **adapted** — ``model.generate(batch, cond=adapter_cond)`` (adapter injected at
  the ``apply_model`` seam via ``compose_fn``, exactly like the Wan flow);
* **GT**      — VAE reconstruction of the clip.

Generation delegates to lvdm's own loop (correct concat + CLIP cross-attn + fps +
first-frame anchoring + zero-SNR dynamic rescale), so the base row matches upstream
by construction — no reimplemented sampler.

Example:

    python scripts/generate_dynamicrafter_compare.py --random-init --fs 10 --num-steps 50
    python scripts/generate_dynamicrafter_compare.py \\
        --checkpoint outputs/diffusion_avid_shortcut_metaworld_native/checkpoints/step_00002000.pt
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
from einops import rearrange  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

from generative_flow_adapters.config import load_config  # noqa: E402
from generative_flow_adapters.data import build_metaworld_clip_dataset  # noqa: E402
from generative_flow_adapters.training import build_experiment  # noqa: E402

_H, _W = 320, 512


def _latest_checkpoint(config) -> Path | None:
    out_dir = getattr(config.training, "output_dir", None)
    if not out_dir:
        return None
    candidates = sorted(Path(out_dir).glob("checkpoints/*.pt"))
    return candidates[-1] if candidates else None


def _norm_video(video_uint8: Tensor, device, dtype) -> Tensor:
    """``[b, t, h, w, 3]`` uint8 -> ``[b, 3, t, H, W]`` in [-1, 1] resized to 320x512."""
    v = video_uint8.to(torch.float32) / 127.5 - 1.0
    v = v.permute(0, 4, 1, 2, 3).contiguous()  # [b,c,t,h,w]
    b, c, t, _, _ = v.shape
    v = rearrange(v, "b c t h w -> (b t) c h w")
    v = F.interpolate(v, size=(_H, _W), mode="bilinear", align_corners=False)
    v = rearrange(v, "(b t) c h w -> b c t h w", b=b, t=t)
    return v.to(device=device, dtype=dtype)


def _to_frames(px: Tensor) -> "list":
    """``[3, T, H, W]`` in [-1, 1] -> list of ``[H, W, 3]`` uint8 frames."""
    v = px.detach().float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
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
    parser.add_argument("--config", default="configs/dynamicrafter/diffusion_avid_shortcut_metaworld_native.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Adapter checkpoint (.pt). Default: newest in <output_dir>/checkpoints/.")
    parser.add_argument("--random-init", action="store_true",
                        help="Skip the checkpoint and perturb adapter-only params (base untouched). "
                        "Exercises the base-vs-adapted generation path without trained weights.")
    parser.add_argument("--checkpoint-file", dest="ckpt_file", default="ckts/dynami512.ckpt",
                        help="DynamiCrafter .ckpt for the native base (also set in the config).")
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM solver steps.")
    parser.add_argument("--fs", type=int, default=10, help="fps conditioning value (AVID default_fs=10).")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/dynamicrafter_compare")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)
    if config.model.provider.lower() != "dynamicrafter_video":
        raise SystemExit(
            f"This script expects the native provider (config.model.provider == 'dynamicrafter_video'); "
            f"got {config.model.provider!r}. Use --config the *_native.yaml."
        )
    if args.ckpt_file and not config.model.pretrained_model_name_or_path:
        config.model.pretrained_model_name_or_path = args.ckpt_file
    config.model.extra["device"] = str(device)
    wandb_cfg = config.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False

    temporal_length = int(config.model.extra.get("temporal_length", 16))

    # Resolve checkpoint before the expensive build.
    payload = None
    ckpt_path = None
    if not args.random_init:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No adapter checkpoint found (--checkpoint / <output_dir>/checkpoints). Got {ckpt_path}. "
                f"Pass --random-init to run without trained weights."
            )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    experiment = build_experiment(config)
    model = experiment.model.to(device)
    model.eval()
    if payload is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"checkpoint: {ckpt_path}  (global_step={payload.get('global_step')})  "
              f"loaded={len(payload['model'])} missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            raise RuntimeError(f"checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]} ...")
    else:
        with torch.no_grad():
            base_ptrs = {p.data_ptr() for p in model.base_model.parameters()}
            adapter_only = [p for p in model.adapter.parameters() if p.data_ptr() not in base_ptrs]
            for p in adapter_only:
                p.add_(0.02 * torch.randn_like(p))
        print(f"RANDOM-INIT MODE: perturbed {len(adapter_only)} adapter-only params (base untouched).")

    native = model.base_model  # DynamiCrafterVideoModel
    autocast_dtype = getattr(native, "_autocast_dtype", None)

    # ---- dataset clip -> raw batch (video/caption/act/fps) ----------------
    _, dataset = build_metaworld_clip_dataset(
        config.data, default_window_width=temporal_length, hdf5=args.hdf5,
        frame_stride=int(config.data.frame_stride or 1), sampling="random",
    )
    if not (0 <= args.clip_index < len(dataset)):
        raise ValueError(f"--clip-index {args.clip_index} out of range (len {len(dataset)}).")
    raw = next(iter(DataLoader(Subset(dataset, [args.clip_index]), batch_size=1)))

    ac_dtype = autocast_dtype or torch.bfloat16
    video = _norm_video(raw["video"], device, ac_dtype)
    act = raw["act"].to(device=device, dtype=ac_dtype)
    fps = raw["fps"].to(device)
    batch = {"video": video, "caption": raw["caption"], "act": act, "fps": fps}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_kw = dict(ddim_steps=args.num_steps, guidance_scale=args.guidance_scale, fs=args.fs)
    print(f"\n=== Native rollout: clip {args.clip_index}, {args.num_steps} DDIM steps, fs={args.fs} ===")

    with torch.no_grad():
        # GT: VAE reconstruction of the clip.
        gt_px = native.decode(native.encode(video))[0]

        # base: native generation, no adapter.
        base_px = native.generate(batch, compose_fn=None, **gen_kw)[0]

        # adapted: adapter injected via compose_fn. The AVID adapter needs
        # {act, fs, concat}; it ignores the base's cross-attn context (uses its
        # own action embedding + self-attention). concat = first-frame latent
        # replicated (from the SAME native VAE).
        with (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else _null()):
            z0 = native.encode(video)
        concat = z0[:, :, 0:1].repeat(1, 1, z0.shape[2], 1, 1)
        fs_t = torch.full((video.shape[0],), int(args.fs), device=device, dtype=torch.long)
        adapter_cond = {"act": act, "fs": fs_t, "concat": concat}
        adapted_px = model.generate(batch, cond=adapter_cond, **gen_kw)[0]

    fps_out = int((config.training.extra.get("video_logging") or {}).get("fps", 5))
    tag = f"clip{args.clip_index}_s{args.num_steps}_native"
    _save_outputs(out_dir, tag, _to_frames(gt_px), _to_frames(base_px), _to_frames(adapted_px), fps_out)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
