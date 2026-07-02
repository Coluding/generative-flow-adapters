"""Validate the new plug-and-play ``BaseVideoModel`` design on Wan2.2 TI2V-5B.

Two things are checked here:

1. **Base-only generation is good.** ``WanTI2VVideoModel.generate(frame)`` with no
   adapter delegates to the upstream ``wan.WanTI2V.generate`` — so the output is
   the real Wan2.2 i2v rollout (native resolution, correct diffusion-forcing
   mask, shift/CFG), not our old reimplemented loop. This is the fix for the
   washed-out garbage.

2. **The adapter seam is wired correctly.** We re-run with a ``ProbeAdapter``
   that (a) records how many times it is called and (b) returns a *zero* delta.
   Because composition is additive, a zero delta must reproduce the base rollout
   exactly — proving the adapter is injected into the native loop without
   corrupting it. The call count confirms the seam actually fired every step.

Example:
    python examples/wan22_base_vs_adapted_generation.py \
        --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --frame-num 121 --steps 50 --out-dir outputs/wan22_base_vs_adapted
"""

from __future__ import annotations

import argparse
import os

import imageio.v3 as iio
import numpy as np
import torch
from torch import Tensor, nn

from generative_flow_adapters.models.adapted_model import AdaptedModel
from generative_flow_adapters.models.base.wan_ti2v import WanTI2VVideoModel


class ProbeAdapter(nn.Module):
    """Zero-delta adapter that records that the injection seam fired.

    Returns an all-zeros residual, so (with additive composition)
    ``base + adapter == base``. ``calls`` counts forwards (2 per step under
    Wan's CFG) — a nonzero count proves the adapter was actually invoked inside
    Wan's native denoising loop, routed through ``AdaptedModel``."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def attach_base_model(self, base_model: object) -> None:  # Adapter protocol
        pass

    def forward(self, x_t: Tensor, t: object = None, cond: object | None = None, base_output: Tensor | None = None) -> Tensor:
        self.calls += 1
        return torch.zeros_like(x_t)


def _save_video(video: Tensor, path: str, fps: int) -> None:
    """``[3, N, H, W]`` in ``[-1, 1]`` -> mp4."""
    frames = video.detach().float().clamp(-1, 1).add(1).mul(127.5)
    frames = frames.permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()  # [N, H, W, 3]
    iio.imwrite(path, frames, fps=fps, codec="libx264")
    print(f"  saved {path}  ({frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]})")


def _load_metaworld_frame(hdf5: str, clip_idx: int) -> np.ndarray:
    from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset

    translator = MetaWorldTranslator(hdf5, caption_mode="empty")
    dataset = TranslatedClipDataset(translator, window_width=8, frame_stride=4, sampling="exhaustive")
    clip = dataset[clip_idx]
    frame = np.asarray(clip["video"])[0]  # [H, W, 3] uint8 — first observation frame
    return frame.astype(np.uint8)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5")
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.2-TI2V-5B")
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=121, help="pixel frames (4n+1)")
    parser.add_argument("--max-area", type=int, default=704 * 1280, help="resolution budget (lower = less memory)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--no-offload", action="store_true", help="keep models on GPU (faster, more VRAM)")
    parser.add_argument("--skip-adapted", action="store_true", help="only run base-only generation")
    parser.add_argument("--out-dir", default="outputs/wan22_base_vs_adapted")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame = _load_metaworld_frame(args.hdf5, args.clip_idx)
    iio.imwrite(os.path.join(args.out_dir, "cond_frame.png"), frame)
    print(f"conditioning frame: {frame.shape} from clip {args.clip_idx}")

    model = WanTI2VVideoModel(args.ckpt_dir, offload_model=not args.no_offload)

    gen_kwargs = dict(
        max_area=args.max_area, frame_num=args.frame_num, sampling_steps=args.steps,
        shift=args.shift, guide_scale=args.guide_scale, seed=args.seed,
    )

    print("\n[1/2] base-only generation (delegates to upstream wan.WanTI2V.generate)")
    base_video = model.generate(frame, adapter=None, **gen_kwargs)
    _save_video(base_video, os.path.join(args.out_dir, "base_only.mp4"), args.fps)

    if args.skip_adapted:
        print("\ndone (base only).")
        return

    print("\n[2/2] adapted generation via AdaptedModel + zero-delta ProbeAdapter (must equal base)")
    probe = ProbeAdapter().to(model.wan.device)
    adapted = AdaptedModel(base_model=model, adapter=probe, condition_encoder=None, output_composition="add")
    adapted.eval()
    adapted_video = adapted.generate(frame, cond=None, **gen_kwargs)
    _save_video(adapted_video, os.path.join(args.out_dir, "adapted_zero.mp4"), args.fps)

    max_abs = (adapted_video.float() - base_video.float()).abs().max().item()
    print(f"\nseam check: adapter called {probe.calls} times "
          f"(expect ~{2 * args.steps} = 2 CFG forwards x {args.steps} steps)")
    print(f"zero-adapter |adapted - base| max = {max_abs:.3e}  (should be ~0)")
    ok = probe.calls > 0 and max_abs < 1e-3
    print("RESULT:", "PASS — seam wired, additive identity holds." if ok else "FAIL — see values above.")
    print(f"\ndone -> {args.out_dir}")


if __name__ == "__main__":
    main()
