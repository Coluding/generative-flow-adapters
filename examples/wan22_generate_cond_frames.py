"""Image-conditioned generation sanity check for the **new plug-and-play**
Wan2.2 stack: ``WanTI2VVideoModel`` (external repo ``BaseVideoModel``) wrapped in
``AdaptedModel`` with the real AVID action adapter — run BEFORE training to
confirm the whole generation path works end to end.

What it does, from a single observation frame (default ``metaworld_frame0.png``):

1. **Base-only rollout** — ``model.base_model.generate(frame)`` delegates to the
   upstream ``wan.WanTI2V.generate`` (native resolution, diffusion-forcing mask,
   shift/CFG). This is the real Wan2.2 i2v output, not a reimplemented loop.
2. **Adapted rollout** — ``AdaptedModel.generate(frame, cond=action)`` injects the
   (untrained) AVID adapter at Wan's ``denoise`` seam every step. Composition is
   additive, so with a residual-initialised adapter the adapted rollout should
   start close to the base one; the max |adapted - base| is reported.

The point is to catch plumbing/shape bugs (the real adapter's timestep + cond
handling inside Wan's native loop) before committing to a training run.

Defaults mirror the upstream ``generate.py`` invocation:
    generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ckpts/Wan2.2-TI2V-5B \
        --convert_model_dtype --image metaworld_frame0.png --offload_model True \
        --sample_guide_scale 1.0 --frame_num 41
i.e. max_area=1280*704, guide_scale=1.0, frame_num=41, offload_model=True,
convert_model_dtype (implied by model.extra.dtype=bf16).

Requires a CUDA device (the upstream WanTI2V pins cuda).

Example:
    python examples/wan22_generate_cond_frames.py \
        --config configs/diffusion_wan22_avid_i2v_metaworld.yaml \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --image metaworld_frame0.png \
        --size 1280*704 --frame-num 41 --guide-scale 1.0 \
        --out-dir outputs/wan22_gen_check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from generative_flow_adapters.adapters.factory import build_adapter
from generative_flow_adapters.conditioning.encoders import build_condition_encoder
from generative_flow_adapters.config import load_config
from generative_flow_adapters.models.adapted_model import AdaptedModel
from generative_flow_adapters.models.base.factory import build_base_model


def _parse_size(size: str | None) -> int | None:
    """``"1280*704"`` / ``"1280x704"`` -> pixel-area budget ``1280*704``."""
    if not size:
        return None
    for sep in ("*", "x", "X", ","):
        if sep in size:
            w, h = size.split(sep, 1)
            return int(w) * int(h)
    raise ValueError(f"--size must be WxH (e.g. 1280*704), got {size!r}")


def _save_video(video: Tensor, path: str, fps: int) -> None:
    """``[3, N, H, W]`` in ``[-1, 1]`` -> mp4."""
    frames = video.detach().float().clamp(-1, 1).add(1).mul(127.5)
    frames = frames.permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()  # [N, H, W, 3]
    iio.imwrite(path, frames, fps=fps, codec="libx264")
    print(f"  saved {path}  ({frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]})")


def _load_frame(image: str | None, hdf5: str, clip_idx: int) -> np.ndarray:
    """Return the conditioning frame ``[H, W, 3]`` uint8 — from an image file when
    given, else the first frame of MetaWorld clip ``clip_idx``."""
    if image:
        return np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset

    translator = MetaWorldTranslator(hdf5, caption_mode="empty")
    dataset = TranslatedClipDataset(translator, window_width=8, frame_stride=4, sampling="exhaustive")
    return np.asarray(dataset[clip_idx]["video"])[0].astype(np.uint8)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffusion_wan22_avid_i2v_metaworld.yaml")
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Wan2.2-TI2V-5B checkpoint DIR (DiT safetensors + Wan2.2_VAE.pth + uncond_context.pt).",
    )
    parser.add_argument(
        "--image",
        default="metaworld_frame0.png",
        help="Conditioning frame (image file). Set empty ('') to pull a frame from --hdf5 instead.",
    )
    parser.add_argument("--hdf5", default="ds/metaworld_corner2.hdf5", help="Fallback frame source when --image is empty.")
    parser.add_argument("--clip-idx", type=int, default=0, help="Clip index when sourcing the frame from --hdf5.")
    # --- Wan native sampling params (defaults mirror the generate.py command) ---
    parser.add_argument("--size", default="1280*704", help="Resolution budget WxH -> max_area (Wan best_output_size).")
    parser.add_argument("--max-area", type=int, default=None, help="Explicit pixel-area budget (overrides --size).")
    parser.add_argument("--frame-num", type=int, default=41, help="Pixel frames to generate (4n+1).")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps (unipc).")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-schedule shift (Wan2.2 TI2V-5B native = 5.0).")
    parser.add_argument("--guide-scale", type=float, default=1.0, help="CFG scale (native command uses 1.0 = no CFG).")
    parser.add_argument(
        "--offload-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload the DiT/T5 to CPU between stages to save VRAM (--no-offload-model keeps them resident).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--action",
        type=float,
        nargs="+",
        default=None,
        help="Constant action fed to the adapter across the rollout (len = conditioning.input_dim). Default: zeros.",
    )
    parser.add_argument("--base-only", action="store_true", help="Only run base-only generation (skip the adapter).")
    parser.add_argument("--out-dir", default="outputs/wan22_gen_check")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA device: the upstream WanTI2V base pins cuda.")
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.config)

    # --- NEW building approach: external-repo base + real AVID adapter ---------
    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "Wan2.2_VAE.pth").exists():
        raise FileNotFoundError(f"Wan2.2_VAE.pth not found in {ckpt_dir}; pass --ckpt-dir with the full checkpoint.")
    if not list(ckpt_dir.glob("diffusion_pytorch_model*.safetensors")):
        raise FileNotFoundError(
            f"No Wan2.2 DiT safetensors in {ckpt_dir}. The external WanTI2V loads real weights from the ckpt dir."
        )
    config.model.provider = "wan2.2_external"                    # -> WanTI2VVideoModel (BaseVideoModel)
    config.model.pretrained_model_name_or_path = str(ckpt_dir)   # WanTI2V takes the DIR
    config.model.extra["offload_model"] = bool(args.offload_model)

    # Build AdaptedModel(WanTI2VVideoModel, AVID adapter) directly — no optimizer,
    # wandb, or checkpoint manager (unlike build_experiment); this is inference.
    base = build_base_model(config.model)
    condition_encoder = build_condition_encoder(config.conditioning)
    adapter = build_adapter(config.model, config.adapter, config.conditioning)
    model = AdaptedModel(
        base_model=base,
        adapter=adapter,
        condition_encoder=condition_encoder,
        pass_cond_to_base=config.model.pass_cond_to_base,
        condition_drop_prob=0.0,                                 # deterministic eval: never drop the condition
        output_composition=config.adapter.composition,
        gate_bias=config.adapter.gate_bias,
    ).to(device)
    model.eval()

    max_area = args.max_area if args.max_area is not None else _parse_size(args.size)
    gen_kwargs = dict(
        max_area=int(max_area),
        frame_num=args.frame_num,
        sampling_steps=args.steps,
        shift=args.shift,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    frame = _load_frame(args.image, args.hdf5, args.clip_idx)
    iio.imwrite(os.path.join(args.out_dir, "cond_frame.png"), frame)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"base={type(model.base_model).__name__} (external wan.WanTI2V)  adapter={config.adapter.extra.get('backbone')}")
    print(f"conditioning frame: {frame.shape}  ({'file ' + args.image if args.image else 'hdf5 clip ' + str(args.clip_idx)})")
    print(f"gen: max_area={max_area} frame_num={args.frame_num} steps={args.steps} "
          f"shift={args.shift} guide_scale={args.guide_scale} offload={args.offload_model}")
    print(f"adapter trainable params={trainable:,} (UNTRAINED — this is a plumbing check)")

    print("\n[1/2] base-only generation (delegates to upstream wan.WanTI2V.generate)")
    base_video = model.base_model.generate(frame, **gen_kwargs)
    _save_video(base_video, os.path.join(args.out_dir, "base_only.mp4"), args.fps)

    if args.base_only:
        print("\ndone (base only).")
        return

    # Constant action fed to the adapter across the whole rollout. Zeros by default
    # ("no motion"); the untrained adapter's delta is arbitrary regardless.
    action_dim = int(config.conditioning.input_dim)
    if args.action is not None:
        if len(args.action) != action_dim:
            raise ValueError(f"--action needs {action_dim} values (conditioning.input_dim), got {len(args.action)}.")
        action = torch.tensor(args.action, dtype=torch.float32, device=device)
    else:
        action = torch.zeros(action_dim, dtype=torch.float32, device=device)
    cond = {"action": action.unsqueeze(0)}  # [1, action_dim]

    print("\n[2/2] adapted generation via AdaptedModel (real AVID adapter injected at the denoise seam)")
    adapted_video = model.generate(frame, cond=cond, **gen_kwargs)
    _save_video(adapted_video, os.path.join(args.out_dir, "adapted.mp4"), args.fps)

    max_abs = (adapted_video.float() - base_video.float()).abs().max().item()
    print(f"\n|adapted - base| max = {max_abs:.4e}  "
          "(small if the adapter residual is ~0 at init; large delta = adapter is already perturbing the base)")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
