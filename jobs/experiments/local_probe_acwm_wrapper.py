"""BISECTION PROBE (2026-07-23): our WanTI2VVideoModel.generate wrapper vs the
upstream pipeline, on the EXACT letterboxed frame + settings that produced
coherent video through upstream generate.py.

Isolates {_ComposedDiT pass-through + _CachedT5 context stub + our wrapper}
from the compare script's machinery (preprocessor, CPU-encode shim, dataset
path). No adapter, no preprocessor — pure base rollout.

  - Coherent video  -> wrapper fine; the bug is in the compare script's
    surroundings (frame handed to generate / VAE shim state / batch path).
  - Noise           -> the wrapper (_ComposedDiT / _CachedT5 / arg plumbing)
    breaks the base at native geometry, despite working on MetaWorld 768².

Run:  bash jobs/experiments/local_probe_acwm_wrapper.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from generative_flow_adapters.models.base.wan_ti2v import WanTI2VVideoModel

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "outputs" / "replace_debug"
OUT.mkdir(parents=True, exist_ok=True)

import argparse  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--translator-frame", action="store_true",
                    help="Use the frame EXACTLY as the compare script gets it: the translator's "
                         "full-res 1024x1862 letterboxed canvas, upstream does the resize. "
                         "Default: pre-resized 1280x704 canvas (the variant that worked).")
parser.add_argument("--pool-index", type=int, default=0,
                    help="Which prompt embedding to use: 0..N-1 = pool entries, -1 = the "
                         "__default__ entry. Only pool[0] was validated against the base; "
                         "eval used to draw randomly (bug, fixed 2026-07-23) — scan the "
                         "others to find toxic prompts.")
probe_args = parser.parse_args()

if probe_args.translator_frame:
    # Compare-script path: translator letterbox at full res, no pre-resize.
    from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator  # noqa: E402

    tr = ACWMPhysTranslator(str(REPO / "ds/acwm-phys/rigid_dynamics/push_block/ind_train"))
    clip = tr.load_clip(tr.list_episodes()[0], start=0, length=1, stride=1)
    canvas = Image.fromarray(clip["video"][0]).convert("RGB")  # 1862x1024 uint8
    print("conditioning frame (translator path):", canvas.size, flush=True)
else:
    # --- the exact frame that worked upstream: episode_0 frame 0, letterboxed ---
    from decord import VideoReader  # noqa: E402

    vr = VideoReader(str(REPO / "ds/acwm-phys/rigid_dynamics/push_block/ind_train/episode_0.mp4"))
    f = vr[0].asnumpy()  # 1024x1024x3 uint8
    h_target, w_target = 704, 1280
    new_w = int(round(f.shape[1] * h_target / f.shape[0]))
    img = Image.fromarray(f).resize((new_w, h_target), Image.LANCZOS)
    canvas = Image.new("RGB", (w_target, h_target), (255, 255, 255))
    canvas.paste(img, ((w_target - new_w) // 2, 0))
    print("conditioning frame (pre-resized):", canvas.size, flush=True)

# --- context from the SAME precomputed table the failing run used -----------
table = torch.load(REPO / "configs/prompts/acwm_pushblock.contexts.pt", map_location="cpu", weights_only=False)
if probe_args.pool_index < 0:
    context = table["positive"]["__default__"]
    print("context: __default__ entry", flush=True)
else:
    context = table["pool"][probe_args.pool_index]
    print(f"context: pool[{probe_args.pool_index}] of {len(table['pool'])}", flush=True)
context_null = table["negative"]
print("context:", tuple(context.shape), "negative:", tuple(context_null.shape), flush=True)

model = WanTI2VVideoModel(str(REPO / "ckpts/Wan2.2-TI2V-5B"), offload_model=True)

# CPU-decode shim (native decode OOMs 24 GB; encode of ONE frame fits on GPU).
vae = model.wan.vae
_decode = vae.decode


def cpu_decode(zs):
    vae.model = vae.model.to("cpu").float()
    vae.scale = [s.cpu() if torch.is_tensor(s) else s for s in vae.scale]
    vae.dtype = torch.float32
    print("decode: CPU fp32", flush=True)
    return _decode([z.detach().float().cpu() for z in zs])


vae.decode = cpu_decode

video = model.generate(
    canvas,
    compose_fn=None,                # pure base
    context=context,
    context_null=context_null,
    max_area=704 * 1280,
    frame_num=17,
    sampling_steps=50,
    shift=5.0,
    guide_scale=5.0,
    seed=0,
)
print("video:", tuple(video.shape), flush=True)

from wan.utils.utils import save_video  # noqa: E402

tag = "acwm_wrapper_probe_translator" if probe_args.translator_frame else "acwm_wrapper_probe"
if probe_args.pool_index != 0:
    tag += f"_pool{probe_args.pool_index}"
save_video(tensor=video[None], save_file=str(OUT / f"{tag}.mp4"), fps=8,
           nrow=1, normalize=True, value_range=(-1, 1))
frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()
strip = np.concatenate([frames[:, i].permute(1, 2, 0).cpu().numpy() for i in (0, 5, 11, 16)], axis=0)
Image.fromarray(strip).save(OUT / f"{tag}_strip.png")
print("saved", OUT / f"{tag}.mp4", "and _strip.png", flush=True)
