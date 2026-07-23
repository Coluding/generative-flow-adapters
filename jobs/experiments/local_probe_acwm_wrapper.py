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
import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument("--via-factory", action="store_true",
                    help="Build the base through build_experiment() + .to(device) + .eval() "
                         "exactly as the compare script does, instead of constructing "
                         "WanTI2VVideoModel directly. Isolates model-construction/state "
                         "differences from generate()-argument differences.")
parser.add_argument("--compare-decode-shim", action="store_true",
                    help="Use the compare script's exact phase-2 CPU round-trip decode shim "
                         "(flip VAE to CPU fp32, decode, restore GPU) instead of this probe's "
                         "simpler decode-only shim. Tests whether that shim's state churn is "
                         "what corrupts the rollout.")
parser.add_argument("--seam-hook", action="store_true",
                    help="Monkeypatch model._compose_with_adapter with the compare script's "
                         "seam instrumentation before generating. Tests whether that wrap "
                         "corrupts the base rollout.")
parser.add_argument("--preprocess", action="store_true",
                    help="Run the Wan2.2 preprocessor on the clip BEFORE generating, exactly "
                         "as the compare script does. Tests whether that step's state (RNG, "
                         "VAE, CUDA) corrupts the subsequent base rollout.")
parser.add_argument("--random-init", action="store_true",
                    help="Perturb adapter params (needs --via-factory) exactly as the compare "
                         "default does — advances global RNG before the rollout.")
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

if probe_args.via_factory:
    # Compare-script construction path: config -> provider factory -> AdaptedModel
    # -> .to("cuda") -> .eval(); the base is model.base_model.
    from generative_flow_adapters.config import load_config  # noqa: E402
    from generative_flow_adapters.training import build_experiment  # noqa: E402

    cfg = load_config(str(REPO / "configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"))
    cfg.model.provider = "wan2.2_external"
    cfg.model.pretrained_model_name_or_path = str(REPO / "ckpts/Wan2.2-TI2V-5B")
    cfg.model.extra["offload_model"] = False
    wandb_cfg = cfg.training.extra.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["enable"] = False
    adapted = build_experiment(cfg).model.to("cuda")
    adapted.eval()

    if probe_args.random_init:
        with torch.no_grad():
            for p in adapted.adapter.parameters():
                p.add_(0.02 * torch.randn_like(p))
        print("random-init: adapter perturbed (advanced global RNG)", flush=True)

    if probe_args.preprocess:
        # Run the exact compare-script preprocessing before the rollout.
        from generative_flow_adapters.data import (  # noqa: E402
            Wan22DiffusionForcingPreprocessor, WanBatchPreprocessConfig,
            build_acwmphys_clip_dataset,
        )
        from generative_flow_adapters.training.trainer import _call_preprocessor  # noqa: E402
        from torch.utils.data import DataLoader, Subset  # noqa: E402

        vae_pp = adapted.base_model.wan.vae
        pctx = str(REPO / "configs/prompts/acwm_pushblock.contexts.pt")
        pp = Wan22DiffusionForcingPreprocessor(
            vae=vae_pp,
            config=WanBatchPreprocessConfig(
                target_height=256, target_width=256, timestep_scale=1000.0,
                max_area=901120, align_h=32, align_w=32,
                prompt_contexts_path=pctx,
                latent_cache_dir=str(REPO / "ds/acwm-phys/rigid_dynamics/push_block/native17.latents"),
            ),
            condition_keys=("act",), cond_frames=1,
        )
        _, ds_pp = build_acwmphys_clip_dataset(
            cfg.data, default_window_width=17,
            data_dir=str(REPO / "ds/acwm-phys/rigid_dynamics/push_block/ind_train"),
            num_windows=1, sampling="random",
        )
        raw = next(iter(DataLoader(Subset(ds_pp, [0]), batch_size=1)))
        _call_preprocessor(pp, raw, train=False)
        print("preprocess: ran compare-script preprocessing before rollout", flush=True)

    model = adapted.base_model
    print("model: built via factory (AdaptedModel.base_model)", flush=True)
else:
    model = WanTI2VVideoModel(str(REPO / "ckpts/Wan2.2-TI2V-5B"), offload_model=True)
    print("model: constructed directly", flush=True)

# CPU-decode shim (native decode OOMs 24 GB; encode of ONE frame fits on GPU).
vae = model.wan.vae
_decode = vae.decode

if probe_args.compare_decode_shim:
    # EXACT copy of the compare script's phase-2 round-trip shim: flip VAE to
    # CPU fp32, decode, then restore to the native device/dtype in a finally.
    _native_dev = vae.device
    _native_attr_dtype = vae.dtype
    _native_wt_dtype = next(vae.model.parameters()).dtype

    def cpu_decode(zs):
        print("decode: CPU fp32 (compare round-trip shim; restored after)", flush=True)
        vae.model = vae.model.to("cpu").float()
        vae.scale = [s.cpu() if torch.is_tensor(s) else s for s in vae.scale]
        vae.dtype = torch.float32
        try:
            return _decode([z.detach().float().cpu() for z in zs])
        finally:
            vae.model = vae.model.to(_native_dev).to(_native_wt_dtype)
            vae.scale = [s.to(_native_dev) if torch.is_tensor(s) else s for s in vae.scale]
            vae.dtype = _native_attr_dtype
else:
    def cpu_decode(zs):
        vae.model = vae.model.to("cpu").float()
        vae.scale = [s.cpu() if torch.is_tensor(s) else s for s in vae.scale]
        vae.dtype = torch.float32
        print("decode: CPU fp32 (simple shim)", flush=True)
        return _decode([z.detach().float().cpu() for z in zs])

vae.decode = cpu_decode

if probe_args.seam_hook and hasattr(model, "_compose_with_adapter"):
    import sys as _s  # noqa: E402
    _s.path.insert(0, str(REPO / "scripts"))
    from generate_wan22_i2v_compare import _seam_hook as _apply_seam_hook  # noqa: E402
    _apply_seam_hook(model, [], {"name": "gen"}, OUT)
    print("seam hook: applied", flush=True)
elif probe_args.seam_hook:
    print("seam hook: base model has no _compose_with_adapter — not applicable to base rollout", flush=True)

# Dump the anchor latent `z` from inside upstream i2v (vae.encode([img])) so we
# can diff it against the compare's — the single tensor that decides the rollout.
_probe_encode = vae.encode


def _encode_dump(vs):
    out = _probe_encode(vs)
    z = out[0].float()
    print(f"ANCHOR z: {tuple(z.shape)} mean={z.mean():.4f} std={z.std():.4f} "
          f"absmax={z.abs().max():.3f}", flush=True)
    return out


vae.encode = _encode_dump

kw = dict(
    max_area=704 * 1280,
    frame_num=17,
    sampling_steps=50,
    shift=5.0,
    guide_scale=5.0,
    seed=0,
    offload_model=True,
    context=context.to("cuda"),
    context_null=context_null.to("cuda"),
)

# Same fingerprint block the compare script prints — diff the two outputs.
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(REPO / "scripts"))
from generate_wan22_i2v_compare import _fingerprint_generate_inputs  # noqa: E402

_fingerprint_generate_inputs(canvas, kw, model)

video = model.generate(canvas, compose_fn=None, **kw)
print("video:", tuple(video.shape), flush=True)

from wan.utils.utils import save_video  # noqa: E402

tag = "acwm_wrapper_probe_translator" if probe_args.translator_frame else "acwm_wrapper_probe"
if probe_args.pool_index != 0:
    tag += f"_pool{probe_args.pool_index}"
if probe_args.via_factory:
    tag += "_factory"
save_video(tensor=video[None], save_file=str(OUT / f"{tag}.mp4"), fps=8,
           nrow=1, normalize=True, value_range=(-1, 1))
frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()
strip = np.concatenate([frames[:, i].permute(1, 2, 0).cpu().numpy() for i in (0, 5, 11, 16)], axis=0)
Image.fromarray(strip).save(OUT / f"{tag}_strip.png")
print("saved", OUT / f"{tag}.mp4", "and _strip.png", flush=True)
