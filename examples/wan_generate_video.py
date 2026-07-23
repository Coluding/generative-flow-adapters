"""Generate a video through the integrated Wan2.1 backbone (GPU).

End-to-end validation of the integration: our ``Wan21DiTWrapper`` (loading the
real DiT weights) drives the vendored FlowUniPC sampler, with the vendored T5
encoder for text and the vendored Wan-VAE for the latent->pixel decode. Output
is an mp4.

With ``--through-adapter`` the sampling runs through the full ``AdaptedModel``
(frozen Wan base + zero-init output adapter, which composes to the base) so the
adapter path is exercised too.

Example:
    python examples/wan_generate_video.py \
        --ckpt-dir ckpts/Wan2.1-T2V-1.3B \
        --prompt "a cat playing piano, cinematic" \
        --frames 33 --height 256 --width 256 --steps 30 \
        --out outputs/wan_cat.mp4
"""

from __future__ import annotations

import argparse
import gc
import os
import imageio.v3 as iio
import torch

from generative_flow_adapters.backbones.wan.modules.t5 import T5EncoderModel
from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
from generative_flow_adapters.backbones.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from generative_flow_adapters.backbones.wan.utils.utils import cache_video
from generative_flow_adapters.models.base.wan import Wan21DiTWrapper

# Wan's default negative prompt (config.sample_neg_prompt), used for CFG.
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


def build_base(ckpt_dir: str, device: str, dtype: torch.dtype) -> Wan21DiTWrapper:
    base = Wan21DiTWrapper.from_config(
        wan_config_path="configs/base/wan2.1_t2v_1_3B.yaml",
        model_type="flow",
        prediction_type="velocity",
        checkpoint_path=ckpt_dir,  # directory of DiT *.safetensors shards
        dtype=dtype,
    )
    return base.to(device).eval()


def build_adapted_model(ckpt_dir: str, device: str, dtype: torch.dtype):
    """Wrap the frozen base in the AdaptedModel from the generation config."""
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    config = load_config("configs/wan21/diffusion_wan_shortcut_metaworld.yaml")
    config.model.pretrained_model_name_or_path = ckpt_dir
    config.model.extra["wan_config_path"] = "configs/base/wan2.1_t2v_1_3B.yaml"
    config.model.extra["allow_missing_checkpoint"] = False
    config.model.extra["dtype"] = "bf16" if dtype == torch.bfloat16 else "float32"
    model = build_experiment(config).model
    return model.to(device).eval()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.1-T2V-1.3B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--neg-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--frames", type=int, default=33, help="pixel frames (4n+1)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=3.0, help="flow schedule shift (3 for <=480p)")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--out", default="outputs/wan_sample.mp4")
    parser.add_argument("--through-adapter", action="store_true", help="sample through AdaptedModel (base+adapter)")
    # Image conditioning (SDEdit / video-to-video): condition the generation on a
    # real MetaWorld frame. The 1.3B model is T2V (no native image input), so we
    # encode the frame to a latent, tile it over time, add partial noise at
    # `--strength`, and denoise the tail of the schedule from there.
    parser.add_argument("--cond-image-hdf5", default=None, help="MetaWorld HDF5 to draw a conditioning frame from")
    parser.add_argument("--cond-clip-idx", type=int, default=0, help="clip index in the dataset")
    parser.add_argument("--strength", type=float, default=0.7,
                        help="SDEdit noise strength in (0,1]; 1.0 = ignore frame (pure T2V), lower = stay closer to it")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) Text FIRST: build T5 on CPU (its fp32 construction would peak ~22GB on
    # GPU), encode there, move the contexts to GPU, then free T5 entirely — so
    # the 11GB encoder never shares the GPU with the DiT.
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device="cpu",
        checkpoint_path=os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"),
    )
    context = [c.to(device=device, dtype=dtype) for c in text_encoder([args.prompt], "cpu")]
    context_null = [c.to(device=device, dtype=dtype) for c in text_encoder([args.neg_prompt], "cpu")]
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("text encoded; T5 freed")

    # 2) Model: our Wan wrapper (optionally wrapped in AdaptedModel) with real weights.
    if args.through_adapter:
        model = build_adapted_model(args.ckpt_dir, device, dtype)
        forward = lambda x, t, ctx: model(x, t, {"context": ctx, "action": torch.zeros(x.shape[0], 4, device=device)})
    else:
        model = build_base(args.ckpt_dir, device, dtype)
        forward = lambda x, t, ctx: model(x, t, {"context": ctx})

    # 3) Wan-VAE (small) — needed for the conditioning encode and the final decode.
    vae = WanVAE(vae_pth=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"), device=device)

    # 4) Latent geometry + initial latent.
    lat_f = (args.frames - 1) // VAE_STRIDE[0] + 1
    lat_h = args.height // VAE_STRIDE[1]
    lat_w = args.width // VAE_STRIDE[2]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(1, 16, lat_f, lat_h, lat_w, generator=gen, device=device, dtype=torch.float32)
    print(f"latent shape {tuple(noise.shape)}  (pixels {args.frames}x{args.height}x{args.width})")

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=device, shift=args.shift)

    if args.cond_image_hdf5:
        # SDEdit init: encode the MetaWorld frame, tile it across time, and start
        # the denoise partway down the schedule at noise level `strength`.
        z0 = encode_cond_frame(vae, args.cond_image_hdf5, args.cond_clip_idx,
                               args.frames, args.height, args.width, device, args.out)
        start_idx = min(int(round((1.0 - args.strength) * args.steps)), args.steps - 1)
        sigma_start = scheduler.sigmas[start_idx].to(device)
        latent = ((1.0 - sigma_start) * z0 + sigma_start * noise).float()
        timesteps = scheduler.timesteps[start_idx:]
        print(f"image-conditioned (SDEdit): strength={args.strength} start_idx={start_idx}/{args.steps}")
    else:
        latent = noise
        timesteps = scheduler.timesteps

    # 5) Denoise: CFG between prompt and negative-prompt velocities.
    for i, t in enumerate(timesteps):
        t_b = t.unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=dtype):
            v_cond = forward(latent, t_b, context)
            v_uncond = forward(latent, t_b, context_null)
        v = (v_uncond + args.guide_scale * (v_cond - v_uncond)).float()
        latent = scheduler.step(v, t, latent, return_dict=False, generator=gen)[0]
        if (i + 1) % max(1, len(timesteps) // 5) == 0:
            print(f"  step {i + 1}/{args.steps}")

    # 6) Decode latent -> pixels and save.
    video = vae.decode([latent[0]])[0]  # [3, F, H, W] in [-1, 1]
    saved = cache_video(video[None], save_file=args.out, fps=args.fps, normalize=True, value_range=(-1, 1))
    print(f"saved: {saved}")


def encode_cond_frame(vae, hdf5_path, clip_idx, frames, height, width, device, out_path):
    """Load a MetaWorld frame, tile it across `frames`, encode to a Wan latent.

    Returns z0 of shape ``[1, 16, lat_f, lat_h, lat_w]`` (the normalised latent
    the DiT operates in). Also saves the conditioning frame next to the output.
    """
    import torch.nn.functional as F
    from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset

    translator = MetaWorldTranslator(hdf5_path, caption_mode="empty")
    dataset = TranslatedClipDataset(translator, window_width=8, frame_stride=4, sampling="exhaustive")
    clip = dataset[clip_idx]
    frame = torch.as_tensor(clip["video"])[0]  # [H, W, 3] uint8 — first frame

    # Save the conditioning frame alongside the output for comparison.
    iio.imwrite(os.path.splitext(out_path)[0] + "_cond_frame.png", frame.numpy().astype("uint8"))

    # [H,W,3] uint8 -> [3,H,W] float in [-1,1], resize to target, tile over time.
    img = frame.permute(2, 0, 1).float().div(127.5).sub(1.0)
    img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    video = img.unsqueeze(1).repeat(1, frames, 1, 1)  # [3, F, H, W]
    z0 = vae.encode([video.to(device)])[0]  # [16, lat_f, lat_h, lat_w]
    return z0.unsqueeze(0).float()


if __name__ == "__main__":
    main()
