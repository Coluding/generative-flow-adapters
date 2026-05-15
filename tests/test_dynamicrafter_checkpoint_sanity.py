"""Sanity tests for the real DynamiCrafter checkpoint.

These tests:
  1. Verify the VAE decoder round-trips a random frame sampled from the
     ``ds/metaworld_corner2.hdf5`` dataset (catches a broken ``first_stage_model.*``
     load — a random VAE would not reconstruct a real scene).
  2. Run a short DDIM rollout on the BASE model only (no adapter), decode the
     latents, and assert the output is finite and non-degenerate (catches a
     broken UNet load).
  3. Optionally save the decoded frames as PNGs under
     ``tests/_outputs/dynamicrafter_sanity/`` when ``DYNAMICRAFTER_SAVE_FRAMES=1``.

Auto-skips when the checkpoint is missing (set ``DYNAMICRAFTER_CHECKPOINT`` or
place it at ``ckts/dynami512.ckpt``) or when CUDA is not available (the
1.5B-param UNet won't fit comfortably on CPU).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from generative_flow_adapters.inference import DiffusionInferenceSampler
from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective
from generative_flow_adapters.models.base.dynamicrafter import DynamicCrafterUNetWrapper

from generative_flow_adapters.data.clip import (
    build_dynamicrafter_resampler_from_checkpoint,
    encode_image_with_openclip,
    encode_with_openclip,
)


UNET_CONFIG_PATH = "external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml"
DEFAULT_CHECKPOINT_PATH = "ckts/dynami512.ckpt"
METAWORLD_DATASET_PATH = "ds/metaworld_corner2.hdf5"


def _resolve_checkpoint_path() -> Path | None:
    candidates = [os.environ.get("DYNAMICRAFTER_CHECKPOINT"), DEFAULT_CHECKPOINT_PATH]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


CHECKPOINT_PATH = _resolve_checkpoint_path()
HAS_CUDA = torch.cuda.is_available()
SKIP_REASON = (
    "DynamiCrafter checkpoint not found (set DYNAMICRAFTER_CHECKPOINT or place at ckts/dynami512.ckpt) "
    "or CUDA unavailable (the 1.5B-param UNet won't fit comfortably on CPU)."
)


@unittest.skipUnless(CHECKPOINT_PATH is not None and HAS_CUDA, SKIP_REASON)
class DynamicCrafterCheckpointSanityTest(unittest.TestCase):
    """End-to-end checks against the real DynamiCrafter weights."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        cls.dtype = torch.float32
        cls.wrapper = DynamicCrafterUNetWrapper.from_config(
            model_type="diffusion",
            unet_config_path=UNET_CONFIG_PATH,
            checkpoint_path=str(CHECKPOINT_PATH),
            prediction_type="velocity",
            allow_dummy_concat_condition=True,
            load_first_stage_model=True,
        ).to(cls.device).eval()
        for parameter in cls.wrapper.parameters():
            parameter.requires_grad_(False)

    def test_vae_round_trip_reconstructs_metaworld_frame(self):
        """Encode-decode a random frame from the metaworld dataset; expect low MSE."""
        clip = _load_random_metaworld_clip(num_frames=2, device=self.device, dtype=self.dtype)
        if clip is None:
            self.skipTest(
                f"Metaworld dataset not found at {METAWORLD_DATASET_PATH}; cannot run VAE round-trip on real frames."
            )
        vae = self.wrapper.first_stage_model

        latents = vae.encode_video(clip)
        decoded = vae.decode_video(latents)

        self.assertEqual(decoded.shape, clip.shape)
        self.assertTrue(torch.isfinite(latents).all(), "VAE produced non-finite latents")
        self.assertTrue(torch.isfinite(decoded).all(), "VAE produced non-finite decoded pixels")

        mse = (decoded.float() - clip.float()).pow(2).mean().item()
        # SD-VAE round-trip on real images typically scores well under 0.05.
        # Random-init weights would score much higher (~1+).
        self.assertLess(mse, 0.05, f"VAE round-trip MSE too high: {mse:.4f} — checkpoint may be miswired.")

        if os.environ.get("DYNAMICRAFTER_SAVE_FRAMES") == "1":
            self._save_frames(clip, name="vae_input")
            self._save_frames(decoded, name="vae_reconstruction")

    def test_base_model_image_to_video_rollout_is_semantically_consistent(self):
        """Run a DDIM rollout on the unadapted base model with a real metaworld
        frame as `concat` conditioning. The first decoded frame should resemble
        the conditioning image — that's the DynamiCrafter image-to-video
        contract. A miswired UNet would still produce finite output but would
        not honor the conditioning image."""
        first_frame = _load_random_metaworld_clip(num_frames=1, device=self.device, dtype=self.dtype)
        if first_frame is None:
            self.skipTest(
                f"Metaworld dataset not found at {METAWORLD_DATASET_PATH}; "
                "the rollout test needs a real image to use as `concat` conditioning."
            )

        # Upscale to the model's training resolution. dynamicrafter_512.yaml has
        # `image_size: [40, 64]` (latent) == 320x512 pixels. Metaworld frames are
        # 128x128 — running the U-Net at that resolution drives its lowest
        # attention block down to 1x1 tokens, well outside its trained regime.
        # F.interpolate works on 4D [N, C, H, W]; collapse the T=1 axis, resize,
        # then put it back.
        first_frame = first_frame.squeeze(2)  # [B, 3, H, W]
        first_frame = torch.nn.functional.interpolate(
            first_frame,
            size=(320, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1.0, 1.0)
        first_frame = first_frame.unsqueeze(2)  # [B, 3, 1, 320, 512]

        # VAE-encode the conditioning frame, broadcast across time as `concat`.
        vae = self.wrapper.first_stage_model
        with torch.no_grad():
            first_frame_latent = vae.encode_video(first_frame)  # [1, 4, 1, 40, 64]
        frames = 16
        concat_latent = first_frame_latent.expand(-1, -1, frames, -1, -1).contiguous()
        batch_size, channels, _, h, w = concat_latent.shape

        texts = [""]
        text_emb = encode_with_openclip(texts, device=self.device).to(dtype=self.dtype)        # [1, 77, 1024]
        null_text_emb = encode_with_openclip([""], device=self.device).to(dtype=self.dtype)    # [1, 77, 1024]
        first_frame_pixels = first_frame[:, :, 0]
        with torch.no_grad():
            image_tokens = encode_image_with_openclip(
                first_frame_pixels, device=self.device
            ).to(dtype=self.dtype)
            resampler = build_dynamicrafter_resampler_from_checkpoint(
                str(CHECKPOINT_PATH), device=self.device
            ).to(dtype=self.dtype)
            image_emb = resampler(image_tokens)  # [1, 16*16, 1024]

        # Conditional context = real text + image embedding.
        context = torch.cat([text_emb, image_emb], dim=1)            # [1, 77 + 256, 1024]
        uncond_context = torch.cat([null_text_emb, image_emb], dim=1)

        cond = {"context": context, "concat": concat_latent}
        unconditional_cond = {"context": uncond_context, "concat": concat_latent}

        schedule = self.wrapper.diffusion_schedule_config or {}
        objective = DiffusionTrainingObjective(
            timesteps=int(schedule.get("timesteps", 1000)),
            beta_schedule=str(schedule.get("beta_schedule", "linear")),
            linear_start=float(schedule.get("linear_start", 8.5e-4)),
            linear_end=float(schedule.get("linear_end", 1.2e-2)),
            rescale_betas_zero_snr=bool(schedule.get("rescale_betas_zero_snr", False)),
            use_dynamic_rescale=bool(schedule.get("use_dynamic_rescale", False)),
            base_scale=float(schedule.get("base_scale", 0.7)),
            turning_step=int(schedule.get("turning_step", 400)),
        )
        sampler = DiffusionInferenceSampler(
            model=self.wrapper,
            objective=objective,
            prediction_type="velocity",
            scheduler_name="ddim",
        )

        with torch.no_grad():
            latents = sampler.sample(
                shape=(batch_size, channels, frames, h, w),
                cond=cond,
                device=self.device,
                dtype=self.dtype,
                num_inference_steps=25,
                unconditional_cond=unconditional_cond,
                guidance_scale=7.5,
            )
            decoded = self.wrapper.decode_first_stage(latents)

        self.assertTrue(torch.isfinite(latents).all(), "Sampled latents contain NaN/inf.")
        self.assertTrue(torch.isfinite(decoded).all(), "Decoded pixels contain NaN/inf.")
        self.assertEqual(decoded.shape, (batch_size, 3, frames, h * 8, w * 8))

        # The DynamiCrafter image-to-video contract: the first generated frame
        # should resemble the conditioning image. Tolerate moderate drift but
        # require *substantially* better than chance against an unrelated frame.
        first_decoded = decoded[:, :, 0]
        first_cond = first_frame[:, :, 0]
        first_frame_mse = (first_decoded.float() - first_cond.float()).pow(2).mean().item()

        if os.environ.get("DYNAMICRAFTER_SAVE_FRAMES") == "1":
            self._save_frames(first_frame.expand(-1, -1, frames, -1, -1).contiguous(), name="conditioning_frame")
            self._save_frames(decoded, name="generated_rollout")

        self.assertLess(
            first_frame_mse,
            0.25,
            f"First generated frame is too far from the conditioning image (MSE={first_frame_mse:.3f}); "
            "DynamiCrafter image conditioning may not be wired correctly.",
        )

    @staticmethod
    def _save_frames(decoded: torch.Tensor, *, name: str) -> None:
        from PIL import Image

        out_dir = Path("tests/_outputs/dynamicrafter_sanity")
        out_dir.mkdir(parents=True, exist_ok=True)
        # decoded: [B, 3, T, H, W] in roughly [-1, 1]. Save sample 0.
        clip = decoded[0].clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8).cpu()
        for frame_index in range(clip.shape[1]):
            frame = clip[:, frame_index].permute(1, 2, 0).numpy()  # HWC
            Image.fromarray(frame).save(out_dir / f"{name}_frame{frame_index:02d}.png")


def _load_random_metaworld_clip(*, num_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    """Sample a random consecutive `num_frames`-clip of pixels from the dataset.

    Returns a `[1, 3, T, H, W]` tensor in `[-1, 1]`, or `None` if the dataset
    file isn't present.
    """
    if not Path(METAWORLD_DATASET_PATH).exists():
        return None
    try:
        import h5py
    except ImportError:
        return None
    import random

    with h5py.File(METAWORLD_DATASET_PATH, "r") as handle:
        # The HDF5 layout is: <task>/<episode>/pixels with shape [T_total, H, W, 3].
        task_names = list(handle.keys())
        if not task_names:
            return None
        task = handle[random.choice(task_names)]
        episode_names = [name for name in task.keys() if "pixels" in task[name]]
        if not episode_names:
            return None
        pixels = task[random.choice(episode_names)]["pixels"]  # [T_total, H, W, 3] uint8
        if pixels.shape[0] < num_frames:
            return None
        start = random.randint(0, pixels.shape[0] - num_frames)
        clip_uint8 = pixels[start : start + num_frames]  # [T, H, W, 3]
    # uint8 [0, 255] HWC → float [-1, 1] CHW, then add batch + permute to [B, C, T, H, W].
    clip_float = torch.from_numpy(clip_uint8).to(device=device, dtype=dtype) / 127.5 - 1.0
    return clip_float.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


if __name__ == "__main__":
    unittest.main()
