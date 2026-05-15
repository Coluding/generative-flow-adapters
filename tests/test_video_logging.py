"""Tests for the DynamiCrafter VAE wrapper and the WandbLogger plumbing.

We avoid actually calling wandb; we only check that:

1. The wrapper can build a `VideoAutoencoderKL` from the UNet YAML and decode.
2. `WandbLogger.log_videos` produces the right number of side-by-side panels
   and `log_metrics` filters non-scalar entries.
3. The builder constructs the logger when configured and the base has a VAE,
   raises a clear error when the base has no VAE (default), and skips logger
   construction when disabled.
4. (Heavy, opt-in via real checkpoint + dataset.) The end-to-end logging panel
   produced by the trainer's eval cadence -- ground-truth / base-only /
   adapter-residual decoded side by side -- is dumped to disk for visual
   inspection, mirroring `WandbLogger._decode_to_uint8` so any bug in the
   uint8 conversion or VAE decode path surfaces identically in the saved files.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import torch

from generative_flow_adapters.config import (
    AdapterConfig,
    ConditioningConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from generative_flow_adapters.models.base.dynamicrafter import DynamicCrafterUNetWrapper


UNET_CONFIG_PATH = "external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml"


class _FakeWandb:
    run = object()

    class Video:
        def __init__(self, data, fps, format, caption):
            self.data = data
            self.fps = fps
            self.format = format
            self.caption = caption


def _make_logger(fake_log: list, *, decode_fn=None, num_samples: int = 1):
    from generative_flow_adapters.training.wandb_logger import WandbLogger

    class FakeWandb(_FakeWandb):
        @staticmethod
        def log(payload, step):
            fake_log.append((payload, step))

    if decode_fn is None:
        # Identity-ish decoder: drop the alpha channel to mimic 4-ch latent → 3-ch RGB.
        def decode_fn(latents):  # type: ignore[no-redef]
            return latents[:, :3]

    with mock.patch.dict("sys.modules", {"wandb": FakeWandb}):
        return WandbLogger(decode_fn=decode_fn, num_samples=num_samples, fps=4)


class DynamicCrafterVAETest(unittest.TestCase):
    def test_load_first_stage_model_attaches_vae(self):
        wrapper = DynamicCrafterUNetWrapper.from_config(
            model_type="diffusion",
            unet_config_path=UNET_CONFIG_PATH,
            checkpoint_path=None,
            load_first_stage_model=True,
        )
        self.assertIsNotNone(wrapper.first_stage_model)
        self.assertEqual(wrapper.first_stage_model.scale_factor, 0.18215)

    def test_decode_first_stage_returns_video_shape(self):
        wrapper = DynamicCrafterUNetWrapper.from_config(
            model_type="diffusion",
            unet_config_path=UNET_CONFIG_PATH,
            checkpoint_path=None,
            load_first_stage_model=True,
        )
        latents = torch.randn(1, 4, 2, 8, 8)
        decoded = wrapper.decode_first_stage(latents)
        self.assertEqual(tuple(decoded.shape), (1, 3, 2, 64, 64))

    def test_decode_first_stage_without_vae_raises(self):
        wrapper = DynamicCrafterUNetWrapper.from_config(
            model_type="diffusion",
            unet_config_path=UNET_CONFIG_PATH,
            checkpoint_path=None,
            load_first_stage_model=False,
        )
        self.assertIsNone(wrapper.first_stage_model)
        with self.assertRaisesRegex(RuntimeError, "first_stage_model"):
            wrapper.decode_first_stage(torch.randn(1, 4, 2, 8, 8))


class WandbLoggerVideoPanelTest(unittest.TestCase):
    def test_three_panel_video_when_base_provided(self):
        fake_log: list = []
        logger = _make_logger(fake_log)
        target = torch.zeros(1, 4, 2, 4, 8)
        adapted = torch.zeros(1, 4, 2, 4, 8)
        base = torch.zeros(1, 4, 2, 4, 8)

        logger.log_videos(
            prediction_latents=adapted,
            target_latents=target,
            base_prediction_latents=base,
            cond={"act": torch.randn(1, 2, 4)},
            step=42,
        )

        payload, step = fake_log[0]
        self.assertEqual(step, 42)
        video = payload["eval/sample_0"]
        # Three panels concatenated along width: 8 + 8 + 8.
        self.assertEqual(video.data.shape[-1], 24)
        self.assertIn("base_model", video.caption)

    def test_two_panel_video_when_base_absent(self):
        fake_log: list = []
        logger = _make_logger(fake_log)
        target = torch.zeros(1, 4, 2, 4, 8)
        adapted = torch.zeros(1, 4, 2, 4, 8)

        logger.log_videos(prediction_latents=adapted, target_latents=target, cond=None, step=7)

        video = fake_log[0][0]["eval/sample_0"]
        self.assertEqual(video.data.shape[-1], 16)
        self.assertNotIn("base_model", video.caption)


class WandbLoggerMetricsTest(unittest.TestCase):
    def test_log_metrics_filters_non_scalar_and_prefixes_keys(self):
        fake_log: list = []
        logger = _make_logger(fake_log)

        metrics = {
            "loss": 0.732,
            "shortcut_direction_loss": torch.tensor(0.123),
            "generated_samples": torch.zeros(2, 4, 8, 16, 16),  # tensor → must be filtered
            "is_eval": True,                                     # bool → coerced to float
        }
        logger.log_metrics(metrics, step=99)

        payload, step = fake_log[0]
        self.assertEqual(step, 99)
        self.assertEqual(set(payload.keys()), {"train/loss", "train/shortcut_direction_loss", "train/is_eval"})
        self.assertAlmostEqual(payload["train/loss"], 0.732)
        self.assertAlmostEqual(payload["train/shortcut_direction_loss"], 0.123, places=4)
        self.assertEqual(payload["train/is_eval"], 1.0)

    def test_log_metrics_skips_when_no_scalar_entries(self):
        fake_log: list = []
        logger = _make_logger(fake_log)

        logger.log_metrics({"only_tensor": torch.zeros(3, 3)}, step=1)

        # No wandb.log call should happen when nothing scalar is present.
        self.assertEqual(len(fake_log), 0)


class WandbBuilderTest(unittest.TestCase):
    def _base_config(self, *, enable_logging: bool, load_vae: bool, key: str = "wandb") -> ExperimentConfig:
        return ExperimentConfig(
            model=ModelConfig(
                type="diffusion",
                provider="dynamicrafter",
                prediction_type="velocity",
                extra={
                    "unet_config_path": UNET_CONFIG_PATH,
                    "latent_channels": 4,
                    "allow_dummy_concat_condition": True,
                    "load_first_stage_model": load_vae,
                },
            ),
            adapter=AdapterConfig(type="hyper", hidden_dim=128, rank=4, extra={"architecture": "hyperalign"}),
            conditioning=ConditioningConfig(type="structured", output_dim=512),
            training=TrainingConfig(
                loss="diffusion",
                extra={key: {"enable": enable_logging, "num_samples": 2, "fps": 4}},
            ),
            name="test_wandb",
        )

    def test_logger_is_none_when_disabled(self):
        from generative_flow_adapters.training.builders import _maybe_build_wandb_logger

        config = self._base_config(enable_logging=False, load_vae=False)
        self.assertIsNone(_maybe_build_wandb_logger(config, base_model=object()))

    def test_logger_raises_clear_error_when_enabled_without_vae_by_default(self):
        from generative_flow_adapters.training.builders import _maybe_build_wandb_logger

        config = self._base_config(enable_logging=True, load_vae=False)

        class BaseWithoutVae:
            first_stage_model = None

        with self.assertRaisesRegex(ValueError, "load_first_stage_model"):
            _maybe_build_wandb_logger(config, base_model=BaseWithoutVae())

    def test_metrics_only_logger_when_require_vae_false(self):
        from generative_flow_adapters.training.builders import _maybe_build_wandb_logger

        config = self._base_config(enable_logging=True, load_vae=False)
        config.training.extra["wandb"]["require_vae"] = False

        class BaseWithoutVae:
            first_stage_model = None

        with mock.patch.dict("sys.modules", {"wandb": _FakeWandb}):
            logger = _maybe_build_wandb_logger(config, base_model=BaseWithoutVae())

        self.assertIsNotNone(logger)
        # Metrics-only path: logger has no decoder, so log_videos must refuse.
        with self.assertRaisesRegex(RuntimeError, "decode_fn"):
            logger.log_videos(
                prediction_latents=torch.zeros(1, 4, 2, 4, 8),
                target_latents=torch.zeros(1, 4, 2, 4, 8),
                cond=None,
                step=1,
            )

    def test_legacy_video_logging_key_still_works(self):
        from generative_flow_adapters.training.builders import _maybe_build_wandb_logger

        # Older configs use `video_logging` instead of `wandb`.
        config = self._base_config(enable_logging=True, load_vae=False, key="video_logging")
        config.training.extra["video_logging"]["require_vae"] = False

        class BaseWithoutVae:
            first_stage_model = None

        with mock.patch.dict("sys.modules", {"wandb": _FakeWandb}):
            logger = _maybe_build_wandb_logger(config, base_model=BaseWithoutVae())
        self.assertIsNotNone(logger)


DEFAULT_CHECKPOINT_PATH = "ckts/dynami512.ckpt"
METAWORLD_DATASET_PATH = "ds/metaworld_corner2.hdf5"
OUTPUT_DIR = Path("tests/_outputs/video_logging")


def _resolve_checkpoint_path() -> Path | None:
    env_path = os.environ.get("DYNAMICRAFTER_CHECKPOINT")
    for candidate in (env_path, DEFAULT_CHECKPOINT_PATH):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


_CHECKPOINT_PATH = _resolve_checkpoint_path()
_HAS_DATASET = Path(METAWORLD_DATASET_PATH).exists()
_HAS_CUDA = torch.cuda.is_available()
_INTEGRATION_SKIP_REASON = (
    "Needs DynamiCrafter checkpoint at ckts/dynami512.ckpt "
    "(or DYNAMICRAFTER_CHECKPOINT env var), the metaworld HDF5 dataset, "
    "and CUDA (the 1.5B-param UNet won't fit on CPU)."
)


@unittest.skipUnless(_CHECKPOINT_PATH is not None and _HAS_DATASET and _HAS_CUDA, _INTEGRATION_SKIP_REASON)
class VideoLoggingPanelIntegrationTest(unittest.TestCase):
    """End-to-end: build adapter at random init, run base + adapted rollouts
    from shared noise on a real metaworld clip, decode all three latent
    trajectories, save them side-by-side to disk for visual inspection.

    No wandb. No assertions about MSE / SSIM -- the point is to give a human
    something to look at when the wandb panel looks wrong, with the same
    decode/uint8 conversion the wandb logger uses so we're not chasing two
    code paths."""

    @classmethod
    def setUpClass(cls) -> None:
        from generative_flow_adapters.training.builders import build_experiment

        cls.device = torch.device("cuda")
        cls.dtype = torch.float32
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        config = ExperimentConfig(
            model=ModelConfig(
                type="diffusion",
                provider="dynamicrafter",
                prediction_type="velocity",
                pretrained_model_name_or_path=str(_CHECKPOINT_PATH),
                extra={
                    "unet_config_path": UNET_CONFIG_PATH,
                    "latent_channels": 4,
                    "allow_dummy_concat_condition": True,
                    "load_first_stage_model": True,
                },
            ),
            adapter=AdapterConfig(
                type="hyper",
                hidden_dim=128,
                rank=4,
                extra={"architecture": "hyperalign"},
            ),
            conditioning=ConditioningConfig(type="structured", output_dim=512),
            training=TrainingConfig(loss="diffusion"),
            name="video_logging_integration",
        )
        components = build_experiment(config)
        cls.model = components.model.to(cls.device).eval()
        for parameter in cls.model.parameters():
            parameter.requires_grad_(False)

    def test_three_panel_rollout_to_disk(self):
        from generative_flow_adapters.data import (
            BatchPreprocessConfig,
            CachedNullCaptionEncoder,
            DynamiCrafterBatchPreprocessor,
            precompute_null_text_embedding,
        )
        from generative_flow_adapters.data.clip import (
            OpenCLIPImageEmbedder,
            build_dynamicrafter_resampler_from_checkpoint,
        )
        from generative_flow_adapters.inference import DiffusionInferenceSampler
        from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective

        base_model = self.model.base_model
        wrapper = base_model  # DynamicCrafterUNetWrapper instance
        vae = wrapper.first_stage_model
        self.assertIsNotNone(vae, "First-stage VAE must be attached for video logging.")

        # ---- preprocessor (same recipe the training script uses) ---------------
        # Match training exactly: real CLIP-encoded null prompt for text context,
        # OpenCLIP image embed + Resampler for image cross-attention. Anything
        # less and the UNet runs in an out-of-distribution conditioning regime
        # (text=zeros or no image branch) and the rollout looks blurry.
        null_embedding = precompute_null_text_embedding(device=self.device, dtype=self.dtype)
        caption_encoder = CachedNullCaptionEncoder(null_embedding)
        image_encoder = OpenCLIPImageEmbedder().to(self.device)
        image_encoder.eval()
        image_resampler = build_dynamicrafter_resampler_from_checkpoint(
            str(_CHECKPOINT_PATH), video_length=16, device=self.device,
        )

        preprocessor = DynamiCrafterBatchPreprocessor(
            vae=vae,
            config=BatchPreprocessConfig(
                target_height=320,
                target_width=512,
                resize_mode="stretch",
                uncond_prob=0.0,
                cond_frame_index=0,
                rand_cond_frame=False,
                context_tokens=77,
                context_dim=1024,
            ),
            caption_encoder=caption_encoder,
            image_encoder=image_encoder,
            image_resampler=image_resampler,
        )

        # ---- load one metaworld clip and shape it like the dataloader would ----
        raw_video = _load_metaworld_clip(num_frames=16)
        if raw_video is None:
            self.skipTest("Metaworld dataset not present.")
        raw_batch = {"video": raw_video}  # uint8 [1, T, H, W, 3]
        batch = preprocessor(raw_batch, train=False)
        target = batch["target"].to(device=self.device, dtype=self.dtype)
        cond = batch["cond"]

        # ---- build the inference samplers exactly the way the trainer does ----
        schedule = wrapper.diffusion_schedule_config or {}
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
        adapted_sampler = DiffusionInferenceSampler(
            model=self.model, objective=objective,
            prediction_type="velocity", scheduler_name="ddim",
        )
        base_sampler = DiffusionInferenceSampler(
            model=base_model, objective=objective,
            prediction_type="velocity", scheduler_name="ddim",
        )

        # ---- shared noise so any visible diff between base/adapted is the adapter ----
        shared_noise = torch.randn_like(target)

        with torch.no_grad():
            adapted_latents = adapted_sampler.sample(
                shape=tuple(target.shape), cond=cond,
                device=self.device, dtype=self.dtype,
                num_inference_steps=25, initial_sample=shared_noise,
            )
            # Base-only rollout: drop adapter-only keys (matches _strip_adapter_only_keys).
            base_cond = {k: v for k, v in cond.items() if k != "embedding"}
            base_latents = base_sampler.sample(
                shape=tuple(target.shape), cond=base_cond,
                device=self.device, dtype=self.dtype,
                num_inference_steps=25, initial_sample=shared_noise,
            )

        # ---- decode via the same path WandbLogger uses (any bug shows here too) ----
        gt_pixels = _decode_to_uint8(wrapper.decode_first_stage(target))           # [B, T, 3, H, W]
        base_pixels = _decode_to_uint8(wrapper.decode_first_stage(base_latents))
        adapted_pixels = _decode_to_uint8(wrapper.decode_first_stage(adapted_latents))

        self.assertEqual(gt_pixels.shape, base_pixels.shape)
        self.assertEqual(gt_pixels.shape, adapted_pixels.shape)
        self.assertTrue(torch.isfinite(target).all(), "ground-truth latents non-finite")
        self.assertTrue(torch.isfinite(base_latents).all(), "base rollout non-finite")
        self.assertTrue(torch.isfinite(adapted_latents).all(), "adapted rollout non-finite")

        # ---- save first-4-frames PNG grid + full-clip side-by-side MP4 ---------
        sample_index = 0
        png_path = OUTPUT_DIR / f"sample{sample_index}_first4frames.png"
        mp4_path = OUTPUT_DIR / f"sample{sample_index}_side_by_side.mp4"
        _save_first_n_frames_grid(
            gt=gt_pixels[sample_index],
            base=base_pixels[sample_index],
            adapted=adapted_pixels[sample_index],
            num_frames=4,
            out_path=png_path,
        )
        _save_side_by_side_mp4(
            gt=gt_pixels[sample_index],
            base=base_pixels[sample_index],
            adapted=adapted_pixels[sample_index],
            out_path=mp4_path,
            fps=4,
        )
        print(f"\nWrote {png_path} and {mp4_path} for visual inspection.")
        self.assertTrue(png_path.exists())
        self.assertTrue(mp4_path.exists())


def _load_metaworld_clip(*, num_frames: int) -> torch.Tensor | None:
    """Return a uint8 ``[1, T, H, W, 3]`` clip from the metaworld HDF5, the
    same shape ``TranslatedClipDataset`` would yield. Returns ``None`` if the
    dataset isn't present."""
    if not Path(METAWORLD_DATASET_PATH).exists():
        return None
    try:
        import h5py
    except ImportError:
        return None
    import random

    with h5py.File(METAWORLD_DATASET_PATH, "r") as handle:
        tasks = list(handle.keys())
        if not tasks:
            return None
        task = handle[random.choice(tasks)]
        episodes = [name for name in task.keys() if "pixels" in task[name]]
        if not episodes:
            return None
        pixels = task[random.choice(episodes)]["pixels"]  # [T_total, H, W, 3] uint8
        if pixels.shape[0] < num_frames:
            return None
        start = random.randint(0, pixels.shape[0] - num_frames)
        clip = pixels[start : start + num_frames]                     # [T, H, W, 3]
    return torch.from_numpy(clip)[None]                                # [1, T, H, W, 3]


@torch.no_grad()
def _decode_to_uint8(decoded_pixels: torch.Tensor) -> torch.Tensor:
    """Same conversion as `WandbLogger._decode_to_uint8` so any bug there
    shows up identically here.

    Input: ``[B, 3, T, H, W]`` in roughly ``[-1, 1]``.
    Output: ``[B, T, 3, H, W]`` uint8.
    """
    if decoded_pixels.dim() != 5:
        raise ValueError(f"Decoder must return 5D [B, 3, T, H, W]; got {tuple(decoded_pixels.shape)}.")
    out = decoded_pixels.clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
    return out.permute(0, 2, 1, 3, 4).to(torch.uint8).cpu()


def _save_first_n_frames_grid(
    *,
    gt: torch.Tensor,            # [T, 3, H, W] uint8
    base: torch.Tensor,
    adapted: torch.Tensor,
    num_frames: int,
    out_path: Path,
) -> None:
    """Save a 3-row by num_frames-col PNG: rows are (gt, base, adapted),
    columns are frames 0..num_frames-1."""
    from PIL import Image

    n = min(num_frames, gt.shape[0])
    rows: list[torch.Tensor] = []
    for label_tensor in (gt, base, adapted):
        # [n, 3, H, W] -> concat along W -> [3, H, n*W]
        row = torch.cat(list(label_tensor[:n]), dim=-1)
        rows.append(row)
    # Stack the 3 rows along H -> [3, 3*H, n*W]
    grid = torch.cat(rows, dim=-2)
    grid_hwc = grid.permute(1, 2, 0).numpy()  # [H, W, 3]
    Image.fromarray(grid_hwc).save(out_path)


def _save_side_by_side_mp4(
    *,
    gt: torch.Tensor,            # [T, 3, H, W] uint8
    base: torch.Tensor,
    adapted: torch.Tensor,
    out_path: Path,
    fps: int,
) -> None:
    """Concat (gt | base | adapted) along width per frame, write as MP4.

    Mirrors `WandbLogger.log_videos`'s ``torch.cat(panels, dim=-1)``, so
    the resulting video is what wandb would have shown if it worked."""
    import imageio.v3 as iio

    side_by_side = torch.cat([gt, base, adapted], dim=-1)  # [T, 3, H, W*3]
    frames_thwc = side_by_side.permute(0, 2, 3, 1).numpy()  # [T, H, W*3, 3]
    iio.imwrite(out_path, frames_thwc, fps=fps, codec="libx264", macro_block_size=1)


if __name__ == "__main__":
    unittest.main()
