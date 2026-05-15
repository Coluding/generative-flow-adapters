"""Tests for the DynamiCrafter VAE wrapper and the WandbLogger plumbing.

We avoid actually calling wandb; we only check that:

1. The wrapper can build a `VideoAutoencoderKL` from the UNet YAML and decode.
2. `WandbLogger.log_videos` produces the right number of side-by-side panels
   and `log_metrics` filters non-scalar entries.
3. The builder constructs the logger when configured and the base has a VAE,
   raises a clear error when the base has no VAE (default), and skips logger
   construction when disabled.
"""

from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
