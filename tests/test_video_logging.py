"""Tests for the DynamiCrafter VAE wrapper and the wandb video logger
plumbing. We avoid calling wandb itself; we only check that:

1. The wrapper can build a VideoAutoencoderKL from the UNet YAML and decode.
2. The builder constructs the logger when configured AND the base has a VAE,
   raises a clear error when configured but the base has no VAE, and skips
   logger construction when disabled.
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
        # VAE upsamples 8x; output should be [B, 3, T, 8*H, 8*W].
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


class WandbVideoLoggerPanelTest(unittest.TestCase):
    """Verify the 3-panel video assembly when the base-only rollout is supplied."""

    def _make_logger(self, fake_log):
        from generative_flow_adapters.training.video_logger import WandbVideoLogger

        class FakeWandb:
            run = object()

            class Video:
                def __init__(self, data, fps, format, caption):
                    self.data = data
                    self.fps = fps
                    self.format = format
                    self.caption = caption

            @staticmethod
            def log(payload, step):
                fake_log.append((payload, step))

        # Identity decode_fn: pretend latents are already pixels. Avoids
        # standing up a real VAE for the panel-stitching check.
        def decode_fn(latents):
            # Pretend the VAE turns the 4-ch latent into a 3-ch pixel video by dropping a channel.
            return latents[:, :3]

        with mock.patch.dict("sys.modules", {"wandb": FakeWandb}):
            logger = WandbVideoLogger(decode_fn=decode_fn, num_samples=1, fps=4)
        return logger, FakeWandb

    def test_three_panel_video_when_base_provided(self):
        fake_log: list = []
        logger, _ = self._make_logger(fake_log)
        target = torch.zeros(1, 4, 2, 4, 8)
        adapted = torch.zeros(1, 4, 2, 4, 8)
        base = torch.zeros(1, 4, 2, 4, 8)

        logger.log(
            prediction_latents=adapted,
            target_latents=target,
            base_prediction_latents=base,
            cond={"act": torch.randn(1, 2, 4)},
            step=42,
        )

        self.assertEqual(len(fake_log), 1)
        payload, step = fake_log[0]
        self.assertEqual(step, 42)
        self.assertIn("eval/sample_0", payload)
        video = payload["eval/sample_0"]
        # Three panels concatenated along width: 8 + 8 + 8 = 24.
        self.assertEqual(video.data.shape[-1], 24)
        self.assertIn("base_model", video.caption)

    def test_two_panel_video_when_base_absent(self):
        fake_log: list = []
        logger, _ = self._make_logger(fake_log)
        target = torch.zeros(1, 4, 2, 4, 8)
        adapted = torch.zeros(1, 4, 2, 4, 8)

        logger.log(
            prediction_latents=adapted,
            target_latents=target,
            cond=None,
            step=7,
        )

        payload, _ = fake_log[0]
        video = payload["eval/sample_0"]
        self.assertEqual(video.data.shape[-1], 16)  # 8 + 8
        self.assertNotIn("base_model", video.caption)


class VideoLoggerBuilderTest(unittest.TestCase):
    """Validate the gating in `_maybe_build_video_logger` without calling wandb."""

    def _base_config(self, *, enable_logging: bool, load_vae: bool) -> ExperimentConfig:
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
                extra={
                    "video_logging": {
                        "enable": enable_logging,
                        "num_samples": 2,
                        "fps": 4,
                    }
                },
            ),
            name="test_video_logging",
        )

    def test_logger_is_none_when_disabled(self):
        from generative_flow_adapters.training.builders import _maybe_build_video_logger

        config = self._base_config(enable_logging=False, load_vae=False)
        # base_model is irrelevant when disabled — it isn't even inspected.
        self.assertIsNone(_maybe_build_video_logger(config, base_model=object()))

    def test_logger_raises_clear_error_when_enabled_without_vae(self):
        from generative_flow_adapters.training.builders import _maybe_build_video_logger

        config = self._base_config(enable_logging=True, load_vae=False)

        class BaseWithoutVae:
            first_stage_model = None

        with self.assertRaisesRegex(ValueError, "load_first_stage_model"):
            _maybe_build_video_logger(config, base_model=BaseWithoutVae())

    def test_logger_constructed_when_enabled_with_vae(self):
        # WandbVideoLogger's __init__ calls wandb.init when no run exists; we
        # patch the wandb module the logger imports so the construction stays
        # local. This catches build-path regressions even without wandb installed.
        from generative_flow_adapters.training.builders import _maybe_build_video_logger

        config = self._base_config(enable_logging=True, load_vae=False)  # we'll stub the VAE instead

        class FakeWandb:
            run = object()  # truthy → __init__ won't call wandb.init

            class Video:  # noqa: D401 - lightweight stub
                def __init__(self, *args, **kwargs):
                    pass

            @staticmethod
            def log(*args, **kwargs):
                pass

        class BaseWithVae:
            first_stage_model = object()

            def decode_first_stage(self, latents):
                raise NotImplementedError  # never called in this test

        with mock.patch.dict("sys.modules", {"wandb": FakeWandb}):
            logger = _maybe_build_video_logger(config, base_model=BaseWithVae())

        self.assertIsNotNone(logger)
        self.assertEqual(logger.num_samples, 2)
        self.assertEqual(logger.fps, 4)


if __name__ == "__main__":
    unittest.main()
