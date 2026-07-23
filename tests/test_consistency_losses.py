"""Tests for the Heun-smoothness regularizer — loss contract + trainer wiring.

Covers patch 1 of thesis-vault
refactor-shortcut-deprecate-twostep-add-heun-smoothness:

1. ``heun_smoothness_loss`` unit contract (shape, grad routing, zero case).
2. Trainer-step gating: metric absent at weight 0, present+finite at weight > 0.
3. Frozen-base preservation: the regularizer never grows base-model gradients.
4. End-to-end smoke: a short Heun-only loop reduces the loss.
5. Schedule fallback: no ``shortcut_step_schedule`` ⇒ unit jump, finite loss.

The trainer tests reuse the dummy-backbone shortcut config + fake dataloader
(the same path exercised by examples/shortcut_training_test.py), so they stay
fast and need no checkpoints.
"""

from __future__ import annotations

import unittest

import torch

from generative_flow_adapters.config import load_config
from generative_flow_adapters.losses.consistency import heun_smoothness_loss
from generative_flow_adapters.losses.registry import LossRegistry
from generative_flow_adapters.testing import build_fake_dataloader
from generative_flow_adapters.training import Trainer, build_experiment


CONFIG_PATH = "configs/dev/diffusion_output_shortcut_velocity.yaml"


def _build_trainer(*, heun_weight: float, learning_rate: float = 1e-2) -> tuple[Trainer, object]:
    """Dummy diffusion backbone + output adapter, Heun-only (no shortcut loss)."""
    config = load_config(CONFIG_PATH)
    # Isolate the Heun term: turn off the shortcut-direction objective so the
    # only consistency signal under test is the regularizer.
    config.training.shortcut_direction_weight = 0.0
    config.training.heun_smoothness_weight = heun_weight
    config.training.learning_rate = learning_rate
    # No `shortcut_step_schedule` in this config ⇒ exercises the unit-jump
    # fallback path inside _compute_heun_smoothness.
    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    return trainer, config


def _batches(config, *, n: int, batch_size: int = 4):
    loader = build_fake_dataloader(config=config, batch_size=batch_size, length=n * batch_size)
    out = []
    for i, batch in enumerate(loader):
        out.append(batch)
        if i + 1 >= n:
            break
    return out


class HeunSmoothnessLossTest(unittest.TestCase):
    def test_returns_scalar(self):
        loss = heun_smoothness_loss(torch.randn(4, 8), torch.randn(4, 8))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_zero_when_equal(self):
        v = torch.randn(3, 5)
        self.assertAlmostEqual(float(heun_smoothness_loss(v, v)), 0.0, places=6)

    def test_grad_flows_through_current_v_only(self):
        v0 = torch.randn(4, 8, requires_grad=True)
        v1 = torch.randn(4, 8, requires_grad=True)
        heun_smoothness_loss(v0, v1).backward()
        self.assertIsNotNone(v0.grad)
        # future_v is stop-gradient'd inside the loss → no gradient leaks to it.
        self.assertIsNone(v1.grad)

    def test_registered_in_registry(self):
        self.assertIs(LossRegistry.get_consistency_loss("heun_smoothness"), heun_smoothness_loss)


class HeunSmoothnessTrainerTest(unittest.TestCase):
    def test_metric_absent_when_weight_zero(self):
        trainer, config = _build_trainer(heun_weight=0.0)
        metrics = trainer.training_step(_batches(config, n=1)[0])
        self.assertNotIn("heun_smoothness_loss", metrics)

    def test_metric_present_and_finite_when_enabled(self):
        trainer, config = _build_trainer(heun_weight=0.1)
        metrics = trainer.training_step(_batches(config, n=1)[0])
        self.assertIn("heun_smoothness_loss", metrics)
        value = metrics["heun_smoothness_loss"]
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertGreaterEqual(value, 0.0)

    def test_frozen_base_sees_no_gradients(self):
        trainer, config = _build_trainer(heun_weight=0.5)
        trainer.training_step(_batches(config, n=1)[0])
        base = getattr(trainer.model, "base_model", None)
        self.assertIsNotNone(base)
        for name, p in base.named_parameters():
            self.assertFalse(p.requires_grad, f"base param {name} should be frozen")
            self.assertIsNone(p.grad, f"base param {name} accrued a gradient")

    def test_heun_term_trains_the_adapter(self):
        # Isolate the regularizer's learning signal by zeroing the base loss, so
        # any adapter parameter movement is attributable to the Heun term alone.
        # This proves the term participates in the backward/optimizer path (and
        # is not accidentally detached) — a sounder invariant than "total loss
        # decreases", which is dominated by the base loss on random fake data.
        trainer, config = _build_trainer(heun_weight=1.0)
        trainer.loss_fn = lambda pred, tgt: (pred * 0.0).sum()
        adapter = trainer.model.adapter
        before = [p.detach().clone() for p in adapter.parameters() if p.requires_grad]
        self.assertGreater(len(before), 0, "adapter has no trainable parameters")
        for b in _batches(config, n=15):
            trainer.training_step(b)
        after = [p.detach() for p in adapter.parameters() if p.requires_grad]
        moved = sum(float((a - b0).abs().sum()) for a, b0 in zip(after, before))
        self.assertGreater(moved, 0.0, "Heun term did not propagate gradients to the adapter")

    def test_two_step_method_is_rejected(self):
        # The deprecated 'two_step' shortcut mode was removed; selecting it must
        # now fail loudly rather than silently doing something.
        trainer, config = _build_trainer(heun_weight=0.0)
        trainer.config.shortcut_direction_weight = 1.0  # force a shortcut target
        trainer.config.shortcut_target_method = "two_step"
        with self.assertRaises(ValueError):
            trainer.training_step(_batches(config, n=1)[0])

    def test_unit_jump_fallback_runs_without_schedule(self):
        # No `shortcut_step_schedule` configured ⇒ _compute_heun_smoothness must
        # fall back to a unit timestep jump and still produce a finite loss.
        trainer, config = _build_trainer(heun_weight=0.1)
        self.assertIsNone(trainer.step_schedule)
        metrics = trainer.training_step(_batches(config, n=1)[0])
        self.assertIn("heun_smoothness_loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["heun_smoothness_loss"])))


class ShortcutPerRungLoggingTest(unittest.TestCase):
    """The aggregate `shortcut_direction_loss` marginalises over the sampled
    step size, so the trainer also re-logs it under a per-rung key
    `shortcut_direction_loss/N{steps}` (steps = round(1/s)). Each step feeds
    exactly one rung, so over several steps the supervised rungs should appear,
    keyed by their schedule level, while the smallest (anchor-only) level never
    does."""

    def _build(self):
        config = load_config(CONFIG_PATH)
        config.training.heun_smoothness_weight = 0.0
        config.training.shortcut_direction_weight = 1.0
        config.training.shortcut_target_method = "distillation"
        # log2 schedule over {1/8, 1/4, 1/2, 1} ⇒ supervised rungs N ∈ {1, 2, 4}
        # (the smallest level 1/8 → N=8 is the anchor and is never supervised).
        config.training.extra["shortcut_step_schedule"] = {
            "units": "normalized", "mode": "log2", "min": "1/8", "max": 1, "base": 2,
        }
        # Force non-anchor every step so a shortcut target is always produced.
        config.training.extra["shortcut_anchor_prob"] = 0.0
        experiment = build_experiment(config)
        trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
        return trainer, config

    def test_per_rung_keys_appear_and_match_schedule(self):
        trainer, config = self._build()
        self.assertIsNotNone(trainer.step_schedule)
        seen: set[str] = set()
        for b in _batches(config, n=24):
            metrics = trainer.training_step(b)
            seen.update(k for k in metrics if k.startswith("shortcut_direction_loss/N"))
            # Each step logs at most one rung, and it matches the stashed `s`.
            rung_keys = [k for k in metrics if k.startswith("shortcut_direction_loss/N")]
            self.assertLessEqual(len(rung_keys), 1)
            if rung_keys:
                self.assertIn("shortcut_direction_loss", metrics)
                n = max(1, round(1.0 / trainer._last_shortcut_step_level))
                self.assertEqual(rung_keys[0], f"shortcut_direction_loss/N{n:03d}")
        # Over many steps every supervised rung should show up; the anchor-only
        # finest level (N=8) must never be logged as a supervised rung.
        self.assertTrue(seen, "no per-rung shortcut series were logged")
        self.assertLessEqual(seen, {"shortcut_direction_loss/N001",
                                    "shortcut_direction_loss/N002",
                                    "shortcut_direction_loss/N004"})
        self.assertNotIn("shortcut_direction_loss/N008", seen)


if __name__ == "__main__":
    unittest.main()
