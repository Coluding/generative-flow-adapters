"""Correctness guards for the two throughput optimisations landed 2026-07-29.

Both remove *redundant* work from the Wan shortcut training step, so the thing
worth testing is that they change nothing observable:

1. **Base-output reuse** — the shortcut self-consistency prep and the training
   forward run the frozen base at the same ``(x_t, t)``, differing only in
   ``step_level``, which the base never sees. Reusing the first result must be
   bit-identical to recomputing it.
2. **Latent prefetch collate** — workers hand back latents (hit) or pixels
   (miss); the collate has to survive an all-hit, all-miss, or mixed batch.

See ``data/latent_prefetch.py`` and ``AdaptedModel.reuses_base_output``.
"""

from __future__ import annotations

import torch
from torch import nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.data.latent_prefetch import collate_latent_windows
from generative_flow_adapters.models.adapted_model import AdaptedModel
from generative_flow_adapters.training.shortcut_targets import (
    compute_self_consistency_target_v_flow,
)


class _CountingBase(nn.Module):
    """Frozen-backbone stand-in: a pure function of ``(x_t, t)`` that counts how
    many times it actually ran, mirroring ``WanTI2VVideoModel.denoise``."""

    model_type = "flow"
    prediction_type = "velocity"
    diffusion_schedule_config = None

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, x_t, t, cond=None):  # noqa: ARG002 — cond is deliberately ignored
        self.calls += 1
        return self.lin(x_t) * t.view(-1, 1)


class _StepLevelAdapter(Adapter):
    """Output adapter whose delta *does* depend on ``step_level`` — so a test that
    passes can't be passing because step_level is inert everywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x_t, t, cond, base_output=None):  # noqa: ARG002
        step_level = cond.get("step_level") if isinstance(cond, dict) else None
        scale = step_level.view(-1, 1) if isinstance(step_level, torch.Tensor) else 1.0
        return self.lin(x_t) * scale


def _fixture():
    torch.manual_seed(0)
    model = AdaptedModel(base_model=_CountingBase(), adapter=_StepLevelAdapter())
    x_t = torch.randn(4, 8)
    t = torch.rand(4)
    return model, x_t, t


def test_base_output_is_independent_of_step_level():
    """The precondition for reuse: shortcut prep and the training forward differ
    only in step_level, which must not reach the frozen base."""
    model, x_t, t = _fixture()
    _, base_half = model(x_t, t, {"step_level": torch.full((4,), 0.25)}, return_base=True)
    _, base_full = model(x_t, t, {"step_level": torch.full((4,), 0.5)}, return_base=True)
    assert torch.equal(base_half, base_full)


def test_reused_base_output_is_bit_identical_and_skips_the_forward():
    model, x_t, t = _fixture()
    cond = {"step_level": torch.full((4,), 0.5)}
    _, base = model(x_t, t, cond, return_base=True)

    before = model.base_model.calls
    recomputed = model(x_t, t, cond)
    assert model.base_model.calls == before + 1

    before = model.base_model.calls
    reused = model(x_t, t, cond, base_output=base)
    assert model.base_model.calls == before, "base_output= still ran the base"
    assert torch.equal(recomputed, reused)


def test_shortcut_prep_returns_the_main_forwards_base_output():
    """``return_base=True`` must hand back the *first* call's base prediction (taken
    at the training forward's ``(x_t, t)``), and must not perturb the target."""
    model, x_t, t = _fixture()
    cond_half = {"step_level": torch.full((4,), 0.25)}
    d = torch.full((4,), 0.25)
    _, expected_base = model(x_t, t, {"step_level": torch.full((4,), 0.5)}, return_base=True)

    target, base_v1 = compute_self_consistency_target_v_flow(
        model=model, x_t=x_t, t=t, cond_half=cond_half, d=d,
        timestep_scale=1.0, return_base=True,
    )
    plain = compute_self_consistency_target_v_flow(
        model=model, x_t=x_t, t=t, cond_half=cond_half, d=d, timestep_scale=1.0,
    )
    assert torch.equal(base_v1, expected_base)
    assert torch.equal(target, plain)


def test_adapters_that_capture_base_internals_opt_out_of_reuse():
    """UniCon/HyperAlign read the base's *hidden states*, which the intervening
    micro-step forward overwrites — they must keep recomputing."""
    model, _, _ = _fixture()
    assert model.reuses_base_output

    class _Capturing(_StepLevelAdapter):
        def clear_captured_base_features(self) -> None:
            return None

    assert not AdaptedModel(base_model=_CountingBase(), adapter=_Capturing()).reuses_base_output


def _clip(**over):
    item = {
        "act": torch.randn(5, 7), "env_name": "e", "episode_idx": 0,
        "start_idx": 0, "frame_stride": 1, "caption": "c",
    }
    item.update(over)
    return item


def test_collate_all_hits_stacks_latents_and_drops_pixels():
    batch = collate_latent_windows([_clip(z0=torch.randn(4, 2, 3, 3)) for _ in range(3)])
    assert batch["z0"].shape == (3, 4, 2, 3, 3)
    assert "video" not in batch and "z0_list" not in batch
    assert batch["act"].shape == (3, 5, 7)


def test_collate_all_misses_falls_back_to_pixels():
    video = torch.zeros(6, 8, 8, 3, dtype=torch.uint8)
    batch = collate_latent_windows([_clip(video=video) for _ in range(3)])
    assert batch["video"].shape == (3, 6, 8, 8, 3)
    assert "z0" not in batch


def test_collate_mixed_batch_keeps_both_per_sample():
    """A partially precomputed cache must still train — default_collate would
    reject this batch outright."""
    items = [_clip(z0=torch.randn(4, 2, 3, 3)),
             _clip(video=torch.zeros(6, 8, 8, 3, dtype=torch.uint8))]
    batch = collate_latent_windows(items)
    assert isinstance(batch["z0_list"][0], torch.Tensor) and batch["z0_list"][1] is None
    assert batch["video_list"][0] is None and isinstance(batch["video_list"][1], torch.Tensor)
