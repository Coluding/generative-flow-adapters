"""Tests for the configurable shortcut step-size schedule."""

from __future__ import annotations

import math

import pytest

from generative_flow_adapters.training.step_schedule import ShortcutStepSchedule


def _approx_set(values, expected, tol=1e-6):
    values = sorted(values)
    expected = sorted(expected)
    assert len(values) == len(expected), (values, expected)
    for v, e in zip(values, expected):
        assert math.isclose(v, e, rel_tol=0, abs_tol=tol), (values, expected)


class TestExplicit:
    def test_normalized_values_sorted_and_deduped(self):
        s = ShortcutStepSchedule.from_config(
            {"units": "normalized", "mode": "explicit", "values": [1.0, 0.25, 0.5, 0.25]},
            timesteps=1000,
        )
        _approx_set(s.discrete_levels(), [0.25, 0.5, 1.0])
        assert s.smallest() == pytest.approx(0.25)

    def test_timesteps_values_are_normalized_by_T(self):
        s = ShortcutStepSchedule.from_config(
            {"units": "timesteps", "mode": "explicit", "values": [250, 500, 1000]},
            timesteps=1000,
        )
        _approx_set(s.discrete_levels(), [0.25, 0.5, 1.0])

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            ShortcutStepSchedule.from_config({"mode": "explicit", "values": []}, timesteps=1000)


class TestLog2:
    def test_paper_dyadic_ladder_normalized(self):
        # 1/128 .. 1 base 2 → the paper's 8 step sizes.
        s = ShortcutStepSchedule.from_config(
            {"units": "normalized", "mode": "log2", "min": 1 / 128, "max": 1.0, "base": 2},
            timesteps=1000,
        )
        expected = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
        _approx_set(s.discrete_levels(), expected)
        assert len(s.discrete_levels()) == 8

    def test_timesteps_ladder_1_to_512(self):
        # User's "1,2,4,...,512" expressed in timestep units, T=1000.
        s = ShortcutStepSchedule.from_config(
            {"units": "timesteps", "mode": "log2", "min": 1, "max": 512, "base": 2},
            timesteps=1000,
        )
        expected = [v / 1000 for v in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)]
        _approx_set(s.discrete_levels(), expected)

    def test_base_must_exceed_one(self):
        with pytest.raises(ValueError):
            ShortcutStepSchedule.from_config({"mode": "log2", "min": 0.1, "max": 1, "base": 1}, timesteps=1000)


class TestFractionStrings:
    def test_log2_bounds_accept_fraction_strings(self):
        # YAML parses `1/128` as the string "1/128"; it must equal the float form.
        frac = ShortcutStepSchedule.from_config(
            {"units": "normalized", "mode": "log2", "min": "1/128", "max": "1", "base": 2},
            timesteps=1000,
        )
        flt = ShortcutStepSchedule.from_config(
            {"units": "normalized", "mode": "log2", "min": 0.0078125, "max": 1.0, "base": 2},
            timesteps=1000,
        )
        _approx_set(frac.discrete_levels(), flt.discrete_levels())

    def test_explicit_fraction_list(self):
        s = ShortcutStepSchedule.from_config(
            {"mode": "explicit", "values": ["1/8", "1/4", "1/2", 1]},
            timesteps=1000,
        )
        _approx_set(s.discrete_levels(), [0.125, 0.25, 0.5, 1.0])

    def test_decimal_and_scientific_strings_still_parse(self):
        s = ShortcutStepSchedule.from_config(
            {"mode": "explicit", "values": ["0.25", "1e-2", 0.5]}, timesteps=1000
        )
        _approx_set(s.discrete_levels(), [0.01, 0.25, 0.5])


class TestUniform:
    def test_range_has_no_discrete_levels(self):
        s = ShortcutStepSchedule.from_config(
            {"units": "normalized", "mode": "uniform", "min": 0.0, "max": 1.0},
            timesteps=1000,
        )
        assert s.discrete_levels() is None
        # min 0 is clamped into (0,1] at 1/T.
        assert s.low == pytest.approx(1 / 1000)
        assert s.high == pytest.approx(1.0)

    def test_sample_within_bounds(self):
        s = ShortcutStepSchedule.from_config(
            {"mode": "uniform", "min": 0.1, "max": 0.9}, timesteps=1000
        )
        for _ in range(50):
            v = s.sample()
            assert 0.1 - 1e-9 <= v <= 0.9 + 1e-9


class TestSamplingAndJumps:
    def test_discrete_sample_is_a_level(self):
        s = ShortcutStepSchedule.from_config(
            {"mode": "explicit", "values": [0.125, 0.25, 0.5, 1.0]}, timesteps=1000
        )
        levels = set(s.discrete_levels())
        for _ in range(50):
            assert s.sample() in levels

    def test_exclude_smallest_never_returns_finest(self):
        s = ShortcutStepSchedule.from_config(
            {"mode": "explicit", "values": [0.125, 0.25, 0.5, 1.0]}, timesteps=1000
        )
        for _ in range(50):
            assert s.sample(exclude_smallest=True) != pytest.approx(0.125)

    def test_to_timestep_jump_rounds_and_clamps(self):
        s = ShortcutStepSchedule.from_config({"mode": "log2", "min": 1 / 128, "max": 1}, timesteps=1000)
        assert s.to_timestep_jump(1.0) == 1000          # one-step = whole trajectory
        assert s.to_timestep_jump(0.5) == 500
        assert s.to_timestep_jump(1 / 1000) == 1         # clamps to >= 1
        assert s.to_timestep_jump(0.0001) == 1           # sub-timestep clamps to 1


class TestStepLevelTransform:
    def test_log2_transform_spreads_dyadic_into_integer_range(self):
        import torch

        from generative_flow_adapters.conditioning.utils.dynamicrafter_conditioning import (
            _apply_step_level_transform,
        )

        s = torch.tensor([1 / 128, 1 / 8, 1.0])
        out = _apply_step_level_transform(s, "log2")
        # log2 of the dyadic normalised levels → exact negative integers / 0.
        assert torch.allclose(out, torch.tensor([-7.0, -3.0, 0.0]), atol=1e-5)

    def test_linear_transform_is_identity(self):
        import torch

        from generative_flow_adapters.conditioning.utils.dynamicrafter_conditioning import (
            _apply_step_level_transform,
        )

        s = torch.tensor([0.25, 0.5, 1.0])
        assert torch.equal(_apply_step_level_transform(s, "linear"), s)

    def test_unknown_transform_raises(self):
        import torch

        from generative_flow_adapters.conditioning.utils.dynamicrafter_conditioning import (
            _apply_step_level_transform,
        )

        with pytest.raises(ValueError):
            _apply_step_level_transform(torch.tensor([0.5]), "bogus")
