"""Regression tests for ``_ComposedDiT`` CFG-forward memoization.

Wan's denoising loop calls the DiT twice per step (positive then negative
context) and combines them as ``uncond + g·(cond - uncond)``. In frame-only /
prompt-free generation both branches use the *same* unconditional embedding, so
the two calls are identical and the second is pure waste. ``_ComposedDiT``
memoizes a 1-entry cache so the frozen base (and, when adapted, the adapter) run
once per step in that case, while a genuine positive!=negative prompt still runs
both branches (real CFG).

CPU-only: uses a fake DiT that counts forwards; no Wan repo / CUDA needed.
"""

from __future__ import annotations

import unittest

import torch

from generative_flow_adapters.models.base.wan_ti2v import _ComposedDiT


class _FakeDiT:
    """Counts forwards; deterministic output depends on x and the context."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x, t, context, seq_len, y=None):
        self.calls += 1
        bias = float(context[0].sum())
        return [xi * 2.0 + bias for xi in x]

    def to(self, *a, **k):  # upstream calls .model.to(device)
        return self


def _latent():
    return [torch.randn(48, 3, 2, 2)]


def _per_token_t(value: float, n: int = 8) -> torch.Tensor:
    return torch.full((1, n), float(value))


class TestComposedDiTMemoize(unittest.TestCase):
    def test_frame_only_runs_single_forward(self) -> None:
        """pos == neg (same context object) -> second call is a cache hit."""
        dit = _FakeDiT()
        w = _ComposedDiT(dit, compose_fn=None)
        x, t, ctx = _latent(), _per_token_t(10.0), [torch.randn(3, 4)]

        cond = w(x, t, ctx, seq_len=8)   # upstream cond branch
        unc = w(x, t, ctx, seq_len=8)    # upstream uncond branch (identical)

        self.assertEqual(dit.calls, 1)
        self.assertIs(cond, unc)
        # CFG term must vanish *exactly* (same object -> difference is 0).
        noise_pred = unc[0] + 5.0 * (cond[0] - unc[0])
        self.assertTrue(torch.equal(noise_pred, unc[0]))

    def test_real_cfg_runs_both_forwards(self) -> None:
        """pos != neg (different context objects) -> both branches run."""
        dit = _FakeDiT()
        w = _ComposedDiT(dit, compose_fn=None)
        x, t = _latent(), _per_token_t(10.0)
        ctx_pos, ctx_neg = [torch.randn(3, 4)], [torch.randn(3, 4)]

        p = w(x, t, ctx_pos, seq_len=8)
        u = w(x, t, ctx_neg, seq_len=8)

        self.assertEqual(dit.calls, 2)
        self.assertFalse(torch.equal(p[0], u[0]))

    def test_next_step_is_cache_miss(self) -> None:
        """A new step (fresh latent + timestep) must recompute, not reuse."""
        dit = _FakeDiT()
        w = _ComposedDiT(dit, compose_fn=None)
        ctx = [torch.randn(3, 4)]

        # Upstream builds `latent_model_input` once per step and passes the SAME
        # object to both CFG branches -> the two same-step calls share id(x[0]).
        x_step1 = _latent()
        w(x_step1, _per_token_t(10.0), ctx, seq_len=8)
        w(x_step1, _per_token_t(10.0), ctx, seq_len=8)  # same step -> hit
        self.assertEqual(dit.calls, 1)
        w(_latent(), _per_token_t(20.0), ctx, seq_len=8)  # next step -> miss
        self.assertEqual(dit.calls, 2)

    def test_adapted_frame_only_composes_once(self) -> None:
        """Adapted path: base + adapter each run once in frame-only mode, and the
        delta is applied and cancels through CFG to base + delta."""
        compose_calls = {"n": 0}

        def compose_fn(x_b, t, base_b):
            compose_calls["n"] += 1
            return base_b + 0.5  # constant adapter delta

        dit = _FakeDiT()
        w = _ComposedDiT(dit, compose_fn=compose_fn)
        x, t, ctx = _latent(), _per_token_t(7.0), [torch.randn(3, 4)]

        cond = w(x, t, ctx, seq_len=8)
        unc = w(x, t, ctx, seq_len=8)

        self.assertEqual(dit.calls, 1)
        self.assertEqual(compose_calls["n"], 1)
        base_only = x[0] * 2.0 + float(ctx[0].sum())
        self.assertTrue(torch.allclose(cond[0], base_only + 0.5))
        noise_pred = unc[0] + 5.0 * (cond[0] - unc[0])
        self.assertTrue(torch.equal(noise_pred, unc[0]))


if __name__ == "__main__":
    unittest.main()
