"""Endpoint-inversion shortcut target (Option A) — exactness regression tests.

Covers thesis-vault bug-losses-shortcut-v-averaging-target:

1. ``invert_ddim_v`` is the exact algebraic inverse of ``ddim_micro_step_v``
   (round-trip), with and without the dynamic-rescale ``scale_arr`` branch.
2. On a synthetic consistent ray, one ``2d`` DDIM step with the
   ``endpoint_inversion`` target reproduces the two-step landing to < 1e-5...
3. ...while the biased ``v_average`` target lands measurably off at the same
   2-step rung (guards against silently re-biasing).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_flow_adapters.training.shortcut_targets import (
    compute_self_consistency_target_v,
    ddim_micro_step_v,
    invert_ddim_v,
)


def _linear_alphas_cumprod(timesteps: int = 1000) -> Tensor:
    betas = torch.linspace(8.5e-4, 1.2e-2, timesteps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


class PerfectRayModel(nn.Module):
    """Returns the true v-prediction of a fixed ray ``(x0, eps)`` at timestep t.

    On the VP circle ``x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·eps`` and the v-parameterised
    tangent is ``v_t = √ᾱ_t·eps − √(1−ᾱ_t)·x0``. The model ignores its ``x``
    input (it is *on-ray* by construction), so v1/v2 are the exact tangents.
    """

    def __init__(self, x0: Tensor, eps: Tensor, alphas_cumprod: Tensor) -> None:
        super().__init__()
        self.x0 = x0
        self.eps = eps
        self.ac = alphas_cumprod

    def forward(self, x: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        a = self.ac[t.long()].sqrt().view(-1, 1)
        s = (1.0 - self.ac[t.long()]).clamp_min(0.0).sqrt().view(-1, 1)
        return a * self.eps - s * self.x0


def _on_ray(x0: Tensor, eps: Tensor, ac: Tensor, t: int) -> Tensor:
    a = ac[t].sqrt()
    s = (1.0 - ac[t]).clamp_min(0.0).sqrt()
    return a * x0 + s * eps


def test_invert_ddim_v_round_trips_no_rescale() -> None:
    ac = _linear_alphas_cumprod()
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    v = torch.randn(4, 8, dtype=torch.float64)
    t = torch.tensor([800, 700, 950, 500])
    prev_t = t - 200

    x_end = ddim_micro_step_v(x=x, v=v, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=None)
    v_rec = invert_ddim_v(x=x, x_end=x_end, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=None)

    # recovers the velocity, and re-stepping reproduces the endpoint
    assert torch.allclose(v_rec, v, atol=1e-9), (v_rec - v).abs().max()
    x_again = ddim_micro_step_v(x=x, v=v_rec, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=None)
    assert torch.allclose(x_again, x_end, atol=1e-9), (x_again - x_end).abs().max()


def test_invert_ddim_v_round_trips_with_rescale() -> None:
    ac = _linear_alphas_cumprod()
    # Monotone dynamic-rescale table (mirrors DynamiCrafter's scale_arr usage).
    scale_arr = torch.linspace(1.0, 0.7, ac.shape[0], dtype=torch.float64)
    torch.manual_seed(1)
    x = torch.randn(4, 8, dtype=torch.float64)
    v = torch.randn(4, 8, dtype=torch.float64)
    t = torch.tensor([800, 700, 950, 500])
    prev_t = t - 200

    x_end = ddim_micro_step_v(x=x, v=v, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=scale_arr)
    v_rec = invert_ddim_v(x=x, x_end=x_end, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=scale_arr)

    assert torch.allclose(v_rec, v, atol=1e-9), (v_rec - v).abs().max()
    x_again = ddim_micro_step_v(x=x, v=v_rec, t=t, prev_t=prev_t, alphas_cumprod=ac, scale_arr=scale_arr)
    assert torch.allclose(x_again, x_end, atol=1e-9), (x_again - x_end).abs().max()


def test_endpoint_inversion_reproduces_two_step_landing() -> None:
    """One 2d step with the endpoint-inversion target hits the true landing;
    the v_average target does not (the bias the ticket is about)."""
    ac = _linear_alphas_cumprod()
    torch.manual_seed(2)
    x0 = torch.randn(4, 8, dtype=torch.float64)
    eps = torch.randn(4, 8, dtype=torch.float64)
    model = PerfectRayModel(x0, eps, ac)

    d = 200  # coarse half-step in timesteps → 2-step rung over [t, t-d, t-2d]
    t = torch.tensor([800, 800, 800, 800])
    x_t = _on_ray(x0, eps, ac, 800)
    prev2_t = t - 2 * d
    x_true_end = _on_ray(x0, eps, ac, 800 - 2 * d)  # the real two-step landing

    inv_target = compute_self_consistency_target_v(
        model=model, x_t=x_t, t=t, cond_half=None, d=d,
        alphas_cumprod=ac, scale_arr=None, target_kind="endpoint_inversion",
    )
    avg_target = compute_self_consistency_target_v(
        model=model, x_t=x_t, t=t, cond_half=None, d=d,
        alphas_cumprod=ac, scale_arr=None, target_kind="v_average",
    )

    # Endpoint inversion: a single 2d DDIM step lands on the true endpoint.
    x_land_inv = ddim_micro_step_v(
        x=x_t, v=inv_target, t=t, prev_t=prev2_t, alphas_cumprod=ac, scale_arr=None
    )
    err_inv = (x_land_inv - x_true_end).abs().max().item()
    assert err_inv < 1e-5, f"endpoint inversion should reproduce the landing, got {err_inv}"

    # v_average: the same 2d step lands measurably off (the bias).
    x_land_avg = ddim_micro_step_v(
        x=x_t, v=avg_target, t=t, prev_t=prev2_t, alphas_cumprod=ac, scale_arr=None
    )
    err_avg = (x_land_avg - x_true_end).abs().max().item()
    assert err_avg > 1e-3, f"v_average must fail the 2-step check, got {err_avg}"
    # and the inversion is orders of magnitude better
    assert err_avg > 100 * err_inv


def test_unknown_and_unimplemented_targets_raise() -> None:
    ac = _linear_alphas_cumprod()
    x0 = torch.randn(2, 8, dtype=torch.float64)
    eps = torch.randn(2, 8, dtype=torch.float64)
    model = PerfectRayModel(x0, eps, ac)
    t = torch.tensor([600, 600])
    x_t = _on_ray(x0, eps, ac, 600)

    import pytest

    with pytest.raises(NotImplementedError):
        compute_self_consistency_target_v(
            model=model, x_t=x_t, t=t, cond_half=None, d=100,
            alphas_cumprod=ac, scale_arr=None, target_kind="displacement",
        )
    with pytest.raises(ValueError):
        compute_self_consistency_target_v(
            model=model, x_t=x_t, t=t, cond_half=None, d=100,
            alphas_cumprod=ac, scale_arr=None, target_kind="bogus",
        )
