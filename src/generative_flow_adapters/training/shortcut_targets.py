"""Shortcut target computation for accelerated diffusion training.

Pure functions used by :class:`~generative_flow_adapters.training.trainer.Trainer`
to build self-supervised targets for shortcut adapters. All dependencies
(schedule tables, models) are passed in explicitly so these helpers stay
testable without spinning up a trainer.

Two target families live here:

- :func:`compute_two_step_target_v` — base-anchored Heun-corrected 2-step
  target for v-prediction. One proper DDIM micro-step under the frozen base,
  velocities averaged. No collapse risk; ``step_level`` on the adapter is
  decorative.

- :func:`compute_self_consistency_target_v` — paper-faithful self-consistency
  target (Frans et al. 2024, eq. 4). Two no-grad calls of the *adapted* model
  at the half step, chained across one ``d``-sized DDIM micro-step. The
  adapter is its own teacher; the frozen base contributes only implicitly via
  the composition inside the model.

:func:`ddim_micro_step_v` is the underlying single-step DDIM update for
v-prediction; it matches
:meth:`DiffusionInferenceSampler._dynamic_rescale_ddim_step` semantically but
takes raw schedule tables and per-sample timestep tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def compute_two_step_target_v(
    *,
    base_model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond: object | None,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
) -> Tensor:
    """Base-anchored Heun-corrected 2-step target for v-prediction."""
    with torch.no_grad():
        v0 = base_model(x_t, t, cond=cond)
        prev_t = (t - 1).clamp_min(0)
        x_mid = ddim_micro_step_v(
            x=x_t, v=v0, t=t, prev_t=prev_t,
            alphas_cumprod=alphas_cumprod, scale_arr=scale_arr,
        )
        v1 = base_model(x_mid, prev_t, cond=cond)
    return ((v0 + v1) / 2.0).detach()


def compute_self_consistency_target_v(
    *,
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond_half: object | None,
    d: int,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
) -> Tensor:
    """Paper-faithful self-consistency target (Frans et al. 2024, eq. 4)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            v1 = model(x_t, t, cond_half)
            prev_t = (t - d).clamp_min(0)
            x_mid = ddim_micro_step_v(
                x=x_t, v=v1, t=t, prev_t=prev_t,
                alphas_cumprod=alphas_cumprod, scale_arr=scale_arr,
            )
            v2 = model(x_mid, prev_t, cond_half)
    finally:
        model.train(was_training)
    return ((v1 + v2) / 2.0).detach()


def ddim_micro_step_v(
    *,
    x: Tensor,
    v: Tensor,
    t: Tensor,
    prev_t: Tensor,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
) -> Tensor:
    """Single-timestep DDIM update for v-prediction, per-sample t/prev_t.

    With ``scale_arr`` provided, applies DynamiCrafter's data-side SNR rescale
    (``pred_x0 *= prev_scale / cur_scale``) between v-decode and recomposition.

    Args:
        x: ``(B, ..., d)`` sample at timestep ``t``.
        v: ``(B, ..., d)`` v-prediction at ``(x, t)`` (same shape as ``x``).
        t: ``(B,)`` long tensor of current timesteps.
        prev_t: ``(B,)`` long tensor of target timesteps (``< t``, clamped at 0).
        alphas_cumprod: ``(T_train,)`` lookup table from the training schedule.
        scale_arr: ``(T_train,)`` dynamic-rescale table, or ``None`` for no rescale.

    Returns:
        ``(B, ..., d)`` sample at timestep ``prev_t``.
    """
    def _gather(table: Tensor, indices: Tensor) -> Tensor:
        idx = indices.to(dtype=torch.long, device=table.device).clamp(min=0, max=table.shape[0] - 1)
        out = table.index_select(0, idx)
        return out.view(-1, *[1] * (x.dim() - 1)).to(device=x.device, dtype=x.dtype)

    alpha_t    = _gather(alphas_cumprod, t)
    alpha_prev = _gather(alphas_cumprod, prev_t)
    sqrt_alpha_t    = alpha_t.sqrt()
    sqrt_sigma_t    = (1.0 - alpha_t).clamp_min(0.0).sqrt()
    sqrt_alpha_prev = alpha_prev.sqrt()
    sqrt_sigma_prev = (1.0 - alpha_prev).clamp_min(0.0).sqrt()

    pred_x0  = sqrt_alpha_t * x - sqrt_sigma_t * v
    pred_eps = sqrt_alpha_t * v + sqrt_sigma_t * x

    if scale_arr is not None:
        cur_scale  = _gather(scale_arr, t)
        prev_scale = _gather(scale_arr, prev_t)
        pred_x0 = pred_x0 * (prev_scale / cur_scale)

    return sqrt_alpha_prev * pred_x0 + sqrt_sigma_prev * pred_eps
