"""Shortcut target computation for accelerated diffusion training.

Pure functions used by :class:`~generative_flow_adapters.training.trainer.Trainer`
to build self-supervised targets for shortcut adapters. All dependencies
(schedule tables, models) are passed in explicitly so these helpers stay
testable without spinning up a trainer.

One target family lives here:

- :func:`compute_self_consistency_target_v` — self-consistency target. Two
  no-grad calls of the *adapted* model at the half step, chained across one
  ``d``-sized DDIM micro-step. The adapter is its own teacher; the frozen base
  contributes only implicitly via the composition inside the model. Two target
  parametrisations are selectable via ``target_kind``:

  - ``"v_average"`` — ``(v1+v2)/2`` (Frans et al. 2024, eq. 4). This is exact
    for **flow matching** (straight path, constant velocity), but **biased for
    diffusion v-prediction**: the DDIM step is a rotation on the noise-angle
    circle, so ``v1`` and ``v2`` are tangents at *different* points and their
    ambient average lands a sagitta (``1−cos δ``) off the true endpoint —
    5–24 % in the few-step regime. The true field is not a fixed point of this
    rule, so more training cannot remove it. Kept as the ablation baseline.
  - ``"endpoint_inversion"`` — follow *both* sub-steps to the real landing
    ``x_end`` and invert the single ``2d`` DDIM step for the velocity that
    reproduces it (:func:`invert_ddim_v`). Exact under v-prediction: the true
    field *is* a fixed point. Target-only change — keeps the velocity head and
    the DDIM sampler.

  Background + derivation: thesis-vault theory/shortcut-v-averaging-bias and
  theory/ddim-step-v-parameterisation §8; decision
  decided/shortcut-target-endpoint-vs-v-averaging.

The former ``compute_two_step_target_v`` (base-anchored Heun 2-step target)
was removed when the ``two_step`` shortcut mode was deprecated — see
thesis-vault decided/deprecate-twostep-shortcut-mode. Its Heun construction
now lives on as an orthogonal regularizer (``heun_smoothness_weight``), built
directly from :func:`ddim_micro_step_v` in the trainer.

:func:`ddim_micro_step_v` is the underlying single-step DDIM update for
v-prediction; it matches
:meth:`DiffusionInferenceSampler._dynamic_rescale_ddim_step` semantically but
takes raw schedule tables and per-sample timestep tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def compute_self_consistency_target_v(
    *,
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond_half: object | None,
    d: int,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
    target_kind: str = "v_average",
) -> Tensor:
    """Self-consistency target for diffusion v-prediction.

    Two no-grad calls of the adapted model at step ``d`` chained across one DDIM
    micro-step. ``target_kind`` selects the parametrisation (see module
    docstring): ``"v_average"`` (biased baseline) or ``"endpoint_inversion"``
    (exact). ``d`` is the half-step in timestep units, so the full step lands at
    ``t − 2d``.
    """
    kind = str(target_kind).lower()
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

            if kind in ("v_average", "average"):
                return ((v1 + v2) / 2.0).detach()

            if kind == "endpoint_inversion":
                # Take the SECOND sub-step to the real landing, then invert the
                # single 2d DDIM step for the velocity that reproduces it.
                prev2_t = (t - 2 * d).clamp_min(0)
                x_end = ddim_micro_step_v(
                    x=x_mid, v=v2, t=prev_t, prev_t=prev2_t,
                    alphas_cumprod=alphas_cumprod, scale_arr=scale_arr,
                )
                v_target = invert_ddim_v(
                    x=x_t, x_end=x_end, t=t, prev_t=prev2_t,
                    alphas_cumprod=alphas_cumprod, scale_arr=scale_arr,
                )
                return v_target.detach()

            if kind == "displacement":
                raise NotImplementedError(
                    "shortcut_consistency_target='displacement' (Option B) needs "
                    "the additive few-step sampler + d→0 head grounding, not yet "
                    "wired — see thesis-vault feat-shortcut-displacement-target-"
                    "parametrisation. Use 'endpoint_inversion' or 'v_average'."
                )

            raise ValueError(
                f"Unknown shortcut_consistency_target={target_kind!r}; "
                "expected 'v_average' or 'endpoint_inversion'."
            )
    finally:
        model.train(was_training)


def flow_micro_step_v(*, x: Tensor, v: Tensor, d: Tensor | float) -> Tensor:
    """Rectified-flow Euler micro-step in the generative direction (toward data).

    With the linear path ``x_t = (1-t)*x0 + t*noise`` and velocity
    ``v = dx/dt = noise - x0``, one generative step of size ``d`` (decreasing
    ``t`` toward data at ``t=0``) is ``x_{t-d} = x_t - d * v``. ``d`` is the
    normalised step in ``t``-units (``t in [0, 1]``). This is the flow analogue
    of :func:`ddim_micro_step_v` — no ``alphas_cumprod``, just a straight-line
    Euler step, which is exactly correct for rectified flow.
    """
    d_view = d.view(-1, *[1] * (x.dim() - 1)) if isinstance(d, Tensor) else d
    return x - d_view * v


def compute_self_consistency_target_v_flow(
    *,
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond_half: object | None,
    d: Tensor,
    timestep_scale: float = 1.0,
) -> Tensor:
    """Flow-native self-consistency target (shortcut models, Frans et al. 2024).

    The 2d-step velocity equals the average of two d-step velocities chained
    across one flow Euler micro-step — the rectified-flow counterpart of
    :func:`compute_self_consistency_target_v` (which uses a DDIM micro-step and
    is only valid for the diffusion parameterisation).

    ``d`` is the step in **sigma** units (∈[0,1]); the spatial micro-step is
    ``x - d·v``. The model is fed ``t = sigma · timestep_scale``, so the second
    call's timestep moves by ``d · timestep_scale`` (not ``d``).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            v1 = model(x_t, t, cond_half)
            x_mid = flow_micro_step_v(x=x_t, v=v1, d=d)
            t_next = (t - d * timestep_scale).clamp_min(0.0)
            v2 = model(x_mid, t_next, cond_half)
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
    # this gives us the prediction at timestep t- s --> We  determine x_0 by using the velocity pred
    # --> Then use this rpedicted x_0 as our stsarting point and add predicted noise to
    # --> The preditec noise is scaled by the imestep of t-s (using the alpha and sigma scalings)
    # --> This gives us a predction of the previous timestep using our current prediction
    # CAVEAT: The previosu timestep is a denoising timestep, so we are just doing the simple denoising step
    return sqrt_alpha_prev * pred_x0 + sqrt_sigma_prev * pred_eps


def invert_ddim_v(
    *,
    x: Tensor,
    x_end: Tensor,
    t: Tensor,
    prev_t: Tensor,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
) -> Tensor:
    """Velocity ``v`` such that ``ddim_micro_step_v(x, v, t, prev_t) == x_end``.

    The DDIM v-step is **affine in v** — write the forward map (mirroring
    :func:`ddim_micro_step_v`, incl. the ``scale_arr`` data-side rescale ``r``)
    as ``x_end = A·x + B·v`` with::

        a_t, s_t = √ᾱ_t,    √(1−ᾱ_t)        a_p, s_p = √ᾱ_prev, √(1−ᾱ_prev)
        r        = scale_prev / scale_t  (1 if scale_arr is None)
        A        = a_p·r·a_t + s_p·s_t
        B        = s_p·a_t   − a_p·r·s_t

    so the inverse is ``v = (x_end − A·x) / B``. ``B`` is ``sin`` of the
    swept noise-angle (× rescale), nonzero for any real step; it only collapses
    when ``prev_t == t`` (fully clamped, no step), where ``x_end == x`` and the
    velocity is irrelevant — handled by clamping the denominator.

    This is the velocity that *reproduces* ``x_end`` under the DDIM rotation —
    **not** the chord ``x_end − x`` (that is Option B's displacement target,
    which pairs with additive stepping). Derivation:
    thesis-vault theory/ddim-step-v-parameterisation §8.
    """
    def _gather(table: Tensor, indices: Tensor) -> Tensor:
        idx = indices.to(dtype=torch.long, device=table.device).clamp(min=0, max=table.shape[0] - 1)
        out = table.index_select(0, idx)
        return out.view(-1, *[1] * (x.dim() - 1)).to(device=x.device, dtype=x.dtype)

    alpha_t    = _gather(alphas_cumprod, t)
    alpha_prev = _gather(alphas_cumprod, prev_t)
    a_t = alpha_t.sqrt()
    s_t = (1.0 - alpha_t).clamp_min(0.0).sqrt()
    a_p = alpha_prev.sqrt()
    s_p = (1.0 - alpha_prev).clamp_min(0.0).sqrt()

    r = torch.ones_like(a_t)
    if scale_arr is not None:
        cur_scale  = _gather(scale_arr, t)
        prev_scale = _gather(scale_arr, prev_t)
        r = prev_scale / cur_scale

    A = a_p * r * a_t + s_p * s_t
    B = s_p * a_t - a_p * r * s_t
    # Guard the degenerate no-step case (prev_t == t ⇒ B == 0); the perturbation
    # is far below any real 2d-step's |B|, so the exactness check still passes.
    B_safe = torch.where(B.abs() < 1e-8, torch.full_like(B, 1e-8), B)
    return (x_end - A * x) / B_safe
