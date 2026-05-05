"""Shortcut target computation for accelerated diffusion/flow matching training.

This module provides utilities for computing shortcut targets from a frozen base model,
which can be used to train adapter models to predict multi-step trajectories in a single step.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def attach_shortcut_targets_from_base(
    model: nn.Module,
    batch: Mapping[str, Tensor | object],
    *,
    step_size_key: str = "step_size",
    normalize_base_direction: bool = True,
    method: str = "linear",
) -> dict[str, Tensor | object]:
    """Attach deterministic shortcut supervision derived from frozen base predictions.

    Args:
        model: The model (or adapted model) to extract base model from
        batch: Training batch containing x_t, t, cond
        step_size_key: Key in cond dict for step size (default "step_size")
        normalize_base_direction: Whether to normalize 2D outputs (default True)
        method: Target construction method:
            - "linear": shortcut_target = base_output * step_size (fast, 1 forward pass)
            - "two_step": Two-step averaging as in official shortcut-models (accurate, 2 forward passes)

    Returns:
        Updated batch with shortcut_target and self_consistency_target added
    """
    x_t = _as_tensor(batch, "x_t")
    t = _as_tensor(batch, "t")
    cond = batch.get("cond")

    base_model = getattr(model, "base_model", model)
    step_size = _resolve_step_size(cond, step_size_key=step_size_key, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

    if method.lower() == "linear":
        shortcut_target = _compute_linear_shortcut_target(
            base_model=base_model,
            x_t=x_t,
            t=t,
            cond=cond,
            step_size=step_size,
            normalize_base_direction=normalize_base_direction,
        )
    elif method.lower() == "two_step":
        shortcut_target = _compute_two_step_shortcut_target(
            base_model=base_model,
            x_t=x_t,
            t=t,
            cond=cond,
            step_size=step_size,
        )
    else:
        raise ValueError(f"Unknown shortcut target method: {method}. Use 'linear' or 'two_step'.")

    updated_batch = dict(batch)
    updated_batch["shortcut_target"] = shortcut_target.detach()
    updated_batch["self_consistency_target"] = shortcut_target.detach()
    return updated_batch


def _compute_linear_shortcut_target(
    base_model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond: object | None,
    step_size: Tensor,
    normalize_base_direction: bool,
) -> Tensor:
    """Linear scaling method: target = base_output * step_size.

    This is a fast approximation that requires only a single forward pass through
    the base model. Optionally normalizes the direction for 2D outputs.
    """
    with torch.no_grad():
        base_output = base_model(x_t, t, cond=cond)

    base_direction = base_output
    if normalize_base_direction and base_direction.dim() == 2:
        base_direction = base_direction / base_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    step_size = _reshape_step_size_for_base(step_size=step_size, base_direction=base_direction)
    return base_direction * step_size


def _compute_two_step_shortcut_target(
    base_model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    cond: object | None,
    step_size: Tensor,
) -> Tensor:
    """Two-step averaging method as in official shortcut-models paper.

    Process:
        1. v_b1 = model(x_t, t, cond)
        2. x_t2 = x_t + step_size * v_b1
        3. v_b2 = model(x_t2, t, cond)  # Use same t (simplified)
        4. target = (v_b1 + v_b2) / 2

    This method is more accurate but requires two forward passes through the base model.
    """
    with torch.no_grad():
        # First prediction
        v_b1 = base_model(x_t, t, cond=cond)

        # Reshape step_size for broadcasting
        step_size_reshaped = _reshape_step_size_for_base(step_size=step_size, base_direction=v_b1)

        # Step forward
        x_t2 = x_t + step_size_reshaped * v_b1

        # Second prediction (at stepped-forward state)
        v_b2 = base_model(x_t2, t, cond=cond)

        # Average the two predictions
        return (v_b1 + v_b2) / 2.0


def _resolve_step_size(
    cond: object | None,
    *,
    step_size_key: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Extract step size from conditioning or return default ones."""
    if isinstance(cond, Mapping):
        step_size = cond.get(step_size_key)
        if step_size is None and step_size_key != "horizon":
            step_size = cond.get("horizon")
        if isinstance(step_size, Tensor):
            return step_size.to(device=device, dtype=dtype)
    return torch.ones(batch_size, device=device, dtype=dtype)


def _as_tensor(batch: Mapping[str, Tensor | object], key: str) -> Tensor:
    """Extract tensor from batch and validate type."""
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch['{key}'] must be a tensor.")
    return value


def _reshape_step_size_for_base(step_size: Tensor, base_direction: Tensor) -> Tensor:
    """Reshape step_size tensor to match base_direction dimensions for broadcasting."""
    if base_direction.dim() == 2:
        if step_size.dim() == 1:
            return step_size.unsqueeze(-1)
        if step_size.dim() == 2 and step_size.shape[-1] == 1:
            return step_size
    if base_direction.dim() == 5:
        if step_size.dim() == 1:
            return step_size[:, None, None, None, None]
        if step_size.dim() == 2:
            return step_size[:, None, :, None, None]
        if step_size.dim() == 3 and step_size.shape[-1] == 1:
            return step_size.permute(0, 2, 1)[:, :, :, None, None]

    while step_size.dim() < base_direction.dim():
        step_size = step_size.unsqueeze(-1)
    return step_size
