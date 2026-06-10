from __future__ import annotations

import torch
from torch import Tensor


def local_consistency_loss(shortcut_prediction: Tensor, one_step_target: Tensor) -> Tensor:
    return torch.mean((shortcut_prediction - one_step_target) ** 2)


def shortcut_direction_loss(shortcut_prediction: Tensor, shortcut_target: Tensor) -> Tensor:
    return torch.mean((shortcut_prediction - shortcut_target) ** 2)


def multistep_self_consistency_loss(prediction: Tensor, detached_target: Tensor) -> Tensor:
    return torch.mean((prediction - detached_target.detach()) ** 2)


def heun_smoothness_loss(current_v: Tensor, future_v_detached: Tensor) -> Tensor:
    """Discrete material-derivative penalty along the predicted ODE trajectory.

    ``L = ||v0 - sg(v1)||^2`` where ``v0`` is the velocity at the current point
    and ``v1`` is the model's velocity at the one-step-ahead predicted endpoint,
    used as a stop-gradient reference. Pushes the composed velocity field toward
    local smoothness along its own trajectory, reducing the per-step integration
    error of a fixed-step sampler. See thesis-vault
    theory/heun-smoothness-regularizer.md eq. (S).
    """
    return torch.mean((current_v - future_v_detached.detach()) ** 2)
