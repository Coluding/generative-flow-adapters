from __future__ import annotations

import torch
from torch import Tensor


def local_consistency_loss(shortcut_prediction: Tensor, one_step_target: Tensor) -> Tensor:
    return torch.mean((shortcut_prediction - one_step_target) ** 2)


def shortcut_direction_loss(shortcut_prediction: Tensor, shortcut_target: Tensor) -> Tensor:
    return torch.mean((shortcut_prediction - shortcut_target) ** 2)


def multistep_self_consistency_loss(prediction: Tensor, detached_target: Tensor) -> Tensor:
    return torch.mean((prediction - detached_target.detach()) ** 2)
