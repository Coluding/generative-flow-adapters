from __future__ import annotations

import torch
from torch import Tensor


def flow_matching_loss(prediction: Tensor, target_velocity: Tensor) -> Tensor:
    return torch.mean((prediction - target_velocity) ** 2)
