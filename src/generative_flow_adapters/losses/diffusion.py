from __future__ import annotations

import torch
from torch import Tensor


def diffusion_loss(prediction: Tensor, target_noise: Tensor) -> Tensor:
    return torch.mean((prediction - target_noise) ** 2)
