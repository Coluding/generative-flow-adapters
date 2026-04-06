from __future__ import annotations

import math

import torch
from torch import Tensor


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.float().view(-1)
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1))
    args = timesteps[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
