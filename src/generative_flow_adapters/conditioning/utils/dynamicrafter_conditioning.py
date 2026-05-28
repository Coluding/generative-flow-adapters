from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def prepare_dynamicrafter_condition(
    cond: object | None,
    *,
    x_t: Tensor,
    use_step_level_conditioning: bool = False,
    step_level_key: str = "step_level",
    step_level_embed: nn.Module | None = None,
    step_level_transform: str = "linear",
) -> object | None:
    if not isinstance(cond, Mapping):
        return cond

    runtime_cond = dict(cond)
    adapter_embedding = runtime_cond.get("embedding")

    if use_step_level_conditioning:
        step_level = runtime_cond.get(step_level_key)
        step_level_embedding = encode_step_level_embedding(
            step_level=step_level,
            step_level_embed=step_level_embed,
            device=x_t.device,
            dtype=x_t.dtype,
            transform=step_level_transform,
        )
        runtime_cond["embedding"] = combine_adapter_embeddings(adapter_embedding, step_level_embedding)

    return runtime_cond


def _apply_step_level_transform(step_level: Tensor, transform: str) -> Tensor:
    """Map the raw step-size scalar to the value fed into ``step_level_embed``.

    ``"linear"`` passes it through (fine when the value is a bounded normalised
    ``s ∈ (0,1]``); ``"log2"`` feeds ``log2(s)``, which spreads the fine end of
    a dyadic schedule into a well-conditioned range (e.g. ``[-7, 0]`` for
    ``s ∈ [1/128, 1]``) instead of crushing it near zero. Both keep the 1-D
    shape the ``Linear(1, ·)`` embed expects."""
    key = transform.lower()
    if key in {"linear", "raw", "none"}:
        return step_level
    if key == "log2":
        return torch.log2(step_level.clamp_min(1e-8))
    raise ValueError(f"Unknown step_level_transform={transform!r}; expected 'linear' or 'log2'.")


def encode_step_level_embedding(
    step_level: object | None,
    *,
    step_level_embed: nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
    transform: str = "linear",
) -> Tensor | None:
    if step_level_embed is None or not isinstance(step_level, Tensor):
        return None
    step_level = _apply_step_level_transform(step_level.to(device=device, dtype=dtype), transform)
    if step_level.dim() == 1:
        return step_level_embed(step_level.unsqueeze(-1))
    if step_level.dim() == 2:
        if step_level.shape[-1] == 1:
            return step_level_embed(step_level)
        batch, frames = step_level.shape
        embedded = step_level_embed(step_level.reshape(batch * frames, 1))
        return embedded.reshape(batch, frames, -1)
    if step_level.dim() == 3 and step_level.shape[-1] == 1:
        batch, frames = step_level.shape[0], step_level.shape[1]
        embedded = step_level_embed(step_level.reshape(batch * frames, 1))
        return embedded.reshape(batch, frames, -1)
    raise ValueError("step_level must have shape [batch], [batch, 1], [batch, frames], or [batch, frames, 1].")


def combine_adapter_embeddings(first: Tensor | None, second: Tensor | None) -> Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    if first.dim() == second.dim():
        if first.shape != second.shape:
            raise ValueError("Cannot combine embeddings with mismatched shapes.")
        return first + second
    if first.dim() == 2 and second.dim() == 3:
        return first.unsqueeze(1).expand(-1, second.shape[1], -1) + second
    if first.dim() == 3 and second.dim() == 2:
        return first + second.unsqueeze(1).expand(-1, first.shape[1], -1)
    raise ValueError("Unsupported embedding ranks for combination.")
