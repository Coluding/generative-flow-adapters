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
) -> dict[str, Tensor | object]:
    """Attach deterministic shortcut supervision derived from frozen base predictions."""
    x_t = _as_tensor(batch, "x_t")
    t = _as_tensor(batch, "t")
    cond = batch.get("cond")

    base_model = getattr(model, "base_model", model)
    with torch.no_grad():
        base_output = base_model(x_t, t, cond=cond)

    base_direction = base_output
    if normalize_base_direction and base_direction.dim() == 2:
        base_direction = base_direction / base_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    step_size = _resolve_step_size(cond, step_size_key=step_size_key, batch_size=x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
    step_size = _reshape_step_size_for_base(step_size=step_size, base_direction=base_direction)
    shortcut_target = base_direction * step_size

    updated_batch = dict(batch)
    updated_batch["shortcut_target"] = shortcut_target.detach()
    updated_batch["self_consistency_target"] = shortcut_target.detach()
    return updated_batch


def _resolve_step_size(
    cond: object | None,
    *,
    step_size_key: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(cond, Mapping):
        step_size = cond.get(step_size_key)
        if step_size is None and step_size_key != "horizon":
            step_size = cond.get("horizon")
        if isinstance(step_size, Tensor):
            return step_size.to(device=device, dtype=dtype)
    return torch.ones(batch_size, device=device, dtype=dtype)


def _as_tensor(batch: Mapping[str, Tensor | object], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch['{key}'] must be a tensor.")
    return value


def _reshape_step_size_for_base(step_size: Tensor, base_direction: Tensor) -> Tensor:
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
