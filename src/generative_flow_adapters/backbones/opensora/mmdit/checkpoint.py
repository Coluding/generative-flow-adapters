"""Gradient checkpointing utilities for MMDiT.

Vendored from Open-Sora (https://github.com/hpcaitech/Open-Sora)
Original source: opensora/acceleration/checkpoint.py
License: Apache License 2.0

Modifications:
    - Removed colossalai dependency
    - Simplified to use PyTorch's native checkpointing
    - Removed activation offloading (requires colossalai)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def set_grad_checkpoint(
    model: nn.Module,
    use_fp32_attention: bool = False,
    gc_step: int = 1,
) -> None:
    """Enable gradient checkpointing on a model.

    Args:
        model: The model to enable checkpointing on
        use_fp32_attention: Whether to use FP32 for attention
        gc_step: Gradient checkpointing step size for sequential
    """

    def set_attr(module: nn.Module) -> None:
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module: nn.Module | Iterable, *args: Any, **kwargs: Any) -> Any:
    """Automatically apply gradient checkpointing if enabled on module.

    Args:
        module: Module or iterable of modules to checkpoint
        *args: Arguments to pass to module
        **kwargs: Keyword arguments to pass to module

    Returns:
        Output of module forward pass
    """
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=True, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)
