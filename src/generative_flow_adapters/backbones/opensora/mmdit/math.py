"""Math utilities for MMDiT attention and RoPE.

Vendored from Open-Sora (https://github.com/hpcaitech/Open-Sora)
Original source: opensora/models/mmdit/math.py
License: Apache License 2.0

Modified from Flux (Black Forest Labs):
    https://github.com/black-forest-labs/flux
    License: Apache License 2.0

Modifications:
    - Made flash_attn optional with PyTorch SDPA fallback
    - Made liger_kernel optional with standard RoPE fallback
"""

from __future__ import annotations

from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor

# Try to import flash attention, fall back to PyTorch SDPA
try:
    from flash_attn import flash_attn_func as flash_attn_func_v2

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func_v2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    SUPPORT_FA3 = True
except ImportError:
    SUPPORT_FA3 = False
    flash_attn_func_v3 = None

# Try to import liger kernel for optimized RoPE
try:
    from liger_kernel.ops.rope import LigerRopeFunction

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    LigerRopeFunction = None


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Flash attention with automatic version selection."""
    if SUPPORT_FA3 and flash_attn_func_v3 is not None:
        return flash_attn_func_v3(q, k, v)[0]
    if FLASH_ATTN_AVAILABLE and flash_attn_func_v2 is not None:
        return flash_attn_func_v2(q, k, v)
    # Fallback to PyTorch SDPA
    return _sdpa_attention(q, k, v)


def _sdpa_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """PyTorch scaled dot-product attention fallback.

    Expects inputs in (B, L, H, D) format.
    """
    # SDPA expects (B, H, L, D) format
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Back to (B, L, H, D)
    return out.transpose(1, 2)


def attention(q: Tensor, k: Tensor, v: Tensor, pe) -> Tensor:
    """Attention with RoPE position embeddings.

    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        v: Value tensor of shape (B, H, L, D)
        pe: Position embeddings - either a tensor or (cos, sin) tuple

    Returns:
        Attention output of shape (B, L, H*D)
    """
    if isinstance(pe, torch.Tensor):
        q, k = apply_rope(q, k, pe)
    else:
        cos, sin = pe
        if LIGER_AVAILABLE and LigerRopeFunction is not None:
            q, k = LigerRopeFunction.apply(q, k, cos, sin)
        else:
            # Fallback to standard RoPE application
            q, k = _apply_rope_from_cos_sin(q, k, cos, sin)

    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    x = flash_attn_func(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")

    return x


def _apply_rope_from_cos_sin(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply RoPE using precomputed cos/sin embeddings.

    Fallback implementation when liger_kernel is not available.
    """
    # cos, sin shapes: (B, L, D/2)
    # q, k shapes: (B, H, L, D)
    B, H, L, D = q.shape

    # Reshape for rotation
    q_r = q.reshape(B, H, L, D // 2, 2)
    k_r = k.reshape(B, H, L, D // 2, 2)

    # Expand cos/sin for heads dimension
    cos = cos.unsqueeze(1)  # (B, 1, L, D/2)
    sin = sin.unsqueeze(1)  # (B, 1, L, D/2)

    # Apply rotation
    q_out = torch.stack(
        [
            q_r[..., 0] * cos - q_r[..., 1] * sin,
            q_r[..., 1] * cos + q_r[..., 0] * sin,
        ],
        dim=-1,
    ).reshape(B, H, L, D)

    k_out = torch.stack(
        [
            k_r[..., 0] * cos - k_r[..., 1] * sin,
            k_r[..., 1] * cos + k_r[..., 0] * sin,
        ],
        dim=-1,
    ).reshape(B, H, L, D)

    return q_out.type_as(q), k_out.type_as(k)


def liger_rope(pos: Tensor, dim: int, theta: int) -> Tuple:
    """Compute RoPE embeddings in liger format (cos, sin).

    Args:
        pos: Position indices of shape (B, L)
        dim: Embedding dimension
        theta: RoPE theta parameter

    Returns:
        Tuple of (cos, sin) embeddings
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return (cos, sin)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Compute RoPE embeddings as rotation matrices.

    Args:
        pos: Position indices of shape (B, L)
        dim: Embedding dimension
        theta: RoPE theta parameter

    Returns:
        Rotation matrix embeddings of shape (B, L, dim/2, 2, 2)
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE using rotation matrix form.

    Args:
        xq: Query tensor of shape (B, H, L, D)
        xk: Key tensor of shape (B, H, L, D)
        freqs_cis: Rotation matrices of shape (B, L, D/2, 2, 2)

    Returns:
        Rotated (query, key) tensors
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rearrange_tensor(tensor: Tensor) -> Tensor:
    """Rearrange tensor dimensions for interleaved to split-half format.

    Mapping: 2d -> d, 2d+1 -> D/2 + d

    Args:
        tensor: Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor with rearranged last dimension, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the indices based on the mapping rule
    indices[:half_D] = torch.arange(0, D, 2, device=tensor.device)
    indices[half_D:] = torch.arange(1, D, 2, device=tensor.device)

    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor: Tensor) -> Tensor:
    """Restore original order from split-half to interleaved format.

    Mapping: d -> 2d, D/2 + d -> 2d + 1

    Args:
        tensor: Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor with restored last dimension order, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    reverse_indices = torch.empty(D, dtype=torch.long, device=tensor.device)

    # Fill the reverse indices to restore the original order
    reverse_indices[::2] = torch.arange(half_D, device=tensor.device)
    reverse_indices[1::2] = torch.arange(half_D, D, device=tensor.device)

    return tensor.index_select(dim=-1, index=reverse_indices)