"""Common utilities for Open-Sora model integration.

This module provides utilities for:
- Packing/unpacking video latents to/from sequence format
- Creating position IDs for image and text tokens
- Flow matching schedule utilities
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def pack_latents(x: Tensor, patch_size: int = 2) -> Tensor:
    """Pack video latents into a sequence of patch tokens.

    Converts spatial video latents to a flat sequence suitable for transformer processing.
    Each patch_size x patch_size spatial region becomes a single token.

    Args:
        x: Video latents of shape (B, C, T, H, W)
        patch_size: Size of spatial patches (default 2)

    Returns:
        Packed tokens of shape (B, T*H*W/patch_size^2, C*patch_size^2)

    Example:
        >>> x = torch.randn(2, 64, 17, 32, 32)  # 2 videos, 64 channels, 17 frames, 32x32
        >>> packed = pack_latents(x, patch_size=2)
        >>> packed.shape
        torch.Size([2, 4352, 256])  # 17 * 16 * 16 = 4352 tokens, 64 * 4 = 256 dims
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {x.dim()}D")

    return rearrange(
        x,
        "b c t (h ph) (w pw) -> b (t h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )


def unpack_latents(
    x: Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 2,
) -> Tensor:
    """Unpack sequence tokens back to video latent format.

    Inverse of pack_latents. Converts flat token sequence back to spatial video format.

    Args:
        x: Packed tokens of shape (B, seq_len, C*patch_size^2)
        num_frames: Number of temporal frames
        height: Spatial height (before packing, i.e., H/patch_size)
        width: Spatial width (before packing, i.e., W/patch_size)
        patch_size: Size of spatial patches (default 2)

    Returns:
        Video latents of shape (B, C, T, H*patch_size, W*patch_size)
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, seq_len, hidden), got {x.dim()}D")

    return rearrange(
        x,
        "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
        t=num_frames,
        h=height,
        w=width,
        ph=patch_size,
        pw=patch_size,
    )


def create_image_ids(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create position IDs for image/video tokens.

    MMDiT uses 3D factorized position IDs (t, h, w) for RoPE embeddings.

    Args:
        batch_size: Batch size
        num_frames: Number of temporal frames
        height: Spatial height (in patch units, i.e., H/patch_size)
        width: Spatial width (in patch units, i.e., W/patch_size)
        device: Target device
        dtype: Data type for position IDs

    Returns:
        Position IDs of shape (B, T*H*W, 3) where last dim is (t, h, w)
    """
    # Create coordinate grids
    t_coords = torch.arange(num_frames, device=device, dtype=dtype)
    h_coords = torch.arange(height, device=device, dtype=dtype)
    w_coords = torch.arange(width, device=device, dtype=dtype)

    # Create meshgrid and flatten
    t_grid, h_grid, w_grid = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    ids = torch.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1)

    # Expand for batch
    ids = ids.unsqueeze(0).expand(batch_size, -1, -1)
    return ids


def create_text_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create position IDs for text tokens.

    Text tokens use (0, 0, position) format to differentiate from image tokens.

    Args:
        batch_size: Batch size
        seq_len: Text sequence length
        device: Target device
        dtype: Data type for position IDs

    Returns:
        Position IDs of shape (B, seq_len, 3) where last dim is (0, 0, position)
    """
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)
    ids[:, 2] = positions  # Only use third axis for text positions

    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def get_schedule_shift(
    height: int,
    width: int,
    num_frames: int,
    patch_size: int = 2,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute the schedule shift (alpha) for flow matching.

    Open-Sora uses resolution and frame-dependent schedule shifting similar to SD3.
    Higher resolutions and more frames get larger shifts.

    Args:
        height: Latent height
        width: Latent width
        num_frames: Number of frames
        patch_size: Patch size used
        base_shift: Base shift value
        max_shift: Maximum shift value

    Returns:
        Schedule shift value (alpha)
    """
    # Compute number of image tokens
    num_tokens = (height // patch_size) * (width // patch_size) * num_frames

    # Linear interpolation based on token count
    # This follows the SD3/Flux approach
    log_tokens = math.log(num_tokens)
    log_min = math.log(256)  # ~16x16 single frame
    log_max = math.log(4096 * 32)  # ~64x64 * 32 frames

    t = (log_tokens - log_min) / (log_max - log_min)
    t = max(0.0, min(1.0, t))

    return base_shift + t * (max_shift - base_shift)


def time_shift(alpha: float, t: Tensor) -> Tensor:
    """Apply schedule shift to timesteps.

    Implements the time shift from SD3: t' = alpha * t / (1 + (alpha - 1) * t)

    This shifts emphasis toward higher timesteps for higher resolutions,
    giving the model more steps for denoising complex content.

    Args:
        alpha: Schedule shift value (from get_schedule_shift)
        t: Timesteps in [0, 1] range

    Returns:
        Shifted timesteps
    """
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shift_alpha: float | None = None,
) -> Tensor:
    """Sample timesteps for flow matching training.

    Uses logit-normal sampling as in SD3/Flux, with optional schedule shifting.

    Args:
        batch_size: Number of timesteps to sample
        device: Target device
        dtype: Data type
        shift_alpha: Optional schedule shift (from get_schedule_shift)

    Returns:
        Timesteps in [0, 1] range of shape (B,)
    """
    # Logit-normal sampling (SD3 style)
    u = torch.randn(batch_size, device=device, dtype=dtype)
    t = torch.sigmoid(u)

    # Apply schedule shift if provided
    if shift_alpha is not None:
        t = time_shift(shift_alpha, t)

    return t


def compute_flow_target(
    x_0: Tensor,
    x_1: Tensor,
    sigma_min: float = 1e-5,
) -> Tensor:
    """Compute the velocity target for flow matching.

    In flow matching, the model predicts the velocity field v_t that transports
    the noise distribution to the data distribution.

    For linear interpolation: x_t = (1-t) * x_0 + t * x_1
    The velocity is: v_t = x_1 - x_0 (constant along the path)

    With sigma_min adjustment: v_t = (1 - sigma_min) * x_1 - x_0

    Args:
        x_0: Clean data
        x_1: Noise
        sigma_min: Minimum noise level (prevents exact zero)

    Returns:
        Target velocity
    """
    return (1 - sigma_min) * x_1 - x_0


def interpolate_flow(
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
    sigma_min: float = 1e-5,
) -> Tensor:
    """Compute the noisy sample at timestep t for flow matching.

    Linear interpolation between data and noise:
    x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1

    Note: This is the REVERSE of diffusion convention.
    - t=0 is clean data
    - t=1 is noise

    Args:
        x_0: Clean data of shape (B, C, T, H, W)
        x_1: Noise of shape (B, C, T, H, W)
        t: Timesteps of shape (B,) in range [0, 1]
        sigma_min: Minimum noise level

    Returns:
        Noisy samples at timestep t
    """
    # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1, 1)
    while t.dim() < x_0.dim():
        t = t.unsqueeze(-1)

    t_scaled = (1 - sigma_min) * t
    return (1 - t_scaled) * x_0 + t * x_1