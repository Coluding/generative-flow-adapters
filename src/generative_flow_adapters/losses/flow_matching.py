from __future__ import annotations

import math

import torch
from torch import Tensor


def flow_matching_loss(prediction: Tensor, target_velocity: Tensor) -> Tensor:
    """Compute flow matching loss (MSE between predicted and target velocity)."""
    return torch.mean((prediction - target_velocity) ** 2)


class FlowMatchingTrainingObjective:
    """Training objective for flow matching models like Open-Sora.

    Flow matching differs from diffusion in several key ways:
    - Uses velocity prediction instead of noise prediction
    - Timesteps are in [0, 1] range (not integer indices)
    - Uses linear interpolation with optional schedule shifting
    - t=0 is clean data, t=1 is noise (opposite of diffusion convention)
    """

    def __init__(
        self,
        sigma_min: float = 1e-5,
        shift_schedule: bool = True,
        base_shift: float = 1.0,
        max_shift: float = 3.0,
        shift_x1: float = 256.0,
        shift_x2: float = 4096.0,
        temporal_sqrt_scaling: bool = True,
    ) -> None:
        """Initialize flow matching objective.

        Args:
            sigma_min: Minimum noise level to avoid exact zeros
            shift_schedule: Whether to apply resolution-dependent schedule shifting
            base_shift: Base shift value for schedule
            max_shift: Maximum shift value for schedule
            shift_x1: First spatial anchor for linear shift interpolation
            shift_x2: Second spatial anchor for linear shift interpolation
            temporal_sqrt_scaling: Whether to scale shift by sqrt(num_frames)
        """
        self.sigma_min = sigma_min
        self.shift_schedule = shift_schedule
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.shift_x1 = shift_x1
        self.shift_x2 = shift_x2
        self.temporal_sqrt_scaling = temporal_sqrt_scaling

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        patch_size: int = 2,
    ) -> Tensor:
        """Sample timesteps using logit-normal distribution with optional shifting.

        Args:
            batch_size: Number of timesteps to sample
            device: Target device
            dtype: Data type
            height: Latent height (for schedule shifting)
            width: Latent width (for schedule shifting)
            num_frames: Number of frames (for schedule shifting)
            patch_size: Patch size (for schedule shifting)

        Returns:
            Timesteps in [0, 1] range
        """
        # Logit-normal sampling (SD3/Flux style)
        u = torch.randn(batch_size, device=device, dtype=torch.float32)
        t = torch.sigmoid(u)

        # Apply schedule shift if enabled and dimensions provided
        if self.shift_schedule and all(d is not None for d in [height, width, num_frames]):
            shift_alpha = self._compute_shift(height, width, num_frames, patch_size)
            t = self._apply_shift(shift_alpha, t)

        return t.to(dtype)

    def _compute_shift(
        self,
        height: int,
        width: int,
        num_frames: int,
        patch_size: int,
    ) -> float:
        """Compute schedule shift following Open-Sora's spatial+temporal scaling."""
        image_seq_len = (height // patch_size) * (width // patch_size)
        m = (self.max_shift - self.base_shift) / (self.shift_x2 - self.shift_x1)
        b = self.base_shift - m * self.shift_x1
        shift_alpha = m * image_seq_len + b
        if self.temporal_sqrt_scaling:
            shift_alpha *= math.sqrt(max(num_frames, 1))
        return shift_alpha

    def _apply_shift(self, alpha: float, t: Tensor) -> Tensor:
        """Apply time shift: t' = alpha * t / (1 + (alpha - 1) * t)"""
        return alpha * t / (1.0 + (alpha - 1.0) * t)

    def sample_noise(self, x_start: Tensor) -> Tensor:
        """Sample noise matching the input shape."""
        return torch.randn_like(x_start)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Compute noisy sample at timestep t using linear interpolation.

        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * noise

        Args:
            x_start: Clean data (x_0)
            t: Timesteps in [0, 1] range
            noise: Noise (x_1)

        Returns:
            Noisy samples at timestep t
        """
        # Reshape t for broadcasting
        while t.dim() < x_start.dim():
            t = t.unsqueeze(-1)

        t_scaled = (1 - self.sigma_min) * t
        return (1 - t_scaled) * x_start + t * noise

    def get_target(
        self,
        prediction_type: str,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Get training target based on prediction type.

        Args:
            prediction_type: One of "velocity", "v", "noise", "eps", "sample", "x0"
            x_start: Clean data
            x_t: Noisy sample (unused, kept for API compatibility)
            t: Timestep (unused for velocity, kept for API compatibility)
            noise: Noise sample

        Returns:
            Target tensor for training
        """
        prediction_key = prediction_type.lower()
        if prediction_key in {"velocity", "v"}:
            return self.get_velocity(x_start=x_start, noise=noise)
        if prediction_key in {"noise", "eps", "epsilon"}:
            return noise
        if prediction_key in {"sample", "x0", "x_start"}:
            return x_start
        raise ValueError(f"Unsupported prediction_type for flow matching: {prediction_type}")

    def get_velocity(self, x_start: Tensor, noise: Tensor) -> Tensor:
        """Compute velocity target: v = (1 - sigma_min) * noise - x_0.

        This is the constant velocity that transports x_0 to x_1 along
        the linear interpolation path.
        """
        return (1 - self.sigma_min) * noise - x_start
