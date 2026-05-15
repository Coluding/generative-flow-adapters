from __future__ import annotations

import torch
from torch import Tensor

from generative_flow_adapters.backbones.dynamicrafter.common import extract_into_tensor
from generative_flow_adapters.backbones.dynamicrafter.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr


def diffusion_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2)


class DiffusionScheduleConfig(dict):
    """Lightweight mapping used to carry backbone-specific diffusion schedule settings."""


class DiffusionTrainingObjective:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 8.5e-4,
        linear_end: float = 1.2e-2,
        rescale_betas_zero_snr: bool = False,
        offset_noise_strength: float = 0.0,
        use_dynamic_rescale: bool = False,
        base_scale: float = 0.7,
        turning_step: int = 400,
    ) -> None:
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.offset_noise_strength = offset_noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.base_scale = base_scale
        self.turning_step = turning_step

        betas = make_beta_schedule(
            schedule=beta_schedule,
            n_timestep=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        betas = torch.as_tensor(betas, dtype=torch.float32)
        if rescale_betas_zero_snr:
            betas = torch.as_tensor(rescale_zero_terminal_snr(betas.cpu().numpy()), dtype=torch.float32)
        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # DynamiCrafter-style data-side SNR shaping (see CLAUDE.md / writeup):
        # multiply the clean latent x_0 by `scale_arr[t]` before noising, so the
        # model learns v in a scale_arr[t]-attenuated space. At inference the
        # sampler reverses this between consecutive DDIM steps.
        if use_dynamic_rescale:
            ramp = torch.linspace(1.0, base_scale, turning_step)
            flat = torch.full((timesteps,), base_scale)
            self.scale_arr = torch.cat([ramp, flat]).to(torch.float32)
        else:
            self.scale_arr = None

    def scale_x_start(self, x_start: Tensor, t: Tensor) -> Tensor:
        """Apply the DynamiCrafter per-timestep data rescale, or no-op when off.

        Called before `q_sample` / `get_target` in training so the noised x_t
        and the regression target (v / eps / x0) all live in the scaled space.
        """
        if not self.use_dynamic_rescale or self.scale_arr is None:
            return x_start
        scale = extract_into_tensor(
            self.scale_arr.to(device=x_start.device, dtype=x_start.dtype), t, x_start.shape
        )
        return x_start * scale

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def sample_noise(self, x_start: Tensor) -> Tensor:
        if self.offset_noise_strength > 0.0 and x_start.dim() == 5:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device, dtype=x_start.dtype)
            return torch.randn_like(x_start) + self.offset_noise_strength * offset_noise
        return torch.randn_like(x_start)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_alphas = extract_into_tensor(self.sqrt_alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype), t, x_start.shape)
        sqrt_one_minus = extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype), t, x_start.shape
        )
        return sqrt_alphas * x_start + sqrt_one_minus * noise

    def get_target(self, prediction_type: str, x_start: Tensor, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        prediction_key = prediction_type.lower()
        if prediction_key in {"noise", "eps", "epsilon"}:
            return noise
        if prediction_key in {"sample", "x0", "x_start"}:
            return x_start
        if prediction_key in {"velocity", "v"}:
            return self.get_velocity(x_start=x_start, noise=noise, t=t)
        raise ValueError(f"Unsupported diffusion prediction_type: {prediction_type}")

    def get_velocity(self, x_start: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        sqrt_alphas = extract_into_tensor(self.sqrt_alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype), t, x_start.shape)
        sqrt_one_minus = extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype), t, x_start.shape
        )
        return sqrt_alphas * noise - sqrt_one_minus * x_start
