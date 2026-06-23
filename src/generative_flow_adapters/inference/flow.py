"""Flow-matching inference sampler.

Drop-in counterpart to :class:`DiffusionInferenceSampler` for rectified-flow /
velocity-prediction models (Wan2.1). Denoises with the vendored
``FlowUniPCMultistepScheduler`` instead of DDIM/DDPM, mirroring the proven
rollout in ``examples/wan_generate_video.py``.

Timestep convention: the model is fed ``t = sigma · timestep_scale`` where
``sigma = scheduler_timestep / num_train_timesteps``. With
``timestep_scale == num_train_timesteps`` (the default, Wan native) this is just
the raw scheduler timestep in ``[0, 1000]`` — matching both the pretrained base
and the fixed ``WanBatchPreprocessor`` training convention. Set
``timestep_scale=1.0`` to feed ``sigma ∈ [0, 1]`` instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from tqdm import tqdm


class FlowInferenceSampler:
    def __init__(
        self,
        model: nn.Module,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        timestep_scale: float | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        # Model is fed t = sigma * timestep_scale (see module docstring).
        self.timestep_scale = float(timestep_scale if timestep_scale is not None else num_train_timesteps)
        self.amp_dtype = amp_dtype
        # Kept for interface parity with DiffusionInferenceSampler.
        self.prediction_type = "velocity"

    def sample_from_batch(
        self,
        batch: Mapping[str, Tensor | object],
        num_inference_steps: int = 50,
        initial_sample: Tensor | None = None,
        unconditional_cond: object | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        target = batch.get("target")
        if not isinstance(target, Tensor):
            raise TypeError("Flow inference requires batch['target'] to infer the latent shape.")
        return self.sample(
            shape=tuple(target.shape),
            cond=batch.get("cond"),
            device=target.device,
            dtype=target.dtype,
            num_inference_steps=num_inference_steps,
            initial_sample=initial_sample,
            unconditional_cond=unconditional_cond,
            guidance_scale=guidance_scale,
        )

    def sample(
        self,
        shape: tuple[int, ...],
        cond: object | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_inference_steps: int = 50,
        initial_sample: Tensor | None = None,
        unconditional_cond: object | None = None,
        guidance_scale: float = 1.0,
        verbose: bool = True,
    ) -> Tensor:
        """Rectified-flow ODE rollout. ``guidance_scale > 1`` with
        ``unconditional_cond`` enables classifier-free guidance (two model
        forwards per step)."""
        from generative_flow_adapters.backbones.wan.utils.fm_solvers_unipc import (
            FlowUniPCMultistepScheduler,
        )

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(num_inference_steps, device=device, shift=self.shift)

        if initial_sample is not None:
            if tuple(initial_sample.shape) != tuple(shape):
                raise ValueError(
                    f"initial_sample shape {tuple(initial_sample.shape)} does not match {tuple(shape)}."
                )
            sample = initial_sample.to(device=device, dtype=torch.float32)
        else:
            sample = torch.randn(shape, device=device, dtype=torch.float32)

        cfg_active = guidance_scale > 1.0 and unconditional_cond is not None
        autocast = (
            torch.amp.autocast(device_type=device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else contextlib.nullcontext()
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                iterator = tqdm(scheduler.timesteps, desc="Flow sampling...") if verbose else scheduler.timesteps
                for timestep in iterator:
                    sigma = float(timestep) / self.num_train_timesteps
                    t_model = torch.full((shape[0],), sigma * self.timestep_scale, device=device, dtype=dtype)
                    with autocast:
                        v = self.model(sample.to(dtype), t_model, cond)
                        if cfg_active:
                            v_uncond = self.model(sample.to(dtype), t_model, unconditional_cond)
                            v = v_uncond + guidance_scale * (v - v_uncond)
                    sample = scheduler.step(v.float(), timestep, sample, return_dict=False)[0]
        finally:
            self.model.train(was_training)

        return sample
