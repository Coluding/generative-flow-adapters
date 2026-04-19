from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from diffusers import DDIMScheduler, DDPMScheduler

from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective


class DiffusionInferenceSampler:
    def __init__(
        self,
        model: nn.Module,
        objective: DiffusionTrainingObjective,
        prediction_type: str,
        scheduler_name: str = "ddim",
    ) -> None:
        self.model = model
        self.objective = objective
        self.prediction_type = prediction_type
        self.scheduler_name = scheduler_name

    def sample_from_batch(
        self,
        batch: Mapping[str, Tensor | object],
        num_inference_steps: int = 50,
    ) -> Tensor:
        target = batch.get("target")
        if not isinstance(target, Tensor):
            raise TypeError("Diffusion inference requires batch['target'] to infer the latent shape.")
        return self.sample(
            shape=tuple(target.shape),
            cond=batch.get("cond"),
            device=target.device,
            dtype=target.dtype,
            num_inference_steps=num_inference_steps,
        )

    def sample(
        self,
        shape: tuple[int, ...],
        cond: object | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_inference_steps: int = 50,
    ) -> Tensor:
        scheduler = self._build_scheduler()
        sample = torch.randn(shape, device=device, dtype=dtype)
        scheduler.set_timesteps(num_inference_steps, device=device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for timestep in scheduler.timesteps:
                    t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
                    model_output = self.model(sample, t, cond)
                    sample = self._scheduler_step(scheduler, model_output, timestep, sample)
        finally:
            self.model.train(was_training)

        return sample

    def _build_scheduler(self):
        prediction_type = _map_prediction_type(self.prediction_type)
        scheduler_name = self.scheduler_name.lower()
        scheduler_kwargs = {
            "num_train_timesteps": self.objective.timesteps,
            "trained_betas": self.objective.betas.detach().cpu().numpy(),
            "prediction_type": prediction_type,
            "clip_sample": False,
        }
        if scheduler_name == "ddim":
            return DDIMScheduler(**scheduler_kwargs)
        if scheduler_name == "ddpm":
            return DDPMScheduler(**scheduler_kwargs)
        raise ValueError(f"Unsupported inference scheduler: {self.scheduler_name}")

    def _scheduler_step(self, scheduler, model_output: Tensor, timestep, sample: Tensor) -> Tensor:
        step_signature = inspect.signature(scheduler.step)
        if "eta" in step_signature.parameters:
            step_output = scheduler.step(model_output, timestep, sample, eta=0.0)
        else:
            step_output = scheduler.step(model_output, timestep, sample)
        return step_output.prev_sample


def _map_prediction_type(prediction_type: str) -> str:
    key = prediction_type.lower()
    if key in {"noise", "eps", "epsilon"}:
        return "epsilon"
    if key in {"velocity", "v"}:
        return "v_prediction"
    if key in {"sample", "x0", "x_start"}:
        return "sample"
    raise ValueError(f"Unsupported diffusion prediction_type for inference: {prediction_type}")
