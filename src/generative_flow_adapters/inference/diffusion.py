from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm

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
        initial_sample: Tensor | None = None,
        unconditional_cond: object | None = None,
        guidance_scale: float = 1.0,
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
        """Run the denoising rollout. Set ``guidance_scale > 1`` and pass
        ``unconditional_cond`` to enable classifier-free guidance — required
        for SD-derived models like DynamiCrafter, which otherwise produce
        near-noise samples.

        With CFG, every step costs two UNet forwards instead of one: one with
        the supplied ``cond``, one with ``unconditional_cond``. The outputs
        are combined as ``e_uncond + scale * (e_cond - e_uncond)`` (single-CFG;
        if you need DynamiCrafter's text/image dual-CFG, build a custom loop).
        """
        cfg_active = guidance_scale > 1.0 and unconditional_cond is not None
        scheduler = self._build_scheduler()
        if initial_sample is not None:
            if tuple(initial_sample.shape) != tuple(shape):
                raise ValueError(
                    f"initial_sample shape {tuple(initial_sample.shape)} does not match target {tuple(shape)}."
                )
            sample = initial_sample.to(device=device, dtype=dtype)
        else:
            sample = torch.randn(shape, device=device, dtype=dtype)
        scheduler.set_timesteps(num_inference_steps, device=device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                if verbose: iterator = tqdm(scheduler.timesteps, desc="Inference sampling...",)
                else: iterator = iter(scheduler.timesteps)
                for timestep in iterator:
                    t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
                    cond_output = self.model(sample, t, cond)
                    if False: #TODO check if we need unconditional sampling
                        uncond_output = self.model(sample, t, unconditional_cond)
                        model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
                    else:
                        model_output = cond_output
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
        if getattr(self.objective, "rescale_betas_zero_snr", False):
            scheduler_kwargs["timestep_spacing"] = "trailing"
        if scheduler_name == "ddim":
            return DDIMScheduler(**scheduler_kwargs)
        if scheduler_name == "ddpm":
            return DDPMScheduler(**scheduler_kwargs)
        raise ValueError(f"Unsupported inference scheduler: {self.scheduler_name}")

    def _scheduler_step(self, scheduler, model_output: Tensor, timestep, sample: Tensor) -> Tensor:
        if getattr(self.objective, "use_dynamic_rescale", False) and self.scheduler_name.lower() == "ddim":
            return self._dynamic_rescale_ddim_step(scheduler, model_output, timestep, sample)
        step_signature = inspect.signature(scheduler.step)
        if "eta" in step_signature.parameters:
            step_output = scheduler.step(model_output, timestep, sample, eta=0.0)
        else:
            step_output = scheduler.step(model_output, timestep, sample)
        return step_output.prev_sample

    def _dynamic_rescale_ddim_step(
        self, scheduler, model_output: Tensor, timestep, sample: Tensor
    ) -> Tensor:
        """DDIM step that reverses DynamiCrafter's data-side SNR shaping.

        At training time `x_0` was multiplied by ``scale_arr[t]`` before
        noising, so the network's predicted x_0 lives in ``scale_arr[t]``-space.
        Between consecutive DDIM steps we re-bucket it into the *previous*
        step's scale space (``prev_scale / cur_scale``) before doing the
        standard DDIM update; at the final step ``scale_arr[0] = 1.0`` so the
        output lands in the unscaled, VAE-compatible space.

        This mirrors
        :class:`external_deps.lvdm.models.samplers.ddim.DDIMSampler.p_sample_ddim`.
        """
        timestep_int = int(timestep)
        num_train = int(scheduler.config.num_train_timesteps)
        num_inference = int(scheduler.num_inference_steps)
        prev_timestep = timestep_int - num_train // num_inference

        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        final_alpha = (
            scheduler.final_alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
            if hasattr(scheduler, "final_alpha_cumprod")
            else alphas_cumprod[0]
        )
        alpha_t = alphas_cumprod[timestep_int]
        alpha_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
        sqrt_alpha_t_prev = alpha_t_prev.sqrt()
        sqrt_one_minus_alpha_t_prev = (1.0 - alpha_t_prev).sqrt()

        # Convert the model output to (pred_x0, pred_eps) regardless of
        # parameterization. pred_eps is left in the model's training-space and
        # NOT rescaled — that matches the reference DDIM exactly.
        prediction_type = _map_prediction_type(self.prediction_type)
        if prediction_type == "v_prediction":
            pred_x0 = sqrt_alpha_t * sample - sqrt_one_minus_alpha_t * model_output
            pred_eps = sqrt_alpha_t * model_output + sqrt_one_minus_alpha_t * sample
        elif prediction_type == "epsilon":
            pred_x0 = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
            pred_eps = model_output
        elif prediction_type == "sample":
            pred_x0 = model_output
            pred_eps = (sample - sqrt_alpha_t * model_output) / sqrt_one_minus_alpha_t
        else:
            raise ValueError(f"Unsupported prediction_type for dynamic-rescale DDIM: {self.prediction_type}")

        scale_arr = self.objective.scale_arr.to(device=sample.device, dtype=sample.dtype)
        cur_scale = scale_arr[timestep_int]
        prev_scale = scale_arr[max(prev_timestep, 0)]
        pred_x0 = pred_x0 * (prev_scale / cur_scale)

        return sqrt_alpha_t_prev * pred_x0 + sqrt_one_minus_alpha_t_prev * pred_eps


def _map_prediction_type(prediction_type: str) -> str:
    key = prediction_type.lower()
    if key in {"noise", "eps", "epsilon"}:
        return "epsilon"
    if key in {"velocity", "v"}:
        return "v_prediction"
    if key in {"sample", "x0", "x_start"}:
        return "sample"
    raise ValueError(f"Unsupported diffusion prediction_type for inference: {prediction_type}")
