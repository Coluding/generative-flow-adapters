from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.config import TrainingConfig
from generative_flow_adapters.inference import DiffusionInferenceSampler
from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective
from generative_flow_adapters.losses.flow_matching import FlowMatchingTrainingObjective
from generative_flow_adapters.losses.registry import LossRegistry


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.global_step = 0
        diffusion_schedule = getattr(model, "diffusion_schedule_config", None) or {}
        self.diffusion_objective = DiffusionTrainingObjective(
            timesteps=int(diffusion_schedule.get("timesteps", config.diffusion_timesteps)),
            beta_schedule=str(diffusion_schedule.get("beta_schedule", config.diffusion_beta_schedule)),
            linear_start=float(diffusion_schedule.get("linear_start", config.diffusion_linear_start)),
            linear_end=float(diffusion_schedule.get("linear_end", config.diffusion_linear_end)),
            rescale_betas_zero_snr=bool(diffusion_schedule.get("rescale_betas_zero_snr", config.diffusion_rescale_betas_zero_snr)),
            offset_noise_strength=config.diffusion_offset_noise_strength,
        )
        self.inference_sampler = DiffusionInferenceSampler(
            model=self.model,
            objective=self.diffusion_objective,
            prediction_type=getattr(model, "prediction_type", "noise"),
            scheduler_name=config.inference_scheduler,
        )
        self.flow_objective = FlowMatchingTrainingObjective(
            sigma_min=float(config.extra.get("flow_sigma_min", 1e-5)),
            shift_schedule=bool(config.extra.get("flow_shift_schedule", True)),
            base_shift=float(config.extra.get("flow_base_shift", 1.0)),
            max_shift=float(config.extra.get("flow_max_shift", 3.0)),
            shift_x1=float(config.extra.get("flow_shift_x1", 256.0)),
            shift_x2=float(config.extra.get("flow_shift_x2", 4096.0)),
            temporal_sqrt_scaling=bool(config.extra.get("flow_temporal_sqrt_scaling", True)),
        )

    def training_step(self, batch: Mapping[str, Tensor | object]) -> dict[str, object]:
        self.model.train()
        model_type = getattr(self.model, "model_type", None)
        prediction_type = getattr(self.model, "prediction_type", None)

        target = batch["target"]
        if not isinstance(target, Tensor):
            raise TypeError("batch['target'] must be a tensor.")

        if model_type == "diffusion":
            batch_size = target.shape[0]
            t_value = batch.get("t")
            if isinstance(t_value, Tensor):
                t = t_value.to(device=target.device, dtype=torch.long)
            else:
                t = self.diffusion_objective.sample_timesteps(batch_size=batch_size, device=target.device)
            noise = self.diffusion_objective.sample_noise(target)
            x_t = self.diffusion_objective.q_sample(x_start=target, t=t, noise=noise)
            prediction = self.model(x_t, t, batch.get("cond"))
            target_tensor = self.diffusion_objective.get_target(
                prediction_type=prediction_type or "noise",
                x_start=target,
                x_t=x_t,
                t=t,
                noise=noise,
            ) ## Very important. We can predict either noise, starting data point or velocity. Velocity is a combination of the first two.
            loss = self.loss_fn(prediction, target_tensor)
        else:
            x_t = batch["x_t"]
            if not isinstance(x_t, Tensor):
                raise TypeError("batch['x_t'] must be a tensor.")
            t_value = batch.get("t")
            use_batch_timesteps = bool(self.config.extra.get("use_batch_timesteps_for_flow", False))
            if use_batch_timesteps:
                if not isinstance(t_value, Tensor):
                    raise TypeError("batch['t'] must be a tensor when use_batch_timesteps_for_flow=true.")
                t = t_value.to(device=x_t.device, dtype=x_t.dtype)
            else:
                batch_size = x_t.shape[0]
                patch_size = 2
                base_model = getattr(self.model, "base_model", None)
                if base_model is not None:
                    patch_size = int(getattr(getattr(base_model, "config", None), "patch_size", patch_size))
                height = int(x_t.shape[-2]) if x_t.dim() >= 4 else None
                width = int(x_t.shape[-1]) if x_t.dim() >= 4 else None
                num_frames = int(x_t.shape[-3]) if x_t.dim() >= 5 else 1
                t = self.flow_objective.sample_timesteps(
                    batch_size=batch_size,
                    device=x_t.device,
                    dtype=x_t.dtype,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    patch_size=patch_size,
                )
            prediction = self.model(x_t, t, batch.get("cond"))
            loss = self.loss_fn(prediction, target)

        if self.config.local_consistency_weight > 0.0 and "shortcut_target" in batch:
            shortcut_target = batch["shortcut_target"]
            if not isinstance(shortcut_target, Tensor):
                raise TypeError("batch['shortcut_target'] must be a tensor.")
            consistency = LossRegistry.get_consistency_loss("local_consistency")(prediction, shortcut_target)
            loss = loss + self.config.local_consistency_weight * consistency

        if self.config.shortcut_direction_weight > 0.0 and "shortcut_target" in batch:
            shortcut_target = batch["shortcut_target"]
            if not isinstance(shortcut_target, Tensor):
                raise TypeError("batch['shortcut_target'] must be a tensor.")
            shortcut_loss = LossRegistry.get_consistency_loss("shortcut_direction")(prediction, shortcut_target)
            loss = loss + self.config.shortcut_direction_weight * shortcut_loss

        if self.config.multistep_consistency_weight > 0.0 and "self_consistency_target" in batch:
            self_consistency_target = batch["self_consistency_target"]
            if not isinstance(self_consistency_target, Tensor):
                raise TypeError("batch['self_consistency_target'] must be a tensor.")
            consistency = LossRegistry.get_consistency_loss("multistep_self_consistency")(prediction, self_consistency_target)
            loss = loss + self.config.multistep_consistency_weight * consistency

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1

        metrics: dict[str, object] = {"loss": float(loss.detach().cpu())}
        if self.config.shortcut_direction_weight > 0.0 and "shortcut_target" in batch:
            shortcut_target = batch["shortcut_target"]
            if isinstance(shortcut_target, Tensor):
                shortcut_metric = LossRegistry.get_consistency_loss("shortcut_direction")(prediction.detach(), shortcut_target.detach())
                metrics["shortcut_direction_loss"] = float(shortcut_metric.detach().cpu())
        generated_samples = self._maybe_generate_samples(batch=batch, model_type=model_type)
        if generated_samples is not None:
            metrics["generated_samples"] = generated_samples.detach().cpu()
        return metrics

    def generate_samples(
        self,
        batch: Mapping[str, Tensor | object],
        num_inference_steps: int | None = None,
    ) -> Tensor:
        model_type = getattr(self.model, "model_type", None)
        if model_type != "diffusion":
            raise ValueError("Inference sampling is only implemented for diffusion models.")
        steps = num_inference_steps or self.config.inference_num_steps
        return self.inference_sampler.sample_from_batch(batch=batch, num_inference_steps=steps)

    def _maybe_generate_samples(self, batch: Mapping[str, Tensor | object], model_type: str | None) -> Tensor | None:
        if model_type != "diffusion":
            return None
        if self.config.inference_every_n_steps is None or self.config.inference_every_n_steps <= 0:
            return None
        if self.global_step % self.config.inference_every_n_steps != 0:
            return None
        return self.generate_samples(batch=batch, num_inference_steps=self.config.inference_num_steps)
