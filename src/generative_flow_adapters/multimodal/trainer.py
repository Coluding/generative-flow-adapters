"""Multimodal trainer.

Forks the diffusion branch of
:class:`generative_flow_adapters.training.trainer.Trainer` for the multi-stream
case: each output stream is noised at its **own** independently-sampled timestep
``t_m ~ U(0, T)`` (the UWM scheme), the model predicts every stream in one
forward, and a single objective sums the per-stream denoising losses with fixed
weights ``Σ w_m · L_m`` (sub-decisions 1 & 3). No shortcut/`d` path — shortcut
is shelved for the multimodal line (decision note).

Reuses the generic IO helpers (``CheckpointManager``, ``JsonlMetricsLogger``)
unchanged; only the loss math is multimodal.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.config import TrainingConfig
from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


class MultiModalTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        modality_specs: list[OutputModalitySpec],
        *,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        wandb_logger: object | None = None,
        jsonl_logger: object | None = None,
        checkpoint_manager: object | None = None,
    ) -> None:
        if getattr(model, "model_type", "diffusion") != "diffusion":
            raise NotImplementedError(
                "MultiModalTrainer currently supports diffusion bases only "
                "(flow-matching multimodal is deferred — see the decision note)."
            )
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.modality_specs = list(modality_specs)
        self.weights = {m.name: float(m.loss_weight) for m in self.modality_specs}
        self.loss_fn = loss_fn or (lambda pred, target: torch.mean((pred - target) ** 2))
        self.wandb_logger = wandb_logger
        self.jsonl_logger = jsonl_logger
        self.checkpoint_manager = checkpoint_manager
        self.global_step = 0

        # Shared objective for the modality streams; the video stream uses the
        # base's own schedule when it exposes one (dynamic-rescale etc.).
        self.objective = DiffusionTrainingObjective(
            timesteps=config.diffusion_timesteps,
            beta_schedule=config.diffusion_beta_schedule,
            linear_start=config.diffusion_linear_start,
            linear_end=config.diffusion_linear_end,
            rescale_betas_zero_snr=config.diffusion_rescale_betas_zero_snr,
            offset_noise_strength=config.diffusion_offset_noise_strength,
        )
        schedule = getattr(model, "diffusion_schedule_config", None) or {}
        self.video_objective = (
            DiffusionTrainingObjective(
                timesteps=int(schedule.get("timesteps", config.diffusion_timesteps)),
                beta_schedule=str(schedule.get("beta_schedule", config.diffusion_beta_schedule)),
                linear_start=float(schedule.get("linear_start", config.diffusion_linear_start)),
                linear_end=float(schedule.get("linear_end", config.diffusion_linear_end)),
                rescale_betas_zero_snr=bool(schedule.get("rescale_betas_zero_snr", config.diffusion_rescale_betas_zero_snr)),
                offset_noise_strength=config.diffusion_offset_noise_strength,
                use_dynamic_rescale=bool(schedule.get("use_dynamic_rescale", False)),
                base_scale=float(schedule.get("base_scale", 0.7)),
                turning_step=int(schedule.get("turning_step", 400)),
            )
            if schedule
            else self.objective
        )

    def _objective_for(self, name: str):
        spec = next(m for m in self.modality_specs if m.name == name)
        return self.video_objective if spec.has_frozen_prior else self.objective

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _forward_and_loss(self, batch: Mapping[str, object]) -> tuple[Tensor, dict[str, float]]:
        targets = batch["targets"]
        if not isinstance(targets, Mapping):
            raise TypeError("multimodal batch['targets'] must be a mapping of {name: tensor}.")
        device = self._device()
        cond = _to_device(batch.get("cond"), device)
        prediction_type = getattr(self.model, "prediction_type", "noise")

        x_t: dict[str, Tensor] = {}
        t: dict[str, Tensor] = {}
        target_tensors: dict[str, Tensor] = {}
        for name, target in targets.items():
            target = target.to(device)
            objective = self._objective_for(name)
            t_m = objective.sample_timesteps(batch_size=target.shape[0], device=device)
            noise = objective.sample_noise(target)
            target_scaled = objective.scale_x_start(target, t_m)
            x_t[name] = objective.q_sample(x_start=target_scaled, t=t_m, noise=noise)
            t[name] = t_m
            target_tensors[name] = objective.get_target(
                prediction_type=prediction_type or "noise",
                x_start=target_scaled, x_t=x_t[name], t=t_m, noise=noise,
            )

        predictions = self.model(x_t, t, cond)

        loss = torch.zeros((), device=device)
        components: dict[str, float] = {}
        for name, target_tensor in target_tensors.items():
            stream_loss = self.loss_fn(predictions[name].float(), target_tensor.float())
            weight = self.weights.get(name, 1.0)
            loss = loss + weight * stream_loss
            components[f"loss_{name}"] = float(stream_loss.detach().cpu())
        components["loss"] = float(loss.detach().cpu())
        return loss, components

    def training_step(self, batch: Mapping[str, object]) -> dict[str, float]:
        self.model.train()
        loss, components = self._forward_and_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(components, step=self.global_step)
        return components

    def train(
        self,
        loader: Iterable,
        *,
        max_steps: int,
        preprocessor: Callable[..., Mapping[str, object]] | None = None,
        log_every: int = 1,
        on_step: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        running = 0.0
        count = 0
        start = time.time()
        while self.global_step < max_steps:
            for raw_batch in loader:
                if self.global_step >= max_steps:
                    break
                batch = preprocessor(raw_batch) if preprocessor is not None else raw_batch
                metrics = self.training_step(batch)
                running += metrics["loss"]
                count += 1
                if self.jsonl_logger is not None:
                    self.jsonl_logger.log(metrics, step=self.global_step, split="train")
                if on_step is not None:
                    on_step(self.global_step, dict(metrics))
                if log_every > 0 and self.global_step % log_every == 0:
                    parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                    print(f"step={self.global_step}/{max_steps} {parts}")
                if self.checkpoint_manager is not None and _due(self.config.checkpoint_every_n_steps, self.global_step):
                    self.checkpoint_manager.save(
                        model=self.model, optimizer=self.optimizer, global_step=self.global_step, tag=None
                    )
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save(
                model=self.model, optimizer=self.optimizer, global_step=self.global_step, tag="final"
            )
        if self.jsonl_logger is not None:
            self.jsonl_logger.close()
        return {
            "final_avg_loss": running / max(count, 1),
            "elapsed_seconds": time.time() - start,
            "steps": float(self.global_step),
        }


def _due(every_n: int | None, step: int) -> bool:
    return bool(every_n) and every_n > 0 and step % every_n == 0


def _to_device(cond: object | None, device: torch.device) -> object | None:
    if isinstance(cond, Mapping):
        return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in cond.items()}
    if isinstance(cond, Tensor):
        return cond.to(device)
    return cond
