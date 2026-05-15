from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable, Mapping

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
        wandb_logger: object | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.wandb_logger = wandb_logger
        self.global_step = 0
        diffusion_schedule = getattr(model, "diffusion_schedule_config", None) or {}
        self.diffusion_objective = DiffusionTrainingObjective(
            timesteps=int(diffusion_schedule.get("timesteps", config.diffusion_timesteps)),
            beta_schedule=str(diffusion_schedule.get("beta_schedule", config.diffusion_beta_schedule)),
            linear_start=float(diffusion_schedule.get("linear_start", config.diffusion_linear_start)),
            linear_end=float(diffusion_schedule.get("linear_end", config.diffusion_linear_end)),
            rescale_betas_zero_snr=bool(diffusion_schedule.get("rescale_betas_zero_snr", config.diffusion_rescale_betas_zero_snr)),
            offset_noise_strength=config.diffusion_offset_noise_strength,
            use_dynamic_rescale=bool(diffusion_schedule.get("use_dynamic_rescale", False)),
            base_scale=float(diffusion_schedule.get("base_scale", 0.7)),
            turning_step=int(diffusion_schedule.get("turning_step", 400)),
        )
        self.inference_sampler = DiffusionInferenceSampler(
            model=self.model,
            objective=self.diffusion_objective,
            prediction_type=getattr(model, "prediction_type", "noise"),
            scheduler_name=config.inference_scheduler,
        )
        # Second sampler that points at the frozen base model only. Used at
        # eval time to produce a "no-adapter" baseline rollout from the same
        # starting noise as the adapted rollout — makes the visual difference
        # exactly attributable to the adapter rather than to noise drift.
        base_model = getattr(model, "base_model", None)
        if self.wandb_logger is not None and base_model is not None:
            self.base_inference_sampler = DiffusionInferenceSampler(
                model=base_model,
                objective=self.diffusion_objective,
                prediction_type=getattr(base_model, "prediction_type", "noise"),
                scheduler_name=config.inference_scheduler,
            )
        else:
            self.base_inference_sampler = None
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
            # DynamiCrafter-style data SNR shaping: scale x_0 by scale_arr[t]
            # BEFORE noising, so q_sample and the v/eps target all live in the
            # same attenuated space the base model was trained in. No-op when
            # `use_dynamic_rescale=False`.
            target_scaled = self.diffusion_objective.scale_x_start(target, t)
            x_t = self.diffusion_objective.q_sample(x_start=target_scaled, t=t, noise=noise)
            prediction = self.model(x_t, t, batch.get("cond"))
            target_tensor = self.diffusion_objective.get_target(
                prediction_type=prediction_type or "noise",
                x_start=target_scaled,
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
        # Push scalar metrics to wandb every step. Non-scalar entries (e.g.
        # the `generated_samples` tensor) are filtered inside log_metrics; the
        # video panels are pushed separately by `_maybe_generate_samples`.
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step=self.global_step)
        return metrics

    def train(
        self,
        loader: Iterable,
        *,
        max_steps: int,
        preprocessor: Callable[..., Mapping[str, object]] | None = None,
        log_every: int = 1,
        on_step: Callable[[int, dict[str, object]], None] | None = None,
    ) -> dict[str, float]:
        """Run the standard outer training loop for ``max_steps`` global steps.

        Folds the boilerplate (epoch counter, dataloader iteration, preprocess
        call, periodic print, elapsed/throughput tracking) into one entry
        point so individual scripts can be ~10 lines of setup + ``trainer.train(...)``.

        Args:
            loader: any iterable yielding raw batches. Re-iterated each epoch
                until ``max_steps`` is reached.
            max_steps: target global step count (compared against
                ``self.global_step``, so resumed runs do the right thing).
            preprocessor: optional callable applied to each raw batch before
                ``training_step``. Called as ``preprocessor(raw_batch, train=True)``
                to match :class:`DynamiCrafterBatchPreprocessor` (extra kwarg
                ignored if the callable doesn't accept it). Pass ``None`` when
                your dataloader already yields fully-formed trainer batches.
            log_every: print a per-step summary every N steps. Set to 0 to disable.
            on_step: optional callback invoked after each step with
                ``(global_step, metrics)`` — useful for custom logging or
                early-stopping logic without subclassing the trainer.

        Returns:
            A dict with ``final_avg_loss``, ``elapsed_seconds``, ``steps``,
            and ``epochs`` so callers can decide what (if anything) to print at
            the end.
        """
        running_loss = 0.0
        running_count = 0
        epoch = 0
        start = time.time()
        while self.global_step < max_steps:
            epoch += 1
            for raw_batch in loader:
                if self.global_step >= max_steps:
                    break
                batch = (
                    _call_preprocessor(preprocessor, raw_batch)
                    if preprocessor is not None
                    else raw_batch
                )
                metrics = self.training_step(batch)
                loss_value = float(metrics["loss"])
                running_loss += loss_value
                running_count += 1
                if on_step is not None:
                    on_step(self.global_step, dict(metrics))
                if log_every > 0 and self.global_step % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"epoch={epoch} step={self.global_step}/{max_steps} "
                        f"loss={loss_value:.5f} avg_loss={running_loss / running_count:.5f} "
                        f"steps/s={self.global_step / max(elapsed, 1e-6):.2f}"
                    )

        elapsed = time.time() - start
        avg_loss = running_loss / max(running_count, 1)
        if log_every > 0:
            print(f"done. final_avg_loss={avg_loss:.5f} elapsed={elapsed:.1f}s")
        return {
            "final_avg_loss": avg_loss,
            "elapsed_seconds": elapsed,
            "steps": float(self.global_step),
            "epochs": float(epoch),
        }

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
        if self.global_step % self.config.inference_every_n_steps - 1 != 0:
            return None

        target = batch.get("target")
        steps = self.config.inference_num_steps
        if self.wandb_logger is not None and self.base_inference_sampler is not None and isinstance(target, Tensor):
            shared_noise = torch.randn_like(target)
            adapted_samples = self.inference_sampler.sample_from_batch(
                batch=batch, num_inference_steps=steps, initial_sample=shared_noise
            )
            base_cond = _strip_adapter_only_keys(batch.get("cond"))
            base_batch = {"target": target, "cond": base_cond}
            base_samples = self.base_inference_sampler.sample_from_batch(
                batch=base_batch, num_inference_steps=steps, initial_sample=shared_noise
            )
            self.wandb_logger.log_videos(
                prediction_latents=adapted_samples,
                base_prediction_latents=base_samples,
                target_latents=target,
                cond=batch.get("cond"),
                step=self.global_step,
            )
            return adapted_samples

        samples = self.generate_samples(batch=batch, num_inference_steps=steps)
        if self.wandb_logger is not None and isinstance(target, Tensor):
            self.wandb_logger.log_videos(
                prediction_latents=samples,
                target_latents=target,
                cond=batch.get("cond"),
                step=self.global_step,
            )
        return samples


def _call_preprocessor(preprocessor: Callable[..., Mapping[str, object]], raw_batch: object) -> Mapping[str, object]:
    """Call ``preprocessor(batch, train=True)`` when its signature accepts
    ``train``, otherwise ``preprocessor(batch)``. Lets the trainer accept
    both :class:`DynamiCrafterBatchPreprocessor` (expects the kwarg) and
    plain user lambdas. Signature is inspected instead of catching TypeError
    so real bugs inside the preprocessor aren't silently swallowed."""
    try:
        params = inspect.signature(preprocessor).parameters
    except (TypeError, ValueError):
        params = {}
    if "train" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return preprocessor(raw_batch, train=True)
    return preprocessor(raw_batch)


def _strip_adapter_only_keys(cond: object | None) -> object | None:
    if not isinstance(cond, Mapping):
        return cond
    stripped = dict(cond)
    # The condition encoder's output lives under "embedding"; the frozen base
    # was never trained to consume it, so feeding it to the base-only rollout
    # would be at best ignored and at worst confusing for downstream hooks.
    stripped.pop("embedding", None)
    return stripped
