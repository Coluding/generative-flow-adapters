from __future__ import annotations

import contextlib
import inspect
import time
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.config import TrainingConfig
from generative_flow_adapters.inference import DiffusionInferenceSampler, FlowInferenceSampler
from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective
from generative_flow_adapters.losses.flow_matching import FlowMatchingTrainingObjective
from generative_flow_adapters.losses.registry import LossRegistry
from generative_flow_adapters.training.shortcut_targets import (
    compute_self_consistency_target_v,
    compute_self_consistency_target_v_flow,
    ddim_micro_step_v,
)
from generative_flow_adapters.training.step_schedule import ShortcutStepSchedule


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        config: TrainingConfig,
        wandb_logger: object | None = None,
        jsonl_logger: object | None = None,
        checkpoint_manager: object | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.wandb_logger = wandb_logger
        self.jsonl_logger = jsonl_logger
        self.checkpoint_manager = checkpoint_manager
        # Best eval metric seen so far (lower is better); drives best.pt saves.
        self.best_eval_metric = float("inf")
        self.global_step = 0
        self._amp_dtype = self._resolve_amp_dtype(config.extra.get("amp_dtype"))
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
        # Flow-matching models (Wan) need the rectified-flow sampler, not DDIM.
        self._is_flow = getattr(model, "model_type", None) in ("flow", "flow_matching")
        # The model is fed t = sigma * flow_timestep_scale (Wan native = 1000).
        self._flow_timestep_scale = float(config.extra.get("flow_timestep_scale", 1000.0))
        self.inference_sampler = self._build_inference_sampler(self.model)
        # Second sampler that points at the frozen base model only. Used at
        # eval time to produce a "no-adapter" baseline rollout from the same
        # starting noise as the adapted rollout — makes the visual difference
        # exactly attributable to the adapter rather than to noise drift.
        base_model = getattr(model, "base_model", None)
        if self.wandb_logger is not None and base_model is not None:
            self.base_inference_sampler = self._build_inference_sampler(base_model)
        else:
            self.base_inference_sampler = None
        # Paper-faithful step-size schedule (normalised s ∈ (0,1]). When set it
        # drives both training (sampled step size per batch) and the eval grid,
        # and the injected step_level becomes normalised. Absent → legacy raw-
        # timestep dyadic behaviour (`shortcut_step_level_max`).
        raw_schedule = config.extra.get("shortcut_step_schedule")
        self.step_schedule: ShortcutStepSchedule | None = (
            ShortcutStepSchedule.from_config(raw_schedule, timesteps=self.diffusion_objective.timesteps)
            if isinstance(raw_schedule, Mapping)
            else None
        )
        # Normalised step size `s` supervised by the most recent shortcut step
        # (None on anchor steps / when no shortcut target). Lets `training_step`
        # bucket the shortcut loss into one per-rung series so the otherwise
        # `s`-marginalised aggregate curve becomes readable.
        self._last_shortcut_step_level: float | None = None
        self.flow_objective = FlowMatchingTrainingObjective(
            sigma_min=float(config.extra.get("flow_sigma_min", 1e-5)),
            shift_schedule=bool(config.extra.get("flow_shift_schedule", True)),
            base_shift=float(config.extra.get("flow_base_shift", 1.0)),
            max_shift=float(config.extra.get("flow_max_shift", 3.0)),
            shift_x1=float(config.extra.get("flow_shift_x1", 256.0)),
            shift_x2=float(config.extra.get("flow_shift_x2", 4096.0)),
            temporal_sqrt_scaling=bool(config.extra.get("flow_temporal_sqrt_scaling", True)),
        )

    def _build_inference_sampler(self, model: nn.Module):
        """Pick the rollout sampler matching the model family.

        Flow-matching models (Wan) denoise with the rectified-flow ODE
        (``FlowInferenceSampler`` over ``FlowUniPCMultistepScheduler``); the Wan
        DiT runs under autocast so its fp32 time-embedding reconciles with bf16
        weights. Everything else keeps the DDIM/DDPM ``DiffusionInferenceSampler``.
        """
        if self._is_flow:
            extra = self.config.extra
            num_train = int(extra.get("flow_num_train_timesteps", 1000))
            return FlowInferenceSampler(
                model=model,
                num_train_timesteps=num_train,
                shift=float(extra.get("flow_inference_shift", 3.0)),
                timestep_scale=self._flow_timestep_scale,
                amp_dtype=self._amp_dtype,
            )
        return DiffusionInferenceSampler(
            model=model,
            objective=self.diffusion_objective,
            prediction_type=getattr(model, "prediction_type", "noise"),
            scheduler_name=self.config.inference_scheduler,
        )

    @staticmethod
    def _resolve_amp_dtype(value: object | None) -> torch.dtype | None:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        key = str(value).strip().lower()
        if key in {"none", "fp32", "float32", ""}:
            return None
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported amp_dtype: {value!r}. Expected one of bf16/fp16/none.")
        return mapping[key]

    def _autocast(self):
        """Autocast context for the model forward (no-op unless amp_dtype is set)."""
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        device_type = next(self.model.parameters()).device.type
        return torch.amp.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _forward_and_loss(
        self, batch: Mapping[str, Tensor | object]
    ) -> tuple[Tensor, dict[str, float], Tensor, Tensor, object, Tensor, dict]:
        """Run the model forward and compute the total loss + per-term components.

        Shared by :meth:`training_step` (which backprops the returned loss) and
        :meth:`evaluate` (which calls it under ``eval()``/``no_grad``). The model
        train/eval mode is the caller's responsibility — this method does not set
        it. Returns ``(loss, loss_components, x_t, t, cond, prediction, batch)``
        where ``loss_components`` holds detached scalar terms and ``batch`` is a
        copy with any synthesized shortcut targets attached.
        """
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
            #TODO this is very dynamicrafter orientated atm --> for the future we need to adjust it
            target_scaled = self.diffusion_objective.scale_x_start(target, t)
            x_t = self.diffusion_objective.q_sample(x_start=target_scaled, t=t, noise=noise)
            cond, shortcut_target = self._maybe_prepare_shortcut(
                batch=batch, x_t=x_t, t=t, cond=batch.get("cond")
            )
            with self._autocast():
                prediction = self.model(x_t, t, cond)
            # Upcast the (possibly bf16) prediction back to fp32 so the loss and
            # backward run in full precision — keeps autograd dtypes consistent
            # and the diffusion loss numerically stable. No-op when amp is off.
            prediction = prediction.float()
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
            cond, shortcut_target = self._maybe_prepare_shortcut(
                batch=batch, x_t=x_t, t=t, cond=batch.get("cond")
            )
            with self._autocast():
                prediction = self.model(x_t, t, cond)
            # See diffusion branch: upcast to fp32 for a precision-consistent
            # loss/backward. No-op when amp is off.
            prediction = prediction.float()
            loss = self.loss_fn(prediction, target)

        # Record each loss term separately so wandb shows the base loss and
        # every shortcut-consistency term next to the combined total. For
        # shortcut training their relative magnitudes are the key signal for
        # spotting collapse or a mis-weighted term.
        loss_components: dict[str, float] = {"base_loss": float(loss.detach().cpu())}

        batch = dict(batch)
        if shortcut_target is not None:
            batch.setdefault("shortcut_target", shortcut_target)
            batch.setdefault("self_consistency_target", shortcut_target)

        if self.config.local_consistency_weight > 0.0 and "shortcut_target" in batch:
            shortcut_target = batch["shortcut_target"]
            if not isinstance(shortcut_target, Tensor):
                raise TypeError("batch['shortcut_target'] must be a tensor.")
            consistency = LossRegistry.get_consistency_loss("local_consistency")(prediction, shortcut_target)
            loss_components["local_consistency_loss"] = float(consistency.detach().cpu())
            loss = loss + self.config.local_consistency_weight * consistency

        if self.config.shortcut_direction_weight > 0.0 and "shortcut_target" in batch:
            shortcut_target = batch["shortcut_target"]
            if not isinstance(shortcut_target, Tensor):
                raise TypeError("batch['shortcut_target'] must be a tensor.")
            shortcut_loss = LossRegistry.get_consistency_loss("shortcut_direction")(prediction, shortcut_target)
            loss_components["shortcut_direction_loss"] = float(shortcut_loss.detach().cpu())
            loss = loss + self.config.shortcut_direction_weight * shortcut_loss

        if self.config.multistep_consistency_weight > 0.0 and "self_consistency_target" in batch:
            self_consistency_target = batch["self_consistency_target"]
            if not isinstance(self_consistency_target, Tensor):
                raise TypeError("batch['self_consistency_target'] must be a tensor.")
            consistency = LossRegistry.get_consistency_loss("multistep_self_consistency")(prediction, self_consistency_target)
            loss_components["multistep_consistency_loss"] = float(consistency.detach().cpu())
            loss = loss + self.config.multistep_consistency_weight * consistency

        # Heun-smoothness regularizer (opt-in, orthogonal to shortcut training):
        # penalize the velocity field's material derivative along its own
        # predicted one-step trajectory. Adds one extra adapter forward (v0,
        # with grad) plus one no-grad reference forward (v1). See thesis-vault
        # theory/heun-smoothness-regularizer.md.
        if self.config.heun_smoothness_weight > 0.0:
            heun_loss = self._compute_heun_smoothness(x_t=x_t, t=t, cond=cond)
            loss_components["heun_smoothness_loss"] = float(heun_loss.detach().cpu())
            loss = loss + self.config.heun_smoothness_weight * heun_loss

        return loss, loss_components, x_t, t, cond, prediction, batch

    def training_step(self, batch: Mapping[str, Tensor | object]) -> dict[str, object]:
        self.model.train()
        loss, loss_components, _x_t, _t, _cond, _prediction, batch = self._forward_and_loss(batch)
        model_type = getattr(self.model, "model_type", None)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1

        metrics: dict[str, object] = {"loss": float(loss.detach().cpu())}
        # Components were captured during the forward (pre-backward), so they
        # reflect the loss at the params actually used — no redundant re-forward.
        metrics.update(loss_components)
        # Per-rung shortcut loss: the aggregate `shortcut_direction_loss` mixes
        # all sampled step sizes, so its magnitude swings with whichever `s` was
        # drawn that batch. Re-log it under a per-`s` key (N = round(1/s) steps)
        # so wandb shows one convergence curve per rung. Sparse by design — each
        # step contributes to exactly one series.
        shortcut_s = self._last_shortcut_step_level
        if shortcut_s is not None and "shortcut_direction_loss" in loss_components:
            n_steps = max(1, int(round(1.0 / shortcut_s)))
            metrics[f"shortcut_direction_loss/N{n_steps:03d}"] = loss_components["shortcut_direction_loss"]
        generated_samples = self._maybe_generate_samples(batch=batch, model_type=model_type)
        if generated_samples is not None:
            metrics["generated_samples"] = generated_samples.detach().cpu()
        # Push scalar metrics to wandb every step. Non-scalar entries (e.g.
        # the `generated_samples` tensor) are filtered inside log_metrics; the
        # video panels are pushed separately by `_maybe_generate_samples`.
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step=self.global_step)
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable,
        *,
        max_batches: int | None = None,
        preprocessor: Callable[..., Mapping[str, object]] | None = None,
    ) -> dict[str, float]:
        """Average the training loss (and every component) over held-out batches.

        Runs the exact forward + loss path as training — denoising loss plus any
        configured consistency / Heun terms — with the model in ``eval()`` and
        grads disabled, so the numbers are directly comparable to the ``train``
        metrics. ``global_step`` and parameters are untouched. Returns a dict of
        mean components, each prefixed ``eval_`` (e.g. ``eval_base_loss``,
        ``eval_loss``); empty if the loader yielded nothing.
        """
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        try:
            for raw_batch in loader:
                if max_batches is not None and count >= max_batches:
                    break
                batch = (
                    _call_preprocessor(preprocessor, raw_batch, train=False)
                    if preprocessor is not None
                    else raw_batch
                )
                loss, loss_components, *_ = self._forward_and_loss(batch)
                loss_components["loss"] = float(loss.detach().cpu())
                for key, value in loss_components.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
        finally:
            self.model.train(was_training)

        if count == 0:
            return {}
        return {f"eval_{key}": value / count for key, value in totals.items()}

    def train(
        self,
        loader: Iterable,
        *,
        max_steps: int,
        preprocessor: Callable[..., Mapping[str, object]] | None = None,
        log_every: int = 1,
        on_step: Callable[[int, dict[str, object]], None] | None = None,
        eval_loader: Iterable | None = None,
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
            eval_loader: optional iterable of held-out batches. When given and
                ``config.eval_every_n_steps`` is set, an eval cycle runs at that
                cadence (averaging over ``config.eval_num_batches`` batches) and
                a ``best.pt`` checkpoint is saved whenever ``config.eval_metric``
                improves. A final eval also runs at the end of training.

        Side effects driven by ``config`` (all opt-in, no-ops when unset):
            * ``jsonl_logger`` — every step's metrics and each eval are appended.
            * ``checkpoint_manager`` — a step-tagged checkpoint every
              ``config.checkpoint_every_n_steps`` plus ``best.pt`` / ``final.pt``.

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
                if self.jsonl_logger is not None:
                    self.jsonl_logger.log(metrics, step=self.global_step, split="train")
                if on_step is not None:
                    on_step(self.global_step, dict(metrics))
                if log_every > 0 and self.global_step % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"epoch={epoch} step={self.global_step}/{max_steps} "
                        f"loss={loss_value:.5f} avg_loss={running_loss / running_count:.5f} "
                        f"steps/s={self.global_step / max(elapsed, 1e-6):.2f}"
                    )
                if eval_loader is not None and self._cadence_due(self.config.eval_every_n_steps):
                    self._run_eval_cycle(eval_loader, preprocessor=preprocessor, log=log_every > 0)
                if self._cadence_due(self.config.checkpoint_every_n_steps):
                    self._save_checkpoint(log=log_every > 0)

        elapsed = time.time() - start
        avg_loss = running_loss / max(running_count, 1)
        # Final eval + checkpoint so a run always ends with up-to-date artifacts,
        # regardless of where the last cadence boundary fell. Skip the final eval
        # if the last step already landed on the eval cadence (no double-compute).
        if (
            eval_loader is not None
            and self.config.eval_every_n_steps
            and not self._cadence_due(self.config.eval_every_n_steps)
        ):
            self._run_eval_cycle(eval_loader, preprocessor=preprocessor, log=log_every > 0)
        if self.checkpoint_manager is not None:
            self._save_checkpoint(tag="final", log=log_every > 0)
        if self.jsonl_logger is not None:
            self.jsonl_logger.close()
        if log_every > 0:
            print(f"done. final_avg_loss={avg_loss:.5f} elapsed={elapsed:.1f}s")
        return {
            "final_avg_loss": avg_loss,
            "elapsed_seconds": elapsed,
            "steps": float(self.global_step),
            "epochs": float(epoch),
        }

    def _cadence_due(self, every_n: int | None) -> bool:
        """True when ``global_step`` lands on a positive ``every_n`` boundary."""
        return bool(every_n) and every_n > 0 and self.global_step % every_n == 0

    def _run_eval_cycle(
        self,
        eval_loader: Iterable,
        *,
        preprocessor: Callable[..., Mapping[str, object]] | None,
        log: bool,
    ) -> dict[str, float]:
        """Evaluate, log to wandb/jsonl, and save ``best.pt`` on improvement."""
        eval_metrics = self.evaluate(
            eval_loader,
            max_batches=self.config.eval_num_batches,
            preprocessor=preprocessor,
        )
        if not eval_metrics:
            return {}
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(eval_metrics, step=self.global_step)
        if self.jsonl_logger is not None:
            self.jsonl_logger.log(eval_metrics, step=self.global_step, split="eval")

        metric_key = f"eval_{self.config.eval_metric}"
        value = eval_metrics.get(metric_key)
        if log:
            summary = " ".join(f"{k}={v:.5f}" for k, v in sorted(eval_metrics.items()))
            print(f"  eval step={self.global_step} {summary}")
        if value is not None and value < self.best_eval_metric and self.checkpoint_manager is not None:
            self.best_eval_metric = value
            path = self._save_checkpoint(
                tag="best",
                extra={"best_metric": metric_key, "best_value": value, "eval_metrics": eval_metrics},
                log=False,
            )
            if log:
                print(f"  new best {metric_key}={value:.5f} -> {path}")
        return eval_metrics

    def _save_checkpoint(
        self,
        *,
        tag: str | None = None,
        extra: dict[str, object] | None = None,
        log: bool = False,
    ) -> object | None:
        if self.checkpoint_manager is None:
            return None
        payload_extra = {"best_eval_metric": self.best_eval_metric}
        if extra:
            payload_extra.update(extra)
        path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            extra=payload_extra,
            tag=tag,
        )
        if log and tag != "best":
            print(f"  saved checkpoint -> {path}")
        return path

    def save_checkpoint(
        self, *, tag: str | None = None, extra: dict[str, object] | None = None
    ) -> object | None:
        """Public manual checkpoint save (raises if no checkpoint_manager)."""
        if self.checkpoint_manager is None:
            raise RuntimeError("Trainer has no checkpoint_manager; cannot save a checkpoint.")
        return self._save_checkpoint(tag=tag, extra=extra, log=True)

    def load_checkpoint(self, path: str) -> dict[str, object]:
        """Restore trainable weights + optimizer state and resume bookkeeping
        (``global_step`` and ``best_eval_metric``) from a saved checkpoint."""
        from generative_flow_adapters.training.checkpoint import CheckpointManager

        payload = CheckpointManager.load(path, self.model, self.optimizer)
        self.global_step = int(payload.get("global_step", self.global_step))
        extra = payload.get("extra") or {}
        if "best_eval_metric" in extra:
            self.best_eval_metric = float(extra["best_eval_metric"])
        return payload

    def _needs_shortcut_target(self) -> bool:
        return (
            self.config.shortcut_direction_weight > 0.0
            or self.config.local_consistency_weight > 0.0
            or self.config.multistep_consistency_weight > 0.0
        )

    def _maybe_prepare_shortcut(
        self,
        *,
        batch: Mapping[str, Tensor | object],
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
    ) -> tuple[object | None, Tensor | None]:
        """Resolve step_level + shortcut target at the same (x_t, t) the adapter will see.

        Single supported method (see ``training.shortcut_target_method``); the
        ``two_step`` mode was removed — see thesis-vault
        decided/deprecate-twostep-shortcut-mode. The Heun construction it used
        survives as an orthogonal regularizer (``heun_smoothness_weight``).

        - ``distillation``: paper-faithful self-consistency (Frans et al. 2024,
          eq. 4). At each training step, with probability ``shortcut_anchor_prob``
          run anchor mode (``step_level=1``, no shortcut target — standard
          diffusion loss is the sole supervision). Otherwise sample dyadic
          ``d`` and set ``step_level=2d``; the target is ``(v1 + v2)/2`` from
          two no-grad calls of the *adapted* model at ``step_level=d``, chained
          across one ``d``-sized DDIM micro-step. Anchored at ``d=1`` by the
          standard loss; see thesis-vault risk-shortcut-self-consistency-collapse
          for why the anchor matters.
        """
        # Reset per-step; only the supervised schedule path below sets a value.
        self._last_shortcut_step_level = None
        existing = batch.get("shortcut_target")
        if isinstance(existing, Tensor):
            return cond, existing.to(device=x_t.device, dtype=x_t.dtype)
        if not self._needs_shortcut_target():
            return cond, None

        method = str(self.config.shortcut_target_method).lower()
        step_level_key = str(self.config.extra.get("shortcut_step_level_key", "step_level"))
        batch_size = int(x_t.shape[0])
        device = x_t.device
        dtype = x_t.dtype

        base_model = getattr(self.model, "base_model", None)
        if base_model is None:
            raise RuntimeError(
                "Shortcut target computation requires `model.base_model`. "
                "Either pre-attach `batch['shortcut_target']` or wrap your "
                "adapter in AdaptedModel."
            )

        # Flow-native shortcut self-consistency. The diffusion path below uses a
        # DDIM micro-step with alphas_cumprod, which is only valid for the
        # diffusion parameterisation; rectified-flow models use a straight-line
        # Euler micro-step in normalised t∈[0,1] units instead. Step sizes are
        # dyadic fractions of the trajectory (d = 2^-k), supervising step_level
        # 2d against the average of two d-steps.
        model_type = getattr(self.model, "model_type", None)
        if method == "distillation" and model_type in ("flow", "flow_matching"):
            anchor_prob = float(self.config.extra.get("shortcut_anchor_prob", 0.75))
            do_anchor = (anchor_prob >= 1.0) or (
                anchor_prob > 0.0 and bool(torch.rand((), device="cpu").item() < anchor_prob)
            )

            if self.step_schedule is not None:
                # Paper-faithful path shared with the diffusion branch below:
                # normalised step sizes s ∈ (0,1] drawn from `shortcut_step_schedule`
                # (the same config knob DynamiCrafter/AVID use), instead of the
                # shallow `shortcut_max_log2_steps` ladder. Flow makes this *exact*:
                # the straight-line Euler micro-step has no curvature, so unlike the
                # diffusion branch there are no alphas_cumprod tables and no
                # timestep-jump conversion — `d` is the normalised sigma step (s/2)
                # fed directly to the flow self-consistency target.
                if do_anchor:
                    step_level = torch.full(
                        (batch_size,), float(self.step_schedule.smallest()), device=device, dtype=dtype
                    )
                    return self._inject_step_level(cond, step_level_key, step_level), None
                s_full = self.step_schedule.sample(exclude_smallest=True)
                self._last_shortcut_step_level = float(s_full)
                s_half = s_full / 2.0
                step_level_full = torch.full((batch_size,), float(s_full), device=device, dtype=dtype)
                step_level_half = torch.full((batch_size,), float(s_half), device=device, dtype=dtype)
                new_cond = self._inject_step_level(cond, step_level_key, step_level_full)
                cond_half = self._inject_step_level(cond, step_level_key, step_level_half)
                d_tensor = torch.full((batch_size,), float(s_half), device=device, dtype=dtype)
                with self._autocast():
                    target = compute_self_consistency_target_v_flow(
                        model=self.model, x_t=x_t, t=t, cond_half=cond_half, d=d_tensor,
                        timestep_scale=self._flow_timestep_scale,
                    )
                return new_cond, target

            # Legacy dyadic ladder (no schedule configured): d = 2^-k, k in 1..max.
            max_log2 = int(self.config.extra.get("shortcut_max_log2_steps", 3))
            if do_anchor:
                d_min = 1.0 / (2 ** max_log2)
                step_level = torch.full((batch_size,), d_min, device=device, dtype=dtype)
                return self._inject_step_level(cond, step_level_key, step_level), None
            k = int(torch.randint(1, max_log2 + 1, (), device="cpu").item())
            d_half = 1.0 / (2 ** k)
            d_full = 2.0 * d_half
            self._last_shortcut_step_level = float(d_full)
            step_level_full = torch.full((batch_size,), d_full, device=device, dtype=dtype)
            step_level_half = torch.full((batch_size,), d_half, device=device, dtype=dtype)
            new_cond = self._inject_step_level(cond, step_level_key, step_level_full)
            cond_half = self._inject_step_level(cond, step_level_key, step_level_half)
            d_tensor = torch.full((batch_size,), d_half, device=device, dtype=dtype)
            # Under autocast: the Wan DiT casts its modulation to fp32 and relies
            # on autocast to reconcile that with bf16 weights. This prep runs
            # outside the main forward's autocast block, so wrap it explicitly.
            with self._autocast():
                target = compute_self_consistency_target_v_flow(
                    model=self.model, x_t=x_t, t=t, cond_half=cond_half, d=d_tensor,
                    timestep_scale=self._flow_timestep_scale,
                )
            return new_cond, target

        if method == "distillation":
            anchor_prob = float(self.config.extra.get("shortcut_anchor_prob", 0.75))
            do_anchor = (anchor_prob >= 1.0) or (
                anchor_prob > 0.0
                and bool(torch.rand((), device="cpu").item() < anchor_prob)
            )

            if self.step_schedule is not None:
                # Paper-faithful path: step sizes are normalised s ∈ (0,1] drawn
                # from the configured schedule. Anchor grounds the model at the
                # finest step with the standard loss (the schedule's empirical-
                # velocity end); otherwise sample a larger s and supervise it
                # against the average of two chained calls at s/2 (the rung
                # below), converting s/2 to a timestep jump for the micro-step.
                if do_anchor:
                    step_level = torch.full(
                        (batch_size,), float(self.step_schedule.smallest()), device=device, dtype=dtype
                    )
                    return self._inject_step_level(cond, step_level_key, step_level), None
                s_full = self.step_schedule.sample(exclude_smallest=True)
                self._last_shortcut_step_level = float(s_full)
                s_half = s_full / 2.0
                jump = self.step_schedule.to_timestep_jump(s_half)
                step_level_full = torch.full((batch_size,), float(s_full), device=device, dtype=dtype)
                step_level_half = torch.full((batch_size,), float(s_half), device=device, dtype=dtype)
                new_cond = self._inject_step_level(cond, step_level_key, step_level_full)
                cond_half = self._inject_step_level(cond, step_level_key, step_level_half)
                alphas, scale_arr = self._diffusion_tables(device=device, dtype=dtype)
                target = compute_self_consistency_target_v(
                    model=self.model, x_t=x_t, t=t, cond_half=cond_half, d=jump,
                    alphas_cumprod=alphas, scale_arr=scale_arr,
                    target_kind=self.config.shortcut_consistency_target,
                )
                return new_cond, target

            # Legacy self-consistency (no schedule): dyadic d in raw timesteps,
            # supervise the adapter at step_level=2d against two calls at d.
            if do_anchor:
                step_level = torch.ones(batch_size, device=device, dtype=dtype)
                new_cond = self._inject_step_level(cond, step_level_key, step_level)
                return new_cond, None

            dyadic_max = int(self.config.extra.get("shortcut_step_level_max", 4))
            d_value = self._sample_dyadic_d(dyadic_max=dyadic_max)
            step_level_full = torch.full(
                (batch_size,), float(2 * d_value), device=device, dtype=dtype
            )
            step_level_half = torch.full(
                (batch_size,), float(d_value), device=device, dtype=dtype
            )
            new_cond  = self._inject_step_level(cond, step_level_key, step_level_full)
            cond_half = self._inject_step_level(cond, step_level_key, step_level_half)
            alphas, scale_arr = self._diffusion_tables(device=device, dtype=dtype)
            target = compute_self_consistency_target_v(
                model=self.model, x_t=x_t, t=t, cond_half=cond_half, d=d_value,
                alphas_cumprod=alphas, scale_arr=scale_arr,
                target_kind=self.config.shortcut_consistency_target,
            )
            return new_cond, target

        raise ValueError(
            f"Unknown shortcut_target_method={self.config.shortcut_target_method!r}; "
            "expected 'distillation'. (The 'two_step' mode was removed — see "
            "thesis-vault decided/deprecate-twostep-shortcut-mode.)"
        )

    def _inject_step_level(
        self,
        cond: object | None,
        step_level_key: str,
        step_level: Tensor,
    ) -> object | None:
        """Return ``cond`` with ``step_level_key`` set to ``step_level``.
        Always returns a mapping (or None if the input is None and we have no
        other keys to set, which currently never happens — we always have
        step_level)."""
        if isinstance(cond, Mapping):
            new_cond = dict(cond)
        elif cond is None:
            new_cond = {}
        else:
            new_cond = {"_raw": cond}
        new_cond[step_level_key] = step_level
        return new_cond

    def _sample_dyadic_d(self, *, dyadic_max: int) -> int:
        """Sample ``d`` uniformly from the dyadic set ``{1, 2, 4, ..., D}``
        where ``D = max(2^k)`` such that ``2D <= dyadic_max`` (so the
        supervised ``step_level = 2d`` stays within the user-set cap).

        Returns a Python int — the same ``d`` applies to every sample in the
        batch, matching the paper's per-step batch-split.
        """
        max_d = max(int(dyadic_max) // 2, 1)
        # Largest j such that 2^j <= max_d.
        log2_max = 0
        while (1 << (log2_max + 1)) <= max_d:
            log2_max += 1
        j = int(torch.randint(0, log2_max + 1, (1,)).item())
        return 1 << j

    def _compute_heun_smoothness(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
    ) -> Tensor:
        """Heun-derived velocity-field smoothness regularizer ``L_heun_smooth``.

        Approximates the material derivative of the composed velocity field by a
        finite difference along one DDIM micro-step and penalizes its magnitude:
        ``||v0 - sg(v1)||^2`` where ``v0`` has grad and ``v1`` (the model's
        velocity at the predicted endpoint) is a detached "future-self"
        reference. See thesis-vault theory/heun-smoothness-regularizer.md eq.(S).

        Orthogonal to shortcut training. The step size only sets the *interval*
        over which smoothness is enforced — drawn from ``shortcut_step_schedule``
        if configured, else a unit timestep jump. ``step_level`` is deliberately
        **not** injected: the regularizer sees the same ``cond`` the model sees
        during standard training.
        """
        model_type = getattr(self.model, "model_type", None)
        if model_type != "diffusion":
            raise NotImplementedError(
                "heun_smoothness_weight is implemented for diffusion "
                "(v-prediction) backbones only; flow-matching support is deferred. "
                "See thesis-vault refactor-shortcut-deprecate-twostep-add-heun-"
                "smoothness (patch 1.5)."
            )

        if self.step_schedule is not None:
            jump = self.step_schedule.to_timestep_jump(self.step_schedule.sample())
        else:
            jump = 1
        jump = max(int(jump), 1)

        with self._autocast():
            v0 = self.model(x_t, t, cond)
        v0 = v0.float()

        alphas, scale_arr = self._diffusion_tables(device=x_t.device, dtype=v0.dtype)
        prev_t = (t - jump).clamp_min(0)
        x_prev = ddim_micro_step_v(
            x=x_t.float(), v=v0, t=t, prev_t=prev_t,
            alphas_cumprod=alphas, scale_arr=scale_arr,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                v1 = self.model(x_prev, prev_t, cond)
        finally:
            self.model.train(was_training)

        return LossRegistry.get_consistency_loss("heun_smoothness")(v0, v1.float())

    def _diffusion_tables(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        """Fetch ``alphas_cumprod`` and ``scale_arr`` on the right device/dtype.

        ``scale_arr`` is ``None`` unless DynamiCrafter-style dynamic-rescale is
        on; the DDIM micro-step skips the rescale branch in that case.
        """
        obj = self.diffusion_objective
        alphas = obj.alphas_cumprod.to(device=device, dtype=dtype)
        scale_arr = None
        if obj.use_dynamic_rescale and obj.scale_arr is not None:
            scale_arr = obj.scale_arr.to(device=device, dtype=dtype)
        return alphas, scale_arr

    def generate_samples(
        self,
        batch: Mapping[str, Tensor | object],
        num_inference_steps: int | None = None,
    ) -> Tensor:
        model_type = getattr(self.model, "model_type", None)
        if model_type not in ("diffusion", "flow", "flow_matching"):
            raise ValueError(
                f"Inference sampling is only implemented for diffusion/flow models, got {model_type!r}."
            )
        steps = num_inference_steps or self.config.inference_num_steps
        return self.inference_sampler.sample_from_batch(batch=batch, num_inference_steps=steps)

    def _eval_step_schedule(self) -> list[tuple[int, float | None]]:
        """``(num_steps, step_level)`` pairs for the multi-step eval grid.

        Priority:
        1. An explicit ``training.extra.eval_step_schedule`` (ints, or
           ``{num_steps, step_level}`` mappings) — wins if present.
        2. Otherwise, if a paper-faithful ``shortcut_step_schedule`` with
           discrete levels is set, derive the grid from it: each normalised
           level ``s`` gives ``num_steps = round(1/s)`` and injects
           ``step_level = s`` (so train and eval share one source of truth).

        ``step_level = None`` means "inject nothing" (correct for the frozen
        base / non-shortcut adapters). Empty when neither is configured (eval
        falls back to the single-step path)."""
        raw = self.config.extra.get("eval_step_schedule")
        if raw:
            schedule: list[tuple[int, float | None]] = []
            for entry in raw:
                if isinstance(entry, Mapping):
                    num_steps = int(entry["num_steps"])
                    step_level = entry.get("step_level")
                    step_level = None if step_level is None else float(step_level)
                elif isinstance(entry, (int, float)):
                    num_steps = int(entry)
                    step_level = None
                else:
                    raise ValueError(
                        "eval_step_schedule entries must be ints or mappings with "
                        f"a 'num_steps' key; got {entry!r}."
                    )
                schedule.append((num_steps, step_level))
            return schedule

        if self.step_schedule is not None:
            levels = self.step_schedule.discrete_levels()
            if levels:
                # Largest step first → fewest sampling steps at the top row.
                return [(max(1, int(round(1.0 / s))), float(s)) for s in sorted(levels, reverse=True)]
        return []

    def _maybe_generate_samples(self, batch: Mapping[str, Tensor | object], model_type: str | None) -> Tensor | None:
        if model_type not in ("diffusion", "flow", "flow_matching"):
            return None
        if self.config.inference_every_n_steps is None or self.config.inference_every_n_steps <= 0:
            return None
        if self.global_step % self.config.inference_every_n_steps - 1 != 0:
            return None

        schedule = self._eval_step_schedule()
        if schedule:
            return self._generate_step_size_grid(batch=batch, schedule=schedule)

        target = batch.get("target")
        steps = self.config.inference_num_steps
        # Decode the clean latent z0 for the GT panel on flow batches (where
        # `target` is the velocity); fall back to `target` for diffusion.
        ground_truth = batch.get("x0")
        if not isinstance(ground_truth, Tensor):
            ground_truth = target
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
                target_latents=ground_truth,
                cond=batch.get("cond"),
                step=self.global_step,
            )
            return adapted_samples

        samples = self.generate_samples(batch=batch, num_inference_steps=steps)
        if self.wandb_logger is not None and isinstance(target, Tensor):
            self.wandb_logger.log_videos(
                prediction_latents=samples,
                target_latents=ground_truth,
                cond=batch.get("cond"),
                step=self.global_step,
            )
        return samples

    def _generate_step_size_grid(
        self,
        *,
        batch: Mapping[str, Tensor | object],
        schedule: list[tuple[int, float | None]],
    ) -> Tensor | None:
        """Sample the adapted model (and the frozen base) at every step count
        in ``schedule`` and log a stacked comparison grid.

        For a shortcut model the per-entry ``step_level`` is injected into the
        adapted cond so the adapter knows which multi-step horizon it is
        approximating; the base never sees step_level. All rollouts share one
        noise draw so differences across step counts and against the base are
        purely model-driven. Returns the highest-step-count adapted sample (so
        the caller can keep the existing ``generated_samples`` metric) or
        ``None`` when prerequisites are missing.
        """
        target = batch.get("target")
        if self.wandb_logger is None or not isinstance(target, Tensor):
            return None

        # Flow batches carry target = (noise - z0) (the velocity), so decode the
        # clean latent z0 for the ground-truth panel when the preprocessor
        # provides it; diffusion batches put the clean latent in `target`.
        ground_truth = batch.get("x0")
        if not isinstance(ground_truth, Tensor):
            ground_truth = target
        shared_noise = torch.randn_like(target)
        step_level_key = str(self.config.extra.get("shortcut_step_level_key", "step_level"))
        cond = batch.get("cond")
        base_cond = _strip_adapter_only_keys(cond)

        adapted_by_steps: list[tuple[int, Tensor]] = []
        base_by_steps: list[tuple[int, Tensor]] = []
        for num_steps, step_level in schedule:
            adapted_cond = cond
            if step_level is not None:
                level = torch.full(
                    (target.shape[0],), float(step_level), device=target.device, dtype=target.dtype
                )
                adapted_cond = self._inject_step_level(cond, step_level_key, level)
            adapted = self.inference_sampler.sample_from_batch(
                batch={"target": target, "cond": adapted_cond},
                num_inference_steps=num_steps,
                initial_sample=shared_noise,
            )
            adapted_by_steps.append((num_steps, adapted))

            if self.base_inference_sampler is not None:
                base = self.base_inference_sampler.sample_from_batch(
                    batch={"target": target, "cond": base_cond},
                    num_inference_steps=num_steps,
                    initial_sample=shared_noise,
                )
                base_by_steps.append((num_steps, base))

        self.wandb_logger.log_step_size_grid(
            target_latents=ground_truth,
            adapted_by_steps=adapted_by_steps,
            base_by_steps=base_by_steps or None,
            cond=cond,
            step=self.global_step,
        )
        return adapted_by_steps[-1][1] if adapted_by_steps else None


def _call_preprocessor(
    preprocessor: Callable[..., Mapping[str, object]],
    raw_batch: object,
    train: bool = True,
) -> Mapping[str, object]:
    """Call ``preprocessor(batch, train=train)`` when its signature accepts
    ``train``, otherwise ``preprocessor(batch)``. Lets the trainer accept
    both :class:`DynamiCrafterBatchPreprocessor` (expects the kwarg) and
    plain user lambdas. Signature is inspected instead of catching TypeError
    so real bugs inside the preprocessor aren't silently swallowed. ``train``
    is forwarded so eval batches disable train-only augmentation (e.g. CFG
    condition dropout)."""
    try:
        params = inspect.signature(preprocessor).parameters
    except (TypeError, ValueError):
        params = {}
    if "train" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return preprocessor(raw_batch, train=train)
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
