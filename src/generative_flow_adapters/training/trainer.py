from __future__ import annotations

import contextlib
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
        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

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
            #TODO this is everything very dynamicrafter orientated atm --> for the future we need to adjust it
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

        batch = dict(batch)
        if shortcut_target is not None:
            batch.setdefault("shortcut_target", shortcut_target)
            batch.setdefault("self_consistency_target", shortcut_target)

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

        Two supported methods (see ``training.shortcut_target_method``):

        - ``two_step``: target = ``(base(x_t,t) + base(x_mid, t-1))/2`` where
          ``x_mid`` is one proper DDIM micro-step under the base. Anchored on
          the frozen base; no collapse risk. ``step_level`` is decorative.

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

        if method == "two_step":
            new_cond, _ = self._resolve_step_level(
                cond=cond,
                step_level_key=step_level_key,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            target = self._compute_two_step_target_v(
                base_model=base_model, x_t=x_t, t=t, cond=new_cond
            )
            return new_cond, target

        if method == "distillation":
            anchor_prob = float(self.config.extra.get("shortcut_anchor_prob", 0.75))
            do_anchor = (anchor_prob >= 1.0) or (
                anchor_prob > 0.0
                and bool(torch.rand((), device="cpu").item() < anchor_prob)
            )
            if do_anchor:
                step_level = torch.ones(batch_size, device=device, dtype=dtype)
                new_cond = self._inject_step_level(cond, step_level_key, step_level)
                return new_cond, None

            # Self-consistency mode: sample dyadic d, supervise the adapter at
            # step_level=2d against the average of two adapter calls at d.
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
            target = self._compute_self_consistency_target_v(
                x_t=x_t, t=t, cond_half=cond_half, d=d_value
            )
            return new_cond, target

        raise ValueError(
            f"Unknown shortcut_target_method={self.config.shortcut_target_method!r}; "
            "expected 'two_step' or 'distillation'."
        )

    def _resolve_step_level(
        self,
        *,
        cond: object | None,
        step_level_key: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object | None, Tensor]:
        """Pluck step_level from ``cond`` or sample a new one. Used by
        ``two_step`` mode where ``step_level`` is decorative on the adapter
        but the user may still want to vary it via the YAML config."""
        if isinstance(cond, Mapping):
            existing = cond.get(step_level_key)
            if isinstance(existing, Tensor):
                return cond, existing.to(device=device, dtype=dtype)

        max_step = int(self.config.extra.get("shortcut_step_level_max", 1))
        min_step = int(self.config.extra.get("shortcut_step_level_min", 1))
        max_step = max(max_step, 1)
        min_step = max(min(min_step, max_step), 1)
        if max_step == min_step:
            step_level = torch.full(
                (batch_size,), float(min_step), device=device, dtype=dtype
            )
        else:
            step_level = torch.randint(
                low=min_step, high=max_step + 1, size=(batch_size,), device=device
            ).to(dtype=dtype)

        return self._inject_step_level(cond, step_level_key, step_level), step_level

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

    def _compute_two_step_target_v(
        self,
        *,
        base_model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
    ) -> Tensor:
        """Base-anchored Heun-corrected 2-step target for v-prediction.

        Runs one proper DDIM micro-step under the frozen base and averages the
        two velocities. Drops the bogus ``*step_size`` scaling the old
        implementation did, since both ``v0`` and ``v1`` already live in
        v-space.
        """
        alphas, scale_arr = self._diffusion_tables(device=x_t.device, dtype=x_t.dtype)
        with torch.no_grad():
            v0 = base_model(x_t, t, cond=cond)
            prev_t = (t - 1).clamp_min(0)
            x_mid = _ddim_micro_step_v(
                x=x_t, v=v0, t=t, prev_t=prev_t,
                alphas_cumprod=alphas, scale_arr=scale_arr,
            )
            v1 = base_model(x_mid, prev_t, cond=cond)
        return ((v0 + v1) / 2.0).detach()

    def _compute_self_consistency_target_v(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        cond_half: object | None,
        d: int,
    ) -> Tensor:
        """Paper-faithful self-consistency target (Frans et al. 2024, eq. 4).

        Two no-grad calls of the *adapted* model at ``step_level=d``, chained
        across one ``d``-sized DDIM micro-step. The adapter is its own teacher;
        the frozen base only contributes implicitly through the composition
        inside ``self.model``.
        """
        alphas, scale_arr = self._diffusion_tables(device=x_t.device, dtype=x_t.dtype)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                v1 = self.model(x_t, t, cond_half)
                prev_t = (t - d).clamp_min(0)
                x_mid = _ddim_micro_step_v(
                    x=x_t, v=v1, t=t, prev_t=prev_t,
                    alphas_cumprod=alphas, scale_arr=scale_arr,
                )
                v2 = self.model(x_mid, prev_t, cond_half)
        finally:
            self.model.train(was_training)
        return ((v1 + v2) / 2.0).detach()

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


def _ddim_micro_step_v(
    *,
    x: Tensor,
    v: Tensor,
    t: Tensor,
    prev_t: Tensor,
    alphas_cumprod: Tensor,
    scale_arr: Tensor | None,
) -> Tensor:
    """Single-timestep DDIM update for v-prediction, per-sample t/prev_t.

    Matches :func:`inference.diffusion.DiffusionInferenceSampler._dynamic_rescale_ddim_step`
    semantically, but takes raw schedule tables instead of a diffusers
    ``DDIMScheduler``, accepts (B,) timestep tensors, and is autograd-friendly.

    With ``scale_arr`` provided, applies DynamiCrafter's data-side SNR rescale
    (``pred_x0 *= prev_scale / cur_scale``) between v-decode and recomposition.

    Args:
        x: ``(B, ..., d)`` sample at timestep ``t``.
        v: ``(B, ..., d)`` v-prediction at ``(x, t)`` (same shape as ``x``).
        t: ``(B,)`` long tensor of current timesteps.
        prev_t: ``(B,)`` long tensor of target timesteps (``< t``, clamped at 0).
        alphas_cumprod: ``(T_train,)`` lookup table from the training schedule.
        scale_arr: ``(T_train,)`` dynamic-rescale table, or ``None`` for no rescale.

    Returns:
        ``(B, ..., d)`` sample at timestep ``prev_t``.
    """
    def _gather(table: Tensor, indices: Tensor) -> Tensor:
        idx = indices.to(dtype=torch.long, device=table.device).clamp(min=0, max=table.shape[0] - 1)
        out = table.index_select(0, idx)
        return out.view(-1, *[1] * (x.dim() - 1)).to(device=x.device, dtype=x.dtype)

    alpha_t    = _gather(alphas_cumprod, t)
    alpha_prev = _gather(alphas_cumprod, prev_t)
    sqrt_alpha_t    = alpha_t.sqrt()
    sqrt_sigma_t    = (1.0 - alpha_t).clamp_min(0.0).sqrt()
    sqrt_alpha_prev = alpha_prev.sqrt()
    sqrt_sigma_prev = (1.0 - alpha_prev).clamp_min(0.0).sqrt()

    pred_x0  = sqrt_alpha_t * x - sqrt_sigma_t * v
    pred_eps = sqrt_alpha_t * v + sqrt_sigma_t * x

    if scale_arr is not None:
        cur_scale  = _gather(scale_arr, t)
        prev_scale = _gather(scale_arr, prev_t)
        pred_x0 = pred_x0 * (prev_scale / cur_scale)

    return sqrt_alpha_prev * pred_x0 + sqrt_sigma_prev * pred_eps


def _reshape_step_size_for_base(step_size: Tensor, base_direction: Tensor) -> Tensor:
    if base_direction.dim() == 2:
        if step_size.dim() == 1:
            return step_size.unsqueeze(-1)
        if step_size.dim() == 2 and step_size.shape[-1] == 1:
            return step_size
    if base_direction.dim() == 5:
        if step_size.dim() == 1:
            return step_size[:, None, None, None, None]
        if step_size.dim() == 2:
            return step_size[:, None, :, None, None]
        if step_size.dim() == 3 and step_size.shape[-1] == 1:
            return step_size.permute(0, 2, 1)[:, :, :, None, None]
    while step_size.dim() < base_direction.dim():
        step_size = step_size.unsqueeze(-1)
    return step_size


def _strip_adapter_only_keys(cond: object | None) -> object | None:
    if not isinstance(cond, Mapping):
        return cond
    stripped = dict(cond)
    # The condition encoder's output lives under "embedding"; the frozen base
    # was never trained to consume it, so feeding it to the base-only rollout
    # would be at best ignored and at worst confusing for downstream hooks.
    stripped.pop("embedding", None)
    return stripped
