"""Eval-time modality rollout + prediction-vs-ground-truth visualization.

The multimodal trainer has no eval loop of its own; this evaluator plugs into
``MultiModalTrainer.train(on_step=...)`` and fires at a configurable cadence.

For every output modality flagged ``visualize: true`` in the config (and not the
video stream), it runs a **full reverse-diffusion rollout** — sampling the whole
``(T, ·)`` trajectory from noise with the shared diffusion sampler, conditioned
on the action sequence and (teacher-forced) the ground-truth video via the
``VideoReadout`` (m←video). It then logs predicted-vs-ground-truth: per-dim line
charts for ``vector`` streams (ideal for proprio), frame images for ``map``
streams, and always an ``.npz`` dump so there is an artifact without W&B.

No matplotlib dependency: charts go through ``wandb.plot.line_series``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from generative_flow_adapters.inference.diffusion import DiffusionInferenceSampler
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


class _ModalityDenoiser(nn.Module):
    """Wrap one modality head as a single-tensor denoiser for the sampler.

    The sampler calls ``model(z_t, t, cond)`` and expects one tensor back; the
    modality head needs the (fixed, teacher-forced) ``head_cond`` instead of the
    sampler's ``cond`` slot, so we close over it here.
    """

    def __init__(self, head: nn.Module, head_cond: Tensor | None) -> None:
        super().__init__()
        self.head = head
        self.head_cond = head_cond

    def forward(self, z_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        return self.head(z_t, t, self.head_cond)


class MultiModalEvaluator:
    def __init__(
        self,
        *,
        model: nn.Module,
        objective,
        video_objective,
        prediction_type: str,
        specs: list[OutputModalitySpec],
        codecs: Mapping[str, object] | None,
        eval_batch: Mapping[str, object],
        every_n_steps: int,
        num_inference_steps: int = 50,
        video_cond_t: int = 50,
        num_samples: int = 1,
        wandb_logger: object | None = None,
        out_dir: str | Path | None = None,
        video_name: str = "video",
    ) -> None:
        self.model = model
        self.objective = objective
        self.video_objective = video_objective
        self.prediction_type = prediction_type or "noise"
        self.specs = list(specs)
        self.codecs = dict(codecs or {})
        self.eval_batch = eval_batch
        self.every_n_steps = int(every_n_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.video_cond_t = int(video_cond_t)
        self.num_samples = max(1, int(num_samples))
        self.wandb_logger = wandb_logger
        self.out_dir = Path(out_dir) / "eval" if out_dir else None
        self.video_name = video_name
        self._targets = [
            s for s in self.specs if s.visualize and s.name != video_name and not s.has_frozen_prior
        ]

    def maybe_eval(self, step: int, _metrics: Mapping[str, float] | None = None) -> None:
        if self.every_n_steps > 0 and self._targets and step % self.every_n_steps == 0:
            self.evaluate(step)

    @torch.no_grad()
    def evaluate(self, step: int) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
            targets = {k: _to_device(v, device) for k, v in self.eval_batch["targets"].items()}
            cond = _to_device(self.eval_batch.get("cond"), device)
            head_cond = self._teacher_forced_head_cond(targets, cond)
            for spec in self._targets:
                pred, gt = self._rollout(spec, head_cond, targets)
                self._log(spec, pred, gt, step)
        finally:
            self.model.train(was_training)

    # -- conditioning -------------------------------------------------------- #
    def _teacher_forced_head_cond(self, targets, cond) -> Tensor | None:
        """Build the modality heads' conditioning: action embedding + pooled GT
        video features (m←video), teacher-forcing the ground-truth video at a
        fixed reference timestep so the rollout is conditioned on the true video.
        """
        cond_emb = self.model.condition_encoder(cond) if self.model.condition_encoder is not None else None

        readout = getattr(self.model, "video_readout", None)
        if readout is None:
            return cond_emb

        z0 = targets[self.video_name]
        t_video = torch.full((z0.shape[0],), self.video_cond_t, device=z0.device, dtype=torch.long)
        noise = self.video_objective.sample_noise(z0)
        z0_scaled = self.video_objective.scale_x_start(z0, t_video)
        z_t = self.video_objective.q_sample(x_start=z0_scaled, t=t_video, noise=noise)
        base_output = self.model.base_model(z_t, t_video, cond=cond)
        video_feat = readout(base_output)  # (B, cond_dim)
        if cond_emb is None:
            return video_feat
        while video_feat.dim() < cond_emb.dim():
            video_feat = video_feat.unsqueeze(1)
        return cond_emb + video_feat

    # -- rollout ------------------------------------------------------------- #
    def _rollout(self, spec: OutputModalitySpec, head_cond: Tensor | None, targets) -> tuple[Tensor, Tensor]:
        gt = targets[spec.name]
        denoiser = _ModalityDenoiser(self.model.modality_heads[spec.name], head_cond)
        sampler = DiffusionInferenceSampler(denoiser, self.objective, self.prediction_type, scheduler_name="ddim")
        x0 = sampler.sample(
            shape=tuple(gt.shape),
            cond=None,
            device=gt.device,
            dtype=gt.dtype,
            num_inference_steps=self.num_inference_steps,
            verbose=False,
        )
        codec = self.codecs.get(spec.name)
        pred = codec.decode(x0) if codec is not None else x0
        gt_raw = codec.decode(gt) if codec is not None else gt
        return pred.detach().cpu(), gt_raw.detach().cpu()

    # -- logging ------------------------------------------------------------- #
    def _log(self, spec: OutputModalitySpec, pred: Tensor, gt: Tensor, step: int) -> None:
        n = min(self.num_samples, pred.shape[0])
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.out_dir / f"step{step}_{spec.name}.npz",
                pred=pred[:n].numpy(), gt=gt[:n].numpy(), kind=spec.kind,
            )
        if self.wandb_logger is not None and hasattr(self.wandb_logger, "log_prediction_vs_gt"):
            for s in range(n):
                self.wandb_logger.log_prediction_vs_gt(
                    name=spec.name, pred=pred[s], gt=gt[s], kind=spec.kind, step=step, sample_index=s,
                )


def _to_device(value, device):
    if isinstance(value, Mapping):
        return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in value.items()}
    if isinstance(value, Tensor):
        return value.to(device)
    return value
