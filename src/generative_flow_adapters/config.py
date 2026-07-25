from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ConditionSpec:
    key: str
    input_dim: int
    encoder: str = "mlp"
    hidden_dim: int | None = None


@dataclass(slots=True)
class ModelConfig:
    type: str
    provider: str = "dummy"
    prediction_type: str | None = None
    pretrained_model_name_or_path: str | None = None
    subfolder: str | None = None
    feature_dim: int = 64
    hidden_dim: int = 128
    freeze: bool = True
    pass_cond_to_base: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterConfig:
    type: str
    hidden_dim: int = 128
    composition: str = "add"
    gate_bias: float = 0.0
    # Upper clamp on the post-sigmoid mask_mix gate (None = uncapped). Caps
    # the keep-base fraction so the adapter branch retains >= (1-cap) of the
    # gradient — counters the measured gate-saturation trap (uxrst2k5:
    # gate 0.5 -> 0.99 in ~70 steps, adapter grad norm -> 0.003).
    gate_cap: float | None = None
    # AVID pure-adapter warmup: for the first `pretrain_steps` optimizer steps,
    # the composition is bypassed and the loss is on the adapter's STANDALONE
    # prediction (mask=0 for mask_mix; full residual for gated_residual), so the
    # adapter head becomes competent BEFORE the gate is learnable — a different
    # escape from gate-saturation than gate_cap. The gate receives no gradient
    # during warmup and stays at its init. 0 = off. (AVID `pretrain_steps`.)
    pretrain_steps: int = 0
    rank: int = 4
    alpha: float = 1.0
    target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out", "ff", "proj"])
    feature_dim: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConditioningConfig:
    type: str = "action"
    input_dim: int | None = None
    output_dim: int = 128
    modalities: dict[str, int] = field(default_factory=dict)
    conditions: list[ConditionSpec] = field(default_factory=list)
    include_horizon: bool = False
    horizon_dim: int = 16
    include_step_size: bool = False
    step_size_key: str = "step_size"
    drop_condition_prob: float = 0.0
    fuse_mode: str = "concat_mlp"
    context_key: str = "context"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    loss: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    shortcut_direction_weight: float = 0.0
    local_consistency_weight: float = 0.0
    multistep_consistency_weight: float = 0.0
    # Opt-in Heun-derived velocity-field smoothness regularizer. Orthogonal to
    # shortcut training (works on any diffusion run); default 0 keeps behaviour
    # unchanged. See thesis-vault theory/heun-smoothness-regularizer.md.
    heun_smoothness_weight: float = 0.0
    shortcut_target_method: str = "distillation"  # only "distillation" (two_step removed)
    # How the self-consistency target is formed from the two half-step velocities
    # (sibling to `shortcut_target_method`, which selects the teacher path).
    #   "v_average"          — (v1+v2)/2, exact for flow matching but BIASED for
    #                          diffusion v-prediction (kept as the ablation baseline).
    #   "endpoint_inversion" — follow both sub-steps to the real landing x_end and
    #                          invert the DDIM 2d-step for the velocity that
    #                          reproduces it. Exact under v-prediction.
    #   "displacement"       — reserved (Option B); needs the additive few-step
    #                          sampler + head grounding, not yet wired (raises).
    # Diffusion (v-pred) only; flow matching is unbiased so the selector is a no-op
    # there. See thesis-vault decided/shortcut-target-endpoint-vs-v-averaging.
    shortcut_consistency_target: str = "v_average"
    grad_clip_norm: float | None = None
    # Micro-batches accumulated per optimizer step (effective batch = physical
    # batch_size * grad_accum_steps). 1 = no accumulation (previous behaviour).
    grad_accum_steps: int = 1
    # Linear LR warmup from 0 -> learning_rate over this many optimizer steps
    # (not micro-batches). None/0 disables (previous behaviour: flat LR from
    # step 0).
    linear_warmup_steps: int | None = None
    diffusion_timesteps: int = 1000
    diffusion_beta_schedule: str = "linear"
    diffusion_linear_start: float = 8.5e-4
    diffusion_linear_end: float = 1.2e-2
    diffusion_rescale_betas_zero_snr: bool = False
    diffusion_offset_noise_strength: float = 0.0
    inference_every_n_steps: int | None = None
    inference_num_steps: int = 50
    inference_scheduler: str = "ddim"
    # --- run outputs: JSONL metrics, periodic checkpoints, periodic eval -----
    # Root directory for this run's artifacts. When set, metrics.jsonl and a
    # checkpoints/ subdirectory are written here. None disables file outputs
    # (wandb-only / smoke runs) so existing scripts are unaffected.
    output_dir: str | None = None
    # Append every step's scalar metrics (and each eval) to output_dir/metrics.jsonl.
    log_metrics_jsonl: bool = True
    # Save a step-tagged checkpoint every N global steps (None/0 disables).
    checkpoint_every_n_steps: int | None = None
    # Keep only the most recent K step-tagged checkpoints (None keeps all; the
    # best.pt / final.pt checkpoints are never rotated away).
    keep_last_checkpoints: int | None = None
    # Run an eval cycle every N steps (None/0 disables). Each cycle averages the
    # loss over `eval_num_batches` held-out batches and, when it improves on the
    # best `eval_metric` so far, writes a best.pt checkpoint.
    eval_every_n_steps: int | None = None
    eval_num_batches: int = 8
    # Which eval loss component drives best-checkpoint selection. Defaults to the
    # standard denoising loss ("base_loss") rather than the total — self-distilled
    # shortcut/consistency terms can collapse to ~0 without the model improving,
    # so the honest denoising loss is the safer selection signal.
    eval_metric: str = "base_loss"
    # --- quality metrics (paper-standard generative-visual eval) -------------
    # Paired, per-frame-vs-ground-truth metrics (subset of psnr/ssim/lpips/mse)
    # scored on decoded pixels every eval cycle. Cheap and reliable because the
    # world-model eval has aligned ground-truth future frames. Empty -> off.
    # Both the adapted rollout and (when a frozen-base sampler exists) the base
    # rollout are scored, so wandb shows the base-vs-adapted delta. Requires a
    # VAE decoder on the wandb logger + an inference sampler; silently skipped
    # otherwise.
    quality_metrics: list[str] = field(default_factory=list)
    quality_eval_num_batches: int = 4
    # Sampler steps for the quality rollout; None -> inference_num_steps.
    quality_eval_num_steps: int | None = None
    # Distribution metrics (fid/fvd) on their own, rarer cadence — they load
    # Inception/I3D and only mean anything over many samples, so they are kept
    # off the per-cycle path. None/0 (or empty list) disables.
    quality_dist_metrics: list[str] = field(default_factory=list)
    quality_dist_every_n_steps: int | None = None
    quality_dist_num_batches: int = 16
    # --- step-0 baseline eval toggles ----------------------------------------
    # On a fresh run (global_step == 0) a full baseline eval runs *before* the
    # first gradient update so every metric has a genuine "at init" reference.
    # These gate which components of that baseline run, independently of the
    # per-cadence toggles above — the periodic mid-training cadences are
    # unaffected. All default True to preserve the full-baseline behaviour.
    #   * baseline_eval_loss     — the held-out loss eval cycle ("normal eval")
    #   * baseline_eval_inference — the native generation grid ("inference")
    #   * baseline_eval_quality  — the quality metrics (paired psnr/ssim/lpips/
    #                              mse AND the distribution fid/fvd metrics)
    # e.g. set baseline_eval_quality: false to run inference + loss eval at
    # step 0 but skip the expensive FID/FVD (+ paired) quality pass there.
    baseline_eval_loss: bool = True
    baseline_eval_inference: bool = True
    baseline_eval_quality: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataConfig:
    """Dataset construction for the MetaWorld clip pipeline.

    Lives in the config so dataset shape — crucially the frame stride — is set
    in YAML rather than via CLI flags. Training scripts may still pass CLI
    overrides, which win only when explicitly provided.
    """

    # HDF5 source path (None -> the script's --hdf5 / its default supplies it).
    hdf5: str | None = None
    # Which environment(s) / camera angle(s) to train on. str, list of str, or
    # None (= all envs / all cameras). `camera` applies to the camera-split
    # HDF5 layout; it is ignored for the legacy flat layout. Multiple cameras
    # multiply the sample count (same rollout seen from several views).
    env: str | list[str] | None = None
    camera: str | list[str] | None = None
    # Clip length in kept frames. None -> fall back to model.extra.temporal_length.
    window_width: int | None = None
    # Effective fs: the real subsample stride. Lengthens the temporal window
    # (16 contiguous frames are only ~5% of a 300-frame episode) and triggers
    # per-window action-SUM aggregation. See
    # thesis-vault decided/metaworld-frame-stride-load-time.
    frame_stride: int = 1
    # The CONSTANT value fed to the frozen base's fps channel, decoupled from
    # frame_stride above (we do not scale fs with the stride for MetaWorld).
    fs_value: int = 1
    sampling: str = "random"
    caption_mode: str = "empty"
    # Held-out eval source (paired with training.eval_every_n_steps). A separate
    # `eval_hdf5` is a clean leak-free split and wins when set; otherwise
    # `val_fraction` of `hdf5` is held out via a random window-level split
    # (in-distribution — adjacent windows from one episode can straddle it).
    # `val_fraction == 0` and no `eval_hdf5` disables eval. CLI overrides win.
    eval_hdf5: str | None = None
    val_fraction: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    adapter: AdapterConfig
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "default"
    # Filesystem path the config was loaded from (set by ``load_config``); used
    # to upload the raw YAML to the wandb run so every run is reproducible.
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_data = dict(data.get("model", {}))
        adapter_data = dict(data.get("adapter", {}))
        conditioning_data = dict(data.get("conditioning", {}))
        training_data = dict(data.get("training", {}))
        data_data = dict(data.get("data", {}))
        raw_conditions = conditioning_data.get("conditions", [])
        if raw_conditions is None:
            raw_conditions = []
        if not isinstance(raw_conditions, list):
            raise TypeError("conditioning.conditions must be a list when provided.")
        known_conditioning = _split_known(ConditioningConfig, conditioning_data)
        known_conditioning["conditions"] = [
            item if isinstance(item, ConditionSpec) else ConditionSpec(**item) for item in raw_conditions
        ]
        return cls(
            name=data.get("name", "default"),
            model=ModelConfig(**_split_known(ModelConfig, model_data)),
            adapter=AdapterConfig(**_split_known(AdapterConfig, adapter_data)),
            conditioning=ConditioningConfig(**known_conditioning),
            training=TrainingConfig(**_split_known(TrainingConfig, training_data)),
            data=DataConfig(**_split_known(DataConfig, data_data)),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configuration files.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Configuration at {config_path} must be a mapping.")

    config = ExperimentConfig.from_dict(raw)
    config.source_path = str(config_path)
    return config


def _split_known(dataclass_type: type[Any], values: dict[str, Any]) -> dict[str, Any]:
    field_names = {field.name for field in dataclass_type.__dataclass_fields__.values()}
    known = {k: v for k, v in values.items() if k in field_names and k != "extra"}
    explicit_extra = values.get("extra", {})
    if explicit_extra is None:
        explicit_extra = {}
    if not isinstance(explicit_extra, dict):
        raise TypeError(f"Expected '{dataclass_type.__name__}.extra' to be a mapping.")
    implicit_extra = {k: v for k, v in values.items() if k not in field_names}
    known["extra"] = {**explicit_extra, **implicit_extra}
    return known
