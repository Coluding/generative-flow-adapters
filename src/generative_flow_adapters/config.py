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
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    adapter: AdapterConfig
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    name: str = "default"

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

    return ExperimentConfig.from_dict(raw)


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
