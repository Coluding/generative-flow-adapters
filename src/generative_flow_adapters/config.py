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
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    adapter: AdapterConfig
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    name: str = "default"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_data = dict(data.get("model", {}))
        adapter_data = dict(data.get("adapter", {}))
        conditioning_data = dict(data.get("conditioning", {}))
        training_data = dict(data.get("training", {}))
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
