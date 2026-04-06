from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    include_horizon: bool = False
    horizon_dim: int = 16
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    loss: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    local_consistency_weight: float = 0.0
    multistep_consistency_weight: float = 0.0
    grad_clip_norm: float | None = None
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
        return cls(
            name=data.get("name", "default"),
            model=ModelConfig(**_split_known(ModelConfig, model_data)),
            adapter=AdapterConfig(**_split_known(AdapterConfig, adapter_data)),
            conditioning=ConditioningConfig(**_split_known(ConditioningConfig, conditioning_data)),
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
    known["extra"] = {k: v for k, v in values.items() if k not in field_names}
    return known
