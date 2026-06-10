"""Config for a multimodal experiment.

Reuses the core config dataclasses (``ModelConfig``, ``AdapterConfig``,
``ConditioningConfig``, ``TrainingConfig``, ``DataConfig``) verbatim by
delegating to :meth:`ExperimentConfig.from_dict`, and adds one new top-level
section — ``output_modalities`` — parsed into :class:`OutputModalitySpec`s.

This keeps the core ``config.py`` byte-unchanged (the parallel-package boundary
from the decision note). A multimodal YAML is a normal experiment YAML plus::

    output_modalities:
      - {name: video,   kind: video,  has_frozen_prior: true, codec: vae, loss_weight: 1.0,
         adapter: {unet_config_path: ...}}
      - {name: proprio, kind: vector, feature_shape: [7],        loss_weight: 0.5}
      - {name: tactile, kind: map,    feature_shape: [2, 64, 64], loss_weight: 0.2}
      - {name: depth,   kind: map,    feature_shape: [1, 64, 64], codec: resize,
         codec_kwargs: {size: [64, 64]}, loss_weight: 0.2}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generative_flow_adapters.config import ExperimentConfig
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


@dataclass(slots=True)
class MultiModalExperimentConfig:
    base: ExperimentConfig
    output_modalities: list[OutputModalitySpec] = field(default_factory=list)

    @property
    def video_modality(self) -> OutputModalitySpec:
        """The single ``has_frozen_prior`` stream (the video the base denoises)."""
        videos = [m for m in self.output_modalities if m.has_frozen_prior]
        if len(videos) != 1:
            raise ValueError(
                f"Expected exactly one has_frozen_prior modality (the video stream); got {len(videos)}."
            )
        return videos[0]

    @property
    def adapter_modalities(self) -> list[OutputModalitySpec]:
        """Every stream predicted from scratch (no frozen prior)."""
        return [m for m in self.output_modalities if not m.has_frozen_prior]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiModalExperimentConfig":
        base = ExperimentConfig.from_dict(data)
        raw_mods = data.get("output_modalities", []) or []
        if not isinstance(raw_mods, list):
            raise TypeError("output_modalities must be a list when provided.")
        mods = [
            m if isinstance(m, OutputModalitySpec) else OutputModalitySpec.from_dict(m)
            for m in raw_mods
        ]
        config = cls(base=base, output_modalities=mods)
        # Validate the frozen-prior invariant eagerly so a bad YAML fails at load.
        if mods:
            config.video_modality  # noqa: B018 — raises if not exactly one
        return config


def load_multimodal_config(path: str | Path) -> MultiModalExperimentConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configuration files.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration at {config_path} must be a mapping.")
    return MultiModalExperimentConfig.from_dict(raw)
