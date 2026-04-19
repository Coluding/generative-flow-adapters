from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from generative_flow_adapters.config import ExperimentConfig


@dataclass(slots=True)
class FakeBatchSpec:
    x_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    cond_kind: str
    timestep_max: int
    include_shortcut_targets: bool
    include_horizon: bool = False
    include_step_size: bool = False
    step_size_key: str = "step_size"
    action_dim: int | None = None
    context_shape: tuple[int, ...] | None = None
    modalities: dict[str, int] | None = None
    structured_conditions: dict[str, int] | None = None


class FakeAdapterDataset(Dataset):
    def __init__(self, spec: FakeBatchSpec, length: int = 128) -> None:
        self.spec = spec
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor | object]:
        del index
        x_t = torch.randn(self.spec.x_shape)
        target = torch.randn(self.spec.target_shape)
        t = torch.randint(0, self.spec.timestep_max, (1,), dtype=torch.long).squeeze(0)
        batch: dict[str, Tensor | object] = {"x_t": x_t, "t": t, "target": target}

        cond = _build_condition(self.spec)
        if cond is not None:
            batch["cond"] = cond

        if self.spec.include_shortcut_targets:
            batch["shortcut_target"] = torch.randn(self.spec.target_shape)
            batch["self_consistency_target"] = torch.randn(self.spec.target_shape)

        return batch


def build_fake_dataloader(
    config: ExperimentConfig,
    batch_size: int = 4,
    length: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    spec = infer_fake_batch_spec(config)
    dataset = FakeAdapterDataset(spec=spec, length=length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def infer_fake_batch_spec(config: ExperimentConfig) -> FakeBatchSpec:
    include_shortcut_targets = (
        config.training.local_consistency_weight > 0.0 or config.training.multistep_consistency_weight > 0.0
    )

    if config.model.provider == "dynamicrafter":
        action_dim = int(config.conditioning.extra.get("action_dim", 7))
        context_tokens = int(config.conditioning.extra.get("context_tokens", 77))
        context_dim = int(config.conditioning.extra.get("context_dim", 1024))
        temporal_length = int(config.conditioning.extra.get("temporal_length", 16))
        latent_channels = int(config.model.extra.get("latent_channels", 4))
        latent_height = int(config.model.extra.get("latent_height", 40))
        latent_width = int(config.model.extra.get("latent_width", 64))
        return FakeBatchSpec(
            x_shape=(latent_channels, temporal_length, latent_height, latent_width),
            target_shape=(latent_channels, temporal_length, latent_height, latent_width),
            cond_kind="dynamicrafter",
            timestep_max=1000,
            include_shortcut_targets=include_shortcut_targets,
            action_dim=action_dim,
            context_shape=(context_tokens, context_dim),
            structured_conditions={spec.key: spec.input_dim for spec in config.conditioning.conditions} or None,
        )

    feature_dim = int(config.adapter.feature_dim or config.model.feature_dim)
    cond_type = config.conditioning.type
    modalities = config.conditioning.modalities if config.conditioning.modalities else None
    action_dim = config.conditioning.input_dim
    return FakeBatchSpec(
        x_shape=(feature_dim,),
        target_shape=(feature_dim,),
        cond_kind=cond_type,
        timestep_max=1000,
        include_shortcut_targets=include_shortcut_targets,
        include_horizon=config.conditioning.include_horizon,
        include_step_size=config.conditioning.include_step_size,
        step_size_key=config.conditioning.step_size_key,
        action_dim=action_dim,
        modalities=modalities,
        structured_conditions={spec.key: spec.input_dim for spec in config.conditioning.conditions} or None,
    )


def _build_condition(spec: FakeBatchSpec) -> Tensor | Mapping[str, Tensor] | None:
    if spec.cond_kind == "identity":
        return None

    if spec.cond_kind == "dynamicrafter":
        if spec.context_shape is None:
            raise ValueError("DynamicCrafter fake data requires a context_shape.")
        latent_channels, temporal_length, latent_height, latent_width = spec.x_shape
        cond: dict[str, Tensor] = {
            "context": torch.randn(spec.context_shape),
            "concat": torch.randn(latent_channels, temporal_length, latent_height, latent_width),
        }
        if spec.structured_conditions is not None:
            for key, dim in spec.structured_conditions.items():
                cond[key] = torch.randn(spec.target_shape[1], dim)
        elif spec.action_dim is not None:
            cond["act"] = torch.randn(spec.target_shape[1], spec.action_dim)
        return cond

    if spec.structured_conditions is not None:
        return {name: torch.randn(dim) for name, dim in spec.structured_conditions.items()}

    if spec.cond_kind == "multimodal":
        if spec.modalities is None:
            raise ValueError("Multimodal fake data requires modality dimensions.")
        return {name: torch.randn(dim) for name, dim in spec.modalities.items()}

    if spec.cond_kind in {"action", "goal"}:
        if spec.action_dim is None:
            raise ValueError(f"{spec.cond_kind} fake data requires an action dimension.")
        cond: dict[str, Tensor] = {"action": torch.randn(spec.action_dim)}
        if spec.include_step_size:
            cond[spec.step_size_key] = torch.randint(1, 5, (1,), dtype=torch.float32).squeeze(0)
        if spec.include_horizon:
            cond["horizon"] = torch.randint(1, 5, (1,), dtype=torch.float32).squeeze(0)
        return cond

    return torch.randn(spec.target_shape[0])
