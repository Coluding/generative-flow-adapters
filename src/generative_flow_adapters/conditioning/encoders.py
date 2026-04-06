from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.config import ConditioningConfig


class ConditionEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, cond: object | None) -> Tensor | None:
        raise NotImplementedError


class IdentityConditionEncoder(ConditionEncoder):
    def forward(self, cond: object | None) -> Tensor | None:
        if cond is None or isinstance(cond, Tensor):
            return cond
        raise TypeError("IdentityConditionEncoder expects a tensor or None.")


class MLPConditionEncoder(ConditionEncoder):
    def __init__(self, input_dim: int, output_dim: int, include_horizon: bool, horizon_dim: int) -> None:
        super().__init__()
        self.include_horizon = include_horizon
        self.horizon_dim = horizon_dim
        in_features = input_dim + (1 if include_horizon else 0)
        self.net = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cond: object | None) -> Tensor | None:
        if cond is None:
            return None
        if isinstance(cond, Mapping):
            action = cond.get("action")
            horizon = cond.get("horizon")
            if not isinstance(action, Tensor):
                raise TypeError("Expected cond['action'] to be a tensor.")
            features = action
            if self.include_horizon:
                if horizon is None:
                    raise KeyError("Conditioning config requires a 'horizon' tensor.")
                if horizon.dim() == 1:
                    horizon = horizon.unsqueeze(-1)
                features = torch.cat([features, horizon], dim=-1)
            return self.net(features)
        if not isinstance(cond, Tensor):
            raise TypeError("MLPConditionEncoder expects a tensor or a mapping containing tensors.")
        return self.net(cond)


class MultimodalConditionEncoder(ConditionEncoder):
    def __init__(self, modalities: dict[str, int], output_dim: int, include_horizon: bool) -> None:
        super().__init__()
        self.modalities = modalities
        self.include_horizon = include_horizon
        input_dim = sum(modalities.values()) + (1 if include_horizon else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cond: object | None) -> Tensor | None:
        if cond is None:
            return None
        if not isinstance(cond, Mapping):
            raise TypeError("MultimodalConditionEncoder expects a mapping of modality tensors.")
        chunks: list[Tensor] = []
        for key in self.modalities:
            value = cond.get(key)
            if not isinstance(value, Tensor):
                raise KeyError(f"Missing tensor for modality '{key}'.")
            chunks.append(value)
        if self.include_horizon:
            horizon = cond.get("horizon")
            if not isinstance(horizon, Tensor):
                raise KeyError("Missing 'horizon' tensor for shortcut conditioning.")
            if horizon.dim() == 1:
                horizon = horizon.unsqueeze(-1)
            chunks.append(horizon)
        return self.net(torch.cat(chunks, dim=-1))


def build_condition_encoder(config: ConditioningConfig) -> ConditionEncoder | None:
    if config.input_dim is None and not config.modalities:
        return None
    if config.type in {"action", "goal"}:
        if config.input_dim is None:
            raise ValueError(f"conditioning.input_dim is required for conditioning type '{config.type}'.")
        return MLPConditionEncoder(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            include_horizon=config.include_horizon,
            horizon_dim=config.horizon_dim,
        )
    if config.type == "multimodal":
        if not config.modalities:
            raise ValueError("conditioning.modalities is required for multimodal conditioning.")
        return MultimodalConditionEncoder(
            modalities=config.modalities,
            output_dim=config.output_dim,
            include_horizon=config.include_horizon,
        )
    if config.type == "identity":
        return IdentityConditionEncoder()
    raise ValueError(f"Unsupported conditioning type: {config.type}")
