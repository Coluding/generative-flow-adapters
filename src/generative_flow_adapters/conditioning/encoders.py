from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.config import ConditionSpec, ConditioningConfig


class ConditionEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, cond: object | None, drop_mask: Tensor | None = None) -> Tensor | None:
        raise NotImplementedError


class IdentityConditionEncoder(ConditionEncoder):
    def forward(self, cond: object | None, drop_mask: Tensor | None = None) -> Tensor | None:
        del drop_mask
        if cond is None or isinstance(cond, Tensor):
            return cond
        raise TypeError("IdentityConditionEncoder expects a tensor or None.")


class MLPConditionEncoder(ConditionEncoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        include_horizon: bool,
        horizon_dim: int,
        include_step_size: bool,
        step_size_key: str,
    ) -> None:
        super().__init__()
        self.include_horizon = include_horizon
        self.horizon_dim = horizon_dim
        self.include_step_size = include_step_size
        self.step_size_key = step_size_key
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))
        in_features = input_dim + (1 if include_horizon or include_step_size else 0)
        self.net = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cond: object | None, drop_mask: Tensor | None = None) -> Tensor | None:
        if cond is None:
            return None
        if isinstance(cond, Mapping):
            action = cond.get("action")
            if not isinstance(action, Tensor):
                raise TypeError("Expected cond['action'] to be a tensor.")
            features = action
            step = _resolve_step_size(cond=cond, include_horizon=self.include_horizon, include_step_size=self.include_step_size, step_size_key=self.step_size_key)
            if step is not None:
                if step.dim() == 1:
                    step = step.unsqueeze(-1)
                features = torch.cat([features, step], dim=-1)
            return _apply_condition_dropout(self.net(features), self.null_embedding, drop_mask)
        if not isinstance(cond, Tensor):
            raise TypeError("MLPConditionEncoder expects a tensor or a mapping containing tensors.")
        return _apply_condition_dropout(self.net(cond), self.null_embedding, drop_mask)


class MultimodalConditionEncoder(ConditionEncoder):
    def __init__(
        self,
        modalities: dict[str, int],
        output_dim: int,
        include_horizon: bool,
        include_step_size: bool,
        step_size_key: str,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.include_horizon = include_horizon
        self.include_step_size = include_step_size
        self.step_size_key = step_size_key
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))
        input_dim = sum(modalities.values()) + (1 if include_horizon or include_step_size else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cond: object | None, drop_mask: Tensor | None = None) -> Tensor | None:
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
        step = _resolve_step_size(cond=cond, include_horizon=self.include_horizon, include_step_size=self.include_step_size, step_size_key=self.step_size_key)
        if step is not None:
            if step.dim() == 1:
                step = step.unsqueeze(-1)
            chunks.append(step)
        return _apply_condition_dropout(self.net(torch.cat(chunks, dim=-1)), self.null_embedding, drop_mask)


class StructuredConditionEncoder(ConditionEncoder):
    def __init__(self, conditions: list[ConditionSpec], output_dim: int, fuse_mode: str, hidden_dim: int) -> None:
        super().__init__()
        self.conditions = conditions
        self.fuse_mode = fuse_mode.lower()
        self.output_dim = output_dim
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))
        self.condition_encoders = nn.ModuleDict()

        if self.fuse_mode != "concat_mlp":
            raise ValueError(f"Unsupported conditioning fuse_mode: {fuse_mode}")

        for spec in conditions:
            if spec.encoder.lower() != "mlp":
                raise ValueError(f"Unsupported condition encoder type: {spec.encoder}")
            branch_hidden = int(spec.hidden_dim or hidden_dim)
            self.condition_encoders[spec.key] = nn.Sequential(
                nn.Linear(spec.input_dim, branch_hidden),
                nn.SiLU(),
                nn.Linear(branch_hidden, output_dim),
            )

        fused_input_dim = len(conditions) * output_dim
        self.fuser = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cond: object | None, drop_mask: Tensor | None = None) -> Tensor | None:
        if cond is None:
            return None
        if not isinstance(cond, Mapping):
            raise TypeError("StructuredConditionEncoder expects a mapping of condition tensors.")

        chunks: list[Tensor] = []
        for spec in self.conditions:
            value = cond.get(spec.key)
            if not isinstance(value, Tensor):
                raise KeyError(f"Missing tensor for structured condition '{spec.key}'.")
            chunks.append(self.condition_encoders[spec.key](value))

        return _apply_condition_dropout(self.fuser(torch.cat(chunks, dim=-1)), self.null_embedding, drop_mask)


def _apply_condition_dropout(encoded: Tensor, null_embedding: Tensor, drop_mask: Tensor | None) -> Tensor:
    if drop_mask is None:
        return encoded
    if drop_mask.dim() != 1 or drop_mask.shape[0] != encoded.shape[0]:
        raise ValueError("Condition drop mask must have shape [batch].")
    null = null_embedding.to(device=encoded.device, dtype=encoded.dtype)
    while null.dim() < encoded.dim():
        null = null.unsqueeze(0)
    mask = drop_mask.to(device=encoded.device).bool()
    while mask.dim() < encoded.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, null, encoded)


def _resolve_step_size(
    cond: Mapping[str, object],
    include_horizon: bool,
    include_step_size: bool,
    step_size_key: str,
) -> Tensor | None:
    if not include_horizon and not include_step_size:
        return None
    if include_step_size:
        step = cond.get(step_size_key)
        if not isinstance(step, Tensor):
            raise KeyError(f"Conditioning config requires a '{step_size_key}' tensor.")
        return step
    horizon = cond.get("horizon")
    if not isinstance(horizon, Tensor):
        raise KeyError("Conditioning config requires a 'horizon' tensor.")
    return horizon


def build_condition_encoder(config: ConditioningConfig) -> ConditionEncoder | None:
    if config.conditions:
        return StructuredConditionEncoder(
            conditions=config.conditions,
            output_dim=config.output_dim,
            fuse_mode=config.fuse_mode,
            hidden_dim=max(config.output_dim, max(int(spec.hidden_dim or config.output_dim) for spec in config.conditions)),
        )
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
            include_step_size=config.include_step_size,
            step_size_key=config.step_size_key,
        )
    if config.type == "multimodal":
        if not config.modalities:
            raise ValueError("conditioning.modalities is required for multimodal conditioning.")
        return MultimodalConditionEncoder(
            modalities=config.modalities,
            output_dim=config.output_dim,
            include_horizon=config.include_horizon,
            include_step_size=config.include_step_size,
            step_size_key=config.step_size_key,
        )
    if config.type == "identity":
        return IdentityConditionEncoder()
    raise ValueError(f"Unsupported conditioning type: {config.type}")
