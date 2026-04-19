from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult


class ShortcutDirectionOutputAdapter(OutputAdapterInterface):
    """Predicts a shortcut direction directly in output space."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int,
        include_x_t: bool = True,
        include_base_direction: bool = True,
        include_step_size: bool = True,
        step_size_key: str = "step_size",
        normalize_base_direction: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.include_x_t = include_x_t
        self.include_base_direction = include_base_direction
        self.include_step_size = include_step_size
        self.step_size_key = step_size_key
        self.normalize_base_direction = normalize_base_direction

        input_dim = cond_dim
        if include_x_t:
            input_dim += feature_dim
        if include_base_direction:
            input_dim += feature_dim
        if include_step_size:
            input_dim += 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        del t, base_output
        if x_t.dim() != 2:
            raise ValueError("ShortcutDirectionOutputAdapter expects x_t shaped [batch, feature_dim].")
        if cond is None or not isinstance(cond, Mapping):
            raise TypeError("ShortcutDirectionOutputAdapter expects mapping conditions with 'embedding'.")

        embedding = cond.get("embedding")
        if not isinstance(embedding, Tensor):
            raise KeyError("ShortcutDirectionOutputAdapter requires cond['embedding'] tensor.")
        if embedding.dim() != 2:
            raise ValueError("ShortcutDirectionOutputAdapter expects cond['embedding'] shaped [batch, cond_dim].")

        chunks: list[Tensor] = [embedding]
        if self.include_x_t:
            chunks.append(x_t)

        if self.include_base_direction:
            base_direction = cond.get("base_direction")
            if not isinstance(base_direction, Tensor):
                raise KeyError("ShortcutDirectionOutputAdapter requires cond['base_direction'] tensor.")
            if base_direction.dim() != 2:
                raise ValueError("ShortcutDirectionOutputAdapter expects cond['base_direction'] shaped [batch, feature_dim].")
            if self.normalize_base_direction:
                base_direction = _normalize_last_dim(base_direction)
            chunks.append(base_direction)

        if self.include_step_size:
            step_size = cond.get(self.step_size_key)
            if step_size is None and self.step_size_key != "horizon":
                step_size = cond.get("horizon")
            if not isinstance(step_size, Tensor):
                raise KeyError(
                    f"ShortcutDirectionOutputAdapter requires cond['{self.step_size_key}'] tensor"
                    " (or 'horizon' fallback)."
                )
            if step_size.dim() == 1:
                step_size = step_size.unsqueeze(-1)
            if step_size.dim() != 2 or step_size.shape[-1] != 1:
                raise ValueError("ShortcutDirectionOutputAdapter expects step size shaped [batch] or [batch, 1].")
            chunks.append(step_size.to(device=x_t.device, dtype=x_t.dtype))

        features = torch.cat(chunks, dim=-1)
        shortcut_direction = self.net(features)
        return OutputAdapterResult(adapter_output=shortcut_direction, output_kind="prediction")


def _normalize_last_dim(value: Tensor, eps: float = 1e-8) -> Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)
