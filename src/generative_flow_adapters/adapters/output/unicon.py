from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult


class UniConOutputAdapter(OutputAdapterInterface):
    """UniCon-style adapter with target and condition branches coupled by joint attention."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        output_kind: str = "prediction",
        output_mask: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for UniConOutputAdapter.")

        self.feature_dim = feature_dim
        self.output_kind = output_kind
        self.output_mask = output_mask
        self.data_proj = nn.Linear(feature_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.layers = nn.ModuleList(UniConBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers))
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.mask_proj = nn.Linear(hidden_dim, 1) if output_mask else None
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.mask_proj is not None:
            nn.init.zeros_(self.mask_proj.weight)
            nn.init.zeros_(self.mask_proj.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        del t
        reference = base_output if base_output is not None else x_t
        data_tokens, restore = _tensor_to_tokens(reference)
        cond_tokens = _build_condition_tokens(cond=cond, num_tokens=data_tokens.shape[1], device=reference.device)
        if cond_tokens is None:
            cond_tokens = torch.zeros(
                data_tokens.shape[0],
                data_tokens.shape[1],
                self.cond_proj.in_features,
                device=reference.device,
                dtype=reference.dtype,
            )

        data_hidden = self.data_proj(data_tokens)
        cond_hidden = self.cond_proj(cond_tokens)

        for layer in self.layers:
            data_hidden, cond_hidden = layer(data_hidden, cond_hidden)

        adapter_output = restore(self.output_proj(data_hidden))
        if self.mask_proj is None:
            return OutputAdapterResult(adapter_output=adapter_output, output_kind=self.output_kind)

        gate = restore(self.mask_proj(data_hidden), channels=1)
        return OutputAdapterResult(adapter_output=adapter_output, output_kind=self.output_kind, gate=gate)


class UniConBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.data_norm = nn.LayerNorm(hidden_dim)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.data_to_cond = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.cond_to_data = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.data_ff = FeedForward(hidden_dim)
        self.cond_ff = FeedForward(hidden_dim)

    def forward(self, data_hidden: Tensor, cond_hidden: Tensor) -> tuple[Tensor, Tensor]:
        data_query = self.data_norm(data_hidden)
        cond_query = self.cond_norm(cond_hidden)
        data_hidden = data_hidden + self.data_to_cond(data_query, cond_query, cond_query, need_weights=False)[0]
        cond_hidden = cond_hidden + self.cond_to_data(cond_query, data_query, data_query, need_weights=False)[0]
        data_hidden = data_hidden + self.data_ff(self.data_norm(data_hidden))
        cond_hidden = cond_hidden + self.cond_ff(self.cond_norm(cond_hidden))
        return data_hidden, cond_hidden


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def _build_condition_tokens(cond: object | None, num_tokens: int, device: torch.device) -> Tensor | None:
    if cond is None:
        return None

    if isinstance(cond, Tensor):
        return _expand_condition(cond, num_tokens)

    if isinstance(cond, Mapping):
        if isinstance(cond.get("condition_tensor"), Tensor):
            return _expand_condition(cond["condition_tensor"], num_tokens)
        if isinstance(cond.get("embedding"), Tensor):
            return _expand_condition(cond["embedding"], num_tokens)
        if isinstance(cond.get("act"), Tensor):
            return _expand_condition(cond["act"], num_tokens)

    raise TypeError("UniConOutputAdapter expects a tensor condition or a mapping with 'embedding', 'act', or 'condition_tensor'.")


def _expand_condition(tensor: Tensor, num_tokens: int) -> Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1).expand(-1, num_tokens, -1)
    if tensor.dim() == 3:
        if tensor.shape[1] == num_tokens:
            return tensor
        pooled = tensor.mean(dim=1, keepdim=True)
        return pooled.expand(-1, num_tokens, -1)
    tokens, _ = _tensor_to_tokens(tensor)
    if tokens.shape[1] == num_tokens:
        return tokens
    pooled = tokens.mean(dim=1, keepdim=True)
    return pooled.expand(-1, num_tokens, -1)


def _tensor_to_tokens(tensor: Tensor) -> tuple[Tensor, callable]:
    if tensor.dim() == 2:
        shape = tensor.shape

        def restore(tokens: Tensor, channels: int | None = None) -> Tensor:
            del channels
            return tokens[:, 0, :]

        return tensor.unsqueeze(1), restore

    if tensor.dim() < 3:
        raise ValueError("UniConOutputAdapter expects at least rank-2 tensors.")

    batch, channels = tensor.shape[:2]
    spatial_shape = tensor.shape[2:]
    tokens = tensor.reshape(batch, channels, -1).transpose(1, 2)

    def restore(token_tensor: Tensor, channels: int | None = None) -> Tensor:
        out_channels = channels if channels is not None else token_tensor.shape[-1]
        return token_tensor.transpose(1, 2).reshape(batch, out_channels, *spatial_shape)

    return tokens, restore
