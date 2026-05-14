from __future__ import annotations

from typing import Mapping

import torch
from einops import rearrange
from torch import Tensor, nn

from generative_flow_adapters.conditioning.utils.dynamicrafter_conditioning import prepare_dynamicrafter_condition
from generative_flow_adapters.adapters.hidden_states.unicon import _resolve_unet_module
from generative_flow_adapters.adapters.hypernetworks.interface import HyperNetworkAdapterInterface
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.adapters.low_rank.common import (
    LoRAHandle,
    clear_dynamic_lora_parameters,
    inject_hyperalign_lora_layers,
)
from generative_flow_adapters.backbones.dynamicrafter.models.utils_diffusion import timestep_embedding
from generative_flow_adapters.backbones.dynamicrafter.utils.helpers import prob_mask_like


class HyperAlignAdapter(HyperNetworkAdapterInterface):
    """Paper-faithful HyperAlign architecture for U-Net backbones with attention projections."""

    def __init__(
        self,
        rank: int,
        alpha: float,
        target_modules: list[str],
        cond_dim: int,
        hidden_dim: int,
        input_summary_dim: int,
        cond_hidden_dim: int | None = None,
        aux_down_dim: int = 16,
        aux_up_dim: int = 16,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        use_step_level_conditioning: bool = False,
        step_level_key: str = "step_level",
        step_level_hidden_dim: int | None = None,
        use_factorized_memory_position: bool = True,
        update_mode: str = "stepwise",
        piecewise_progress_markers: tuple[float, ...] = (0.0, 0.05, 0.20),
        condition_injection_mode: str = "memory_tokens",
        condition_input_dim: int | None = None,
        condition_cross_attention_heads: int = 4,
        output_composition: str = "add",
        mask_mix_gate_kind: str = "channel",
        output_channels: int | None = None,
    ) -> None:
        del input_summary_dim
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.target_modules = tuple(target_modules)
        self.cond_dim = cond_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.aux_down_dim = aux_down_dim
        self.aux_up_dim = aux_up_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.use_step_level_conditioning = use_step_level_conditioning
        self.step_level_key = step_level_key
        self.step_level_hidden_dim = int(step_level_hidden_dim or (cond_hidden_dim or cond_dim or 128))
        self.use_factorized_memory_position = bool(use_factorized_memory_position)
        self.update_mode = _normalize_update_mode(update_mode)
        self.piecewise_progress_markers = _normalize_piecewise_markers(piecewise_progress_markers)
        self.condition_injection_mode = _normalize_condition_injection_mode(condition_injection_mode)
        self.condition_input_dim = int(condition_input_dim if condition_input_dim is not None else (cond_dim or 0))
        self.condition_cross_attention_heads = int(condition_cross_attention_heads)
        self.output_composition = _normalize_output_composition(output_composition)
        self.mask_mix_gate_kind = _normalize_mask_mix_gate_kind(mask_mix_gate_kind)
        self.output_channels = int(output_channels) if output_channels is not None else None

        expected_hidden_dim = self.rank * (self.aux_down_dim + self.aux_up_dim)
        if self.hidden_dim != expected_hidden_dim:
            raise ValueError(
                "Paper-aligned HyperAlign requires hidden_dim == rank * (aux_down_dim + aux_up_dim)."
            )
        if not self.target_modules:
            raise ValueError("HyperAlign requires at least one target module name.")

        self._handles: list[LoRAHandle] = []
        self._decoder: nn.TransformerDecoder | None = None
        self._factor_head: nn.Linear | None = None
        self._memory_projections: nn.ModuleList | None = None
        self._prepared = False
        self._feature_store = _HyperAlignInputFeatureStore()

        self._cached_hyper_factors: tuple[Tensor, Tensor] | None = None
        self._cached_stage_index: int | None = None
        self._last_timestep_value: int | None = None
        self._last_batch_signature: tuple[int, torch.device, torch.dtype] | None = None

        self.register_buffer("_query_tokens", torch.zeros(1, 1, self.hidden_dim), persistent=False)
        self.register_buffer("_query_positional_encoding", torch.zeros(1, 1, self.hidden_dim), persistent=False)
        self.step_level_embed: nn.Module | None = None
        if self.use_step_level_conditioning:
            if self.cond_dim is None or self.cond_dim <= 0:
                raise ValueError("Step-level conditioning requires a positive cond_dim.")
            self.step_level_embed = nn.Sequential(
                nn.Linear(1, self.step_level_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.step_level_hidden_dim, int(self.cond_dim)),
            )

        self._condition_token_proj: nn.Linear | None = None
        self._condition_type_embed: nn.Parameter | None = None
        self._condition_cross_attn: nn.MultiheadAttention | None = None
        self._condition_cross_attn_norm: nn.LayerNorm | None = None

        self._gate_head_channel: nn.Linear | None = None
        self._gate_head_spatial: nn.Conv2d | None = None
        self._cached_gate: Tensor | None = None

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = _resolve_unet_module(base_model)
        self._feature_store.attach(module)
        self._handles = inject_hyperalign_lora_layers(
            module,
            rank=self.rank,
            alpha=self.alpha,
            aux_down_dim=self.aux_down_dim,
            aux_up_dim=self.aux_up_dim,
            target_modules=self.target_modules,
        )
        if not self._handles:
            raise ValueError("No matching attention projection layers found for paper-aligned HyperAlign injection.")
        self._prepare_architecture()

    def clear_dynamic_parameters(self) -> None:
        clear_dynamic_lora_parameters(self._handles)

    def clear_captured_base_features(self) -> None:
        self._feature_store.clear()

    def reset_trajectory_state(self) -> None:
        self.clear_dynamic_parameters()
        self._cached_hyper_factors = None
        self._cached_stage_index = None
        self._last_timestep_value = None
        self._last_batch_signature = None

    def build_hyper_input(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor:
        del base_output
        prepared_cond = self._prepare_runtime_cond(cond, x_t=x_t)
        cond_embedding = (
            prepared_cond.get("embedding") if isinstance(prepared_cond, Mapping) else None
        )
        memory = self._build_memory_tokens(
            x_t=x_t,
            t=t,
            cond=prepared_cond,
            cond_embedding=cond_embedding,
        )
        decoder = self._require_decoder()
        queries = self._require_query_tokens(memory.shape[0], memory.device, memory.dtype)
        if (
            self.condition_injection_mode == "cross_attention"
            and isinstance(cond_embedding, Tensor)
        ):
            condition_tokens = self._build_condition_tokens(cond_embedding, frames=int(x_t.shape[2]))
            queries = self._inject_condition_via_cross_attention(queries, condition_tokens)
        decoded = decoder(tgt=queries, memory=memory)
        factor_head = self._require_factor_head()
        factor_tokens = factor_head(decoded)
        # Compute the mask-mix gate while we have `decoded` and the captured base
        # features in hand; cache it alongside the factors so forward() can pick
        # it up without recomputation under stepwise/initial/piecewise modes.
        self._cached_gate = self._compute_gate(decoded=decoded, x_t=x_t) if self.output_composition == "mask_mix" else None
        return factor_tokens

    def _compute_gate(self, *, decoded: Tensor, x_t: Tensor) -> Tensor:
        batch_size, _, frames, height, width = x_t.shape
        if self.mask_mix_gate_kind == "channel":
            if self._gate_head_channel is None:
                raise RuntimeError("Channel gate head was not initialized.")
            pooled = decoded.mean(dim=1)
            gate = self._gate_head_channel(pooled.to(dtype=self._gate_head_channel.weight.dtype))
            return gate.view(batch_size, self.output_channels or gate.shape[-1], 1, 1, 1).to(dtype=x_t.dtype)
        if self._gate_head_spatial is None:
            raise RuntimeError("Spatial gate head was not initialized.")
        expected_blocks = len(self._memory_projections) if self._memory_projections is not None else 0
        captured = self._feature_store.get(expected_count=expected_blocks)
        if captured is None or not captured:
            raise RuntimeError(
                "Spatial mask-mix gate requires captured base features from the frozen base pass."
            )
        shallow = captured[0].to(dtype=self._gate_head_spatial.weight.dtype)
        gate_flat = self._gate_head_spatial(shallow)
        if gate_flat.shape[-2:] != (height, width):
            gate_flat = nn.functional.interpolate(
                gate_flat, size=(height, width), mode="bilinear", align_corners=False
            )
        channels = self.output_channels or gate_flat.shape[1]
        gate = gate_flat.view(batch_size, frames, channels, height, width)
        return gate.permute(0, 2, 1, 3, 4).contiguous().to(dtype=x_t.dtype)

    def _prepare_runtime_cond(self, cond: object | None, *, x_t: Tensor) -> object | None:
        if not isinstance(cond, Mapping):
            return cond
        return prepare_dynamicrafter_condition(
            cond,
            x_t=x_t,
            use_step_level_conditioning=self.use_step_level_conditioning,
            step_level_key=self.step_level_key,
            step_level_embed=self.step_level_embed,
        )

    def _build_condition_tokens(self, condition_embedding: Tensor, *, frames: int) -> Tensor:
        if self._condition_token_proj is None or self._condition_type_embed is None:
            raise RuntimeError("Condition injection modules were not initialized.")
        proj_dtype = self._condition_token_proj.weight.dtype
        if condition_embedding.dim() == 2:
            projected = self._condition_token_proj(condition_embedding.to(dtype=proj_dtype))
            tokens = projected.unsqueeze(1)
        elif condition_embedding.dim() == 3:
            projected = self._condition_token_proj(condition_embedding.to(dtype=proj_dtype))
            if projected.shape[1] == frames and frames > 0:
                frame_pos = _sinusoidal_position_embeddings(frames, self.hidden_dim).to(
                    device=projected.device, dtype=projected.dtype
                )
                projected = projected + frame_pos.unsqueeze(0)
            tokens = projected
        else:
            raise ValueError("HyperAlign condition embedding must have rank 2 or 3.")
        type_embed = self._condition_type_embed.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + type_embed.view(1, 1, -1) # The type embedding is like a learanble categorical embedding telling the cross attention to distinguish between memory and codnition tokens

    def _inject_condition_via_cross_attention(
        self, queries: Tensor, condition_tokens: Tensor
    ) -> Tensor:
        if self._condition_cross_attn is None or self._condition_cross_attn_norm is None:
            raise RuntimeError("Condition cross-attention modules were not initialized.")
        attended, _ = self._condition_cross_attn(
            query=queries,
            key=condition_tokens,
            value=condition_tokens,
            need_weights=False,
        )
        return self._condition_cross_attn_norm(queries + attended)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> Tensor | OutputAdapterResult:
        if self.base_model is None:
            raise RuntimeError("HyperAlignAdapter must be attached to a base model before use.")

        # Clear any LoRA factors left over from a previous step so the
        # reference pass (and any base-model call that happens before we
        # explicitly set fresh factors below) runs on the unmodified frozen
        # weights. We deliberately do NOT clear at the end of forward — the
        # factors must remain set through backward so that gradient
        # checkpointing's recomputation produces the same graph as the
        # original forward.
        self.clear_dynamic_parameters()
        reference = base_output if base_output is not None else self.base_model(x_t, t, cond=self._resolve_base_condition(cond))
        hyper_down, hyper_up = self._resolve_hyper_factors(x_t=x_t, t=t, cond=cond)

        for index, handle in enumerate(self._handles):
            handle.wrapped.set_dynamic_hyper_factors(
                down_hyper=hyper_down[:, index, :, :],
                up_hyper=hyper_up[:, index, :, :],
                alpha=self.alpha,
            )

        adapted = self.base_model(x_t, t, cond=self._resolve_base_condition(cond))

        if self.output_composition == "add":
            return adapted - reference
        if self.output_composition == "replace":
            return OutputAdapterResult(adapter_output=adapted, output_kind="prediction")
        gate = self._cached_gate
        if gate is None:
            raise RuntimeError(
                "Mask-mix composition requires a cached gate from build_hyper_input; "
                "ensure _resolve_hyper_factors ran before forward returned."
            )
        return OutputAdapterResult(adapter_output=adapted, output_kind="prediction", gate=gate)

    def _prepare_architecture(self) -> None:
        if self._prepared:
            return

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_decoder_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        self._factor_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self._prepare_condition_injection_modules()
        self._prepared = True

        if self.base_model is None:
            return
        module = _resolve_unet_module(self.base_model)
        encoder_dims = _infer_encoder_feature_dims(module)
        self._memory_projections = nn.ModuleList(nn.Linear(channels, self.hidden_dim) for channels in encoder_dims)
        query_count = len(self._handles)
        self._query_tokens = torch.zeros(1, query_count, self.hidden_dim)
        self._query_positional_encoding = _sinusoidal_position_embeddings(query_count, self.hidden_dim).unsqueeze(0)
        self._prepare_mask_mix_gate_modules(encoder_dims=encoder_dims)

    def _prepare_mask_mix_gate_modules(self, encoder_dims: list[int]) -> None:
        if self.output_composition != "mask_mix":
            return
        if self.output_channels is None or self.output_channels <= 0:
            raise ValueError(
                "HyperAlign mask_mix composition requires a positive output_channels "
                "(set adapter.extra.output_channels or model.extra.latent_channels)."
            )
        if self.mask_mix_gate_kind == "channel":
            head = nn.Linear(self.hidden_dim, self.output_channels)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self._gate_head_channel = head
        elif self.mask_mix_gate_kind == "spatial":
            if not encoder_dims:
                raise ValueError("Cannot build spatial gate head without encoder feature dims.")
            shallow_channels = int(encoder_dims[0])
            conv = nn.Conv2d(shallow_channels, self.output_channels, kernel_size=3, padding=1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self._gate_head_spatial = conv

    def _prepare_condition_injection_modules(self) -> None:
        if self.condition_injection_mode == "none":
            return
        if self.condition_input_dim <= 0:
            raise ValueError(
                "HyperAlign condition injection requires a positive condition_input_dim "
                "(set adapter.extra.condition_input_dim or conditioning.output_dim)."
            )
        self._condition_token_proj = nn.Linear(self.condition_input_dim, self.hidden_dim)
        self._condition_type_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        if self.condition_injection_mode == "cross_attention":
            self._condition_cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.condition_cross_attention_heads,
                batch_first=True,
            )
            self._condition_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

    def _resolve_hyper_factors(self, x_t: Tensor, t: Tensor, cond: object | None) -> tuple[Tensor, Tensor]:
        should_refresh, stage_index = self._should_refresh_hyper_factors(x_t=x_t, t=t)
        if should_refresh: # Check whether recomputation of hypernetwork is necessary
            factor_tokens = self.build_hyper_input(x_t=x_t, t=t, cond=cond)
            hyper_down, hyper_up = self._split_hyper_factors(factor_tokens)
            self._cached_hyper_factors = (hyper_down, hyper_up)
            self._cached_stage_index = stage_index
        if self._cached_hyper_factors is None:
            raise RuntimeError("HyperAlign factor cache was not initialized.")
        self._update_trajectory_state(x_t=x_t, t=t, stage_index=stage_index)
        return self._cached_hyper_factors

    def _should_refresh_hyper_factors(self, x_t: Tensor, t: Tensor) -> tuple[bool, int | None]:
        current_timestep = int(t.detach().max().item())
        stage_index = self._piecewise_stage_index(current_timestep) if self.update_mode == "piecewise" else None
        batch_signature = (int(x_t.shape[0]), x_t.device, x_t.dtype)
        new_trajectory = (
            self._last_timestep_value is None
            or self._last_batch_signature is None
            or batch_signature != self._last_batch_signature
            or current_timestep > self._last_timestep_value
        )
        if new_trajectory or self._cached_hyper_factors is None:
            return True, stage_index
        if self.update_mode == "stepwise":
            return True, stage_index
        if self.update_mode == "initial":
            return False, stage_index
        return stage_index != self._cached_stage_index, stage_index

    def _update_trajectory_state(self, x_t: Tensor, t: Tensor, stage_index: int | None) -> None:
        del stage_index
        self._last_timestep_value = int(t.detach().max().item())
        self._last_batch_signature = (int(x_t.shape[0]), x_t.device, x_t.dtype)

    def _piecewise_stage_index(self, timestep: int) -> int:
        total_timesteps = self._diffusion_total_timesteps()
        start_timestep = max(total_timesteps - 1, 1)
        progress = float(start_timestep - timestep) / float(start_timestep)
        stage_index = 0
        for index, marker in enumerate(self.piecewise_progress_markers):
            if progress >= marker:
                stage_index = index
        return stage_index

    def _diffusion_total_timesteps(self) -> int:
        if self.base_model is None:
            return 1000
        schedule = getattr(self.base_model, "diffusion_schedule_config", None)
        if schedule is None:
            return 1000
        return int(getattr(schedule, "timesteps", 1000))

    def _split_hyper_factors(self, factor_tokens: Tensor) -> tuple[Tensor, Tensor]:
        down_width = self.aux_down_dim * self.rank
        up_width = self.rank * self.aux_up_dim
        down_flat, up_flat = torch.split(factor_tokens, [down_width, up_width], dim=-1)
        hyper_down = down_flat.view(factor_tokens.shape[0], factor_tokens.shape[1], self.aux_down_dim, self.rank)
        hyper_up = up_flat.view(factor_tokens.shape[0], factor_tokens.shape[1], self.rank, self.aux_up_dim)
        return hyper_down, hyper_up

    def _build_memory_tokens(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        cond_embedding: Tensor | None = None,
    ) -> Tensor:
        if self.base_model is None:
            raise RuntimeError("HyperAlignAdapter must be attached to a base model before use.")
        if x_t.dim() != 5:
            raise ValueError("Paper-aligned HyperAlign currently expects video latents shaped [batch, channels, frames, height, width].")
        if not isinstance(cond, Mapping) or "context" not in cond:
            raise TypeError("Paper-aligned HyperAlign expects a mapping condition containing a 'context' tensor.")
        # The frozen base UNet was never trained to interpret the adapter's
        # condition embedding; strip it so the encoder pass (live or captured)
        # reflects only what the base knows. The condition signal reaches the
        # hypernetwork through the condition_injection path instead.
        base_cond = self._strip_condition_embedding(cond)
        encoder_memory = self._build_encoder_memory(x_t=x_t, t=t, base_cond=base_cond)
        if (
            self.condition_injection_mode == "memory_tokens"
            and isinstance(cond_embedding, Tensor)
        ):
            condition_tokens = self._build_condition_tokens(cond_embedding, frames=int(x_t.shape[2]))
            condition_tokens = condition_tokens.to(dtype=encoder_memory.dtype, device=encoder_memory.device)
            return torch.cat([encoder_memory, condition_tokens], dim=1)
        return encoder_memory

    def _build_encoder_memory(self, *, x_t: Tensor, t: Tensor, base_cond: Mapping[str, object]) -> Tensor:
        expected_blocks = len(self._memory_projections) if self._memory_projections is not None else 0
        captured = self._feature_store.get(expected_count=expected_blocks)
        if captured is not None:
            return self._build_memory_from_captured_features(
                captured,
                batch_size=int(x_t.shape[0]),
                frames=int(x_t.shape[2]),
                dtype=x_t.dtype,
            )

        module = _resolve_unet_module(self.base_model)
        emb, context, batch_size = _prepare_hyperalign_runtime(module, x_t, t, base_cond)
        frames = int(x_t.shape[2])
        h = x_t.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, x_t.shape[1], x_t.shape[3], x_t.shape[4])
        h = _apply_channel_concat_for_input_blocks(
            h=h,
            cond=base_cond,
            base_model=self.base_model,
            module=module,
            batch_size=batch_size,
            frames=frames,
        )
        h = h.type(getattr(module, "dtype", x_t.dtype))
        projected_tokens: list[Tensor] = []
        with torch.no_grad():
            for index, block in enumerate(module.input_blocks):
                h = block(h, emb, context=context, batch_size=batch_size)
                pooled = _pool_video_encoder_features(h, batch_size=batch_size, frames=frames)
                projection = self._require_memory_projection(index)
                projected_tokens.append(projection(pooled))
        return self._compose_memory_tokens(projected_tokens)

    @staticmethod
    def _strip_condition_embedding(cond: Mapping[str, object]) -> dict[str, object]:
        stripped = dict(cond)
        stripped.pop("embedding", None)
        return stripped

    def _build_memory_from_captured_features(
        self,
        captured_features: list[Tensor],
        *,
        batch_size: int,
        frames: int,
        dtype: torch.dtype,
    ) -> Tensor:
        projected_tokens: list[Tensor] = []
        for index, block_features in enumerate(captured_features):
            pooled = _pool_video_encoder_features(block_features.to(dtype=dtype), batch_size=batch_size, frames=frames)
            projection = self._require_memory_projection(index)
            projected_tokens.append(projection(pooled))
        return self._compose_memory_tokens(projected_tokens)

    def _compose_memory_tokens(self, projected_tokens: list[Tensor]) -> Tensor:
        """Compose HyperAlign memory tokens with optional factorized positional encoding.

        When `use_factorized_memory_position` is enabled, we treat memory as a 2D grid:
        encoder-layer axis and video-frame axis. We then add
        `layer_position[layer] + frame_position[frame]` before flattening, so tokens from the
        same layer share the same layer identity and tokens at the same frame share the same
        temporal identity. This avoids conflating layer and frame semantics into a single
        flattened token index.
        Why? Without it:
        - Each block gives frames tokens after pooling/projection: [B, F, H].
        - Concatenating all blocks gives [B, L*F, H].
        - One sinusoidal table over length L*F assigns a unique position to every (layer, frame) token.
         --> Tokens from the same layer get different positional vectors.
        """
        if not projected_tokens:
            raise ValueError("HyperAlign requires at least one projected memory token tensor.")
        memory_grid = torch.stack(projected_tokens, dim=1)
        if self.use_factorized_memory_position:
            num_layers = memory_grid.shape[1]
            num_frames = memory_grid.shape[2]
            layer_pos = _sinusoidal_position_embeddings(num_layers, self.hidden_dim).to(
                device=memory_grid.device, dtype=memory_grid.dtype
            )
            frame_pos = _sinusoidal_position_embeddings(num_frames, self.hidden_dim).to(
                device=memory_grid.device, dtype=memory_grid.dtype
            )
            memory_grid = memory_grid + layer_pos.unsqueeze(0).unsqueeze(2) + frame_pos.unsqueeze(0).unsqueeze(1)
            return memory_grid.flatten(1, 2)
        memory = memory_grid.flatten(1, 2)
        flat_position = _sinusoidal_position_embeddings(memory.shape[1], self.hidden_dim).to(
            device=memory.device, dtype=memory.dtype
        )
        return memory + flat_position.unsqueeze(0)

    def _resolve_base_condition(self, cond: object | None) -> object | None:
        if isinstance(cond, dict) and "embedding" in cond:
            base_cond = dict(cond)
            base_cond.pop("embedding", None)
            return base_cond
        return cond

    def _require_decoder(self) -> nn.TransformerDecoder:
        if self._decoder is None:
            raise RuntimeError("HyperAlign decoder was not initialized.")
        return self._decoder

    def _require_factor_head(self) -> nn.Linear:
        if self._factor_head is None:
            raise RuntimeError("HyperAlign factor head was not initialized.")
        return self._factor_head

    def _require_query_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._query_tokens.numel() == 0:
            raise RuntimeError("HyperAlign query tokens were not initialized.")
        base = self._query_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        position = self._query_positional_encoding.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        return base + position

    def _require_memory_projection(self, index: int) -> nn.Linear:
        if self._memory_projections is None:
            raise RuntimeError("HyperAlign memory projections were not initialized.")
        return self._memory_projections[index]


def _infer_encoder_feature_dims(module: nn.Module) -> list[int]:
    input_blocks = getattr(module, "input_blocks", None)
    if input_blocks is None:
        raise ValueError("HyperAlign requires a U-Net backbone exposing input_blocks.")
    dims: list[int] = []
    for block in input_blocks:
        channels = _infer_block_channels(block)
        if channels is not None:
            dims.append(channels)
    if not dims:
        raise ValueError("Unable to infer HyperAlign encoder feature dimensionality.")
    return dims


def _infer_block_channels(block: nn.Module) -> int | None:
    for submodule in block.modules():
        if hasattr(submodule, "out_channels"):
            return int(submodule.out_channels)
    return None


def _pool_video_encoder_features(features: Tensor, batch_size: int, frames: int) -> Tensor:
    if features.dim() != 4:
        raise ValueError("Expected encoder features shaped [(batch * frames), channels, height, width].")
    features = features.reshape(batch_size, frames, features.shape[1], features.shape[2], features.shape[3])
    return features.mean(dim=(3, 4)).to(dtype=features.dtype)


def _apply_channel_concat_for_input_blocks(
    *,
    h: Tensor,
    cond: Mapping[str, object],
    base_model: nn.Module,
    module: nn.Module,
    batch_size: int,
    frames: int,
) -> Tensor:
    """Append concat-conditioning channels to ``h`` so it matches the UNet's
    first input block. Mirrors ``DynamiCrafterUNetWrapper.forward``: prefer
    a real ``cond["concat"]`` tensor; otherwise zero-pad if the wrapper has
    ``allow_dummy_concat_condition=True``; otherwise leave ``h`` untouched.
    """
    concat = cond.get("concat") if isinstance(cond, Mapping) else None
    if isinstance(concat, Tensor):
        flat = concat.to(device=h.device, dtype=h.dtype)
        flat = flat.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, flat.shape[1], flat.shape[3], flat.shape[4])
        return torch.cat([h, flat], dim=1)
    if getattr(base_model, "allow_dummy_concat_condition", False):
        expected = getattr(module, "in_channels", h.shape[1])
        missing = int(expected) - int(h.shape[1])
        if missing > 0:
            dummy = torch.zeros(h.shape[0], missing, *h.shape[2:], device=h.device, dtype=h.dtype)
            return torch.cat([h, dummy], dim=1)
    return h


def _prepare_hyperalign_runtime(
    module: nn.Module,
    x: Tensor,
    timesteps: Tensor,
    cond: Mapping[str, object],
) -> tuple[Tensor, Tensor, int]:
    """Build the encoder-pass `emb` and `context` for the frozen UNet.

    The adapter's condition embedding is intentionally not consumed here —
    it reaches the hypernetwork through the condition_injection path instead.
    Native action-conditioned bases (`module.action_conditioned=True`) still
    receive `cond["act"]` directly through their own action head.
    """
    context = cond.get("context")
    if not isinstance(context, Tensor):
        raise KeyError("Condition mapping must provide a 'context' tensor for HyperAlign.")

    act = cond.get("act")
    fs = cond.get("fs")
    dropout_actions = bool(cond.get("dropout_actions", True))

    batch_size, _, frames, _, _ = x.shape
    t_emb = timestep_embedding(timesteps, module.model_channels, repeat_only=False).type(x.dtype)

    _, context_tokens, _ = context.shape
    if context_tokens == 77 + frames * 16:
        context_text, context_img = context[:, :77, :], context[:, 77:, :]
        context_text = context_text.repeat_interleave(repeats=frames, dim=0)
        context_img = rearrange(context_img, "b (t l) c -> (b t) l c", t=frames)
        context = torch.cat([context_text, context_img], dim=1)
    else:
        context = context.repeat_interleave(repeats=frames, dim=0)

    if not getattr(module, "action_conditioned", False):
        emb = module.time_embed(t_emb)
        emb = emb.repeat_interleave(repeats=frames, dim=0)
    else:
        act_drop_prob = module.action_dropout_prob if dropout_actions else 0.0
        time_emb = module.time_embed(t_emb)
        time_emb = time_emb.repeat_interleave(repeats=frames, dim=0)
        if act is not None:
            act_emb = module.action_embed(act)
            keep_mask = prob_mask_like((act.shape[0],), 1 - act_drop_prob, device=act.device)
            act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, module.null_action_emb)
            cond_emb = rearrange(act_emb, "b t c -> (b t) c")
        else:
            cond_emb = module.null_action_emb.repeat_interleave(repeats=frames * batch_size, dim=0)
        if not getattr(module, "add_act_time_emb", False):
            emb = torch.cat([time_emb, cond_emb], dim=1)
        else:
            emb = time_emb + cond_emb

    if getattr(module, "fs_condition", False):
        if fs is None:
            fs = torch.tensor([module.default_fs] * batch_size, dtype=torch.long, device=x.device)
        fs_emb = timestep_embedding(fs, module.model_channels, repeat_only=False).type(x.dtype)
        fs_embed = module.fps_embedding(fs_emb)
        fs_embed = fs_embed.repeat_interleave(repeats=frames, dim=0)
        emb = emb + fs_embed

    return emb, context, batch_size


class _HyperAlignInputFeatureStore:
    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.input_activations: list[Tensor] = []

    def attach(self, module: nn.Module) -> None:
        self.clear_handles()

        def capture_input(_module, _args, output):
            if isinstance(output, Tensor):
                self.input_activations.append(output.detach())

        for block in module.input_blocks:
            self._handles.append(block.register_forward_hook(capture_input))

    def get(self, expected_count: int) -> list[Tensor] | None:
        if expected_count <= 0 or len(self.input_activations) < expected_count:
            return None
        return self.input_activations[-expected_count:]

    def clear(self) -> None:
        self.input_activations = []

    def clear_handles(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _normalize_output_composition(composition: str) -> str:
    normalized = str(composition).strip().lower()
    aliases = {
        "add": "add",
        "delta": "add",
        "replace": "replace",
        "adapter_only": "replace",
        "mask_mix": "mask_mix",
        "avid_mask_mix": "mask_mix",
        "gated": "mask_mix",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported HyperAlign output_composition: {composition}")
    return aliases[normalized]


def _normalize_mask_mix_gate_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    aliases = {
        "channel": "channel",
        "c": "channel",
        "per_channel": "channel",
        "spatial": "spatial",
        "s": "spatial",
        "per_spatial": "spatial",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported HyperAlign mask_mix_gate_kind: {kind}")
    return aliases[normalized]


def _normalize_condition_injection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "memory_tokens": "memory_tokens",
        "memory": "memory_tokens",
        "tokens": "memory_tokens",
        "b": "memory_tokens",
        "cross_attention": "cross_attention",
        "cross": "cross_attention",
        "ca": "cross_attention",
        "a": "cross_attention",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported HyperAlign condition_injection_mode: {mode}")
    return aliases[normalized]


def _normalize_update_mode(update_mode: str) -> str:
    mode = str(update_mode).strip().lower()
    aliases = {
        "s": "stepwise",
        "stepwise": "stepwise",
        "hyperalign-s": "stepwise",
        "i": "initial",
        "initial": "initial",
        "hyperalign-i": "initial",
        "p": "piecewise",
        "piecewise": "piecewise",
        "hyperalign-p": "piecewise",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported HyperAlign update mode: {update_mode}")
    return aliases[mode]


def _normalize_piecewise_markers(markers: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    normalized = tuple(sorted(float(marker) for marker in markers))
    if not normalized or normalized[0] != 0.0:
        raise ValueError("HyperAlign piecewise progress markers must start at 0.0.")
    if normalized[-1] >= 1.0:
        raise ValueError("HyperAlign piecewise progress markers must be strictly less than 1.0.")
    return normalized


def _sinusoidal_position_embeddings(length: int, dim: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("Sinusoidal position embeddings require an even feature dimension.")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
    embeddings = torch.zeros(length, dim, dtype=torch.float32)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings
