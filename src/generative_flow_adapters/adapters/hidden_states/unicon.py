from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from einops import rearrange
from torch import Tensor, nn

from generative_flow_adapters.adapters.base import Adapter
from generative_flow_adapters.adapters.common import resolve_condition_embedding
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.backbones.dynamicrafter.models.utils_diffusion import timestep_embedding
from generative_flow_adapters.backbones.dynamicrafter.utils.helpers import prob_mask_like


class UniConHiddenStateAdapter(Adapter):
    """Figure 3(d): decoder-part-focused UniCon design for U-Net backbones."""

    def __init__(
        self,
        cond_dim: int | None = None,
        cond_hidden_dim: int | None = None,
        use_adapter_conditioning: bool = True,
        connector_type: str = "zeroft",
        output_mask: bool = False,
        output_kind: str = "prediction",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.use_adapter_conditioning = use_adapter_conditioning
        self.connector_type = connector_type
        self.output_mask = output_mask
        self.output_kind = output_kind
        self._prepared = False
        self._feature_store = _UNetFeatureStore()

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = _resolve_unet_module(base_model)
        self._feature_store.attach(module)
        self._prepare_from_module(module)

    def clear_captured_base_features(self) -> None:
        self._feature_store.clear()

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        del base_output
        module = self._require_module()
        features = self._feature_store.require()
        emb, context, batch_size = _prepare_unet_runtime(module, x_t, t, cond, adapter=self)

        h = self.middle_connector(features.middle)
        for index, block in enumerate(self.decoder_blocks):
            skip = self.skip_connectors[index](features.input_skips[-(index + 1)])
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb, context=context, batch_size=batch_size)
            h = self.decoder_connectors[index](h, features.output_activations[index]) # USING output features here

        h = h.type(features.final_dtype)
        prediction = self.out_head(h)
        prediction = rearrange(prediction, "(b t) c h w -> b c t h w", b=batch_size)

        if self.mask_head is None:
            return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind)

        gate = self.mask_head(h)
        gate = rearrange(gate, "(b t) c h w -> b c t h w", b=batch_size)
        return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind, gate=gate)

    def _prepare_from_module(self, module: nn.Module) -> None:
        if self._prepared:
            return
        if not all(hasattr(module, name) for name in ("output_blocks", "out", "input_block_chans", "middle_block")):
            raise TypeError("UniConHiddenStateAdapter requires a U-Net style backbone with encoder/decoder blocks.")

        self.decoder_blocks = nn.ModuleList(copy.deepcopy(module.output_blocks))
        self.out_head = copy.deepcopy(module.out)
        self.mask_head = copy.deepcopy(module.out_mask) if self.output_mask and hasattr(module, "out_mask") else None

        middle_channels = _infer_middle_channels(module)
        self.middle_connector = build_connector(self.connector_type, middle_channels)

        skip_channels = list(module.input_block_chans)[::-1]
        self.skip_connectors = nn.ModuleList(build_connector(self.connector_type, channels) for channels in skip_channels)
        output_channels = [_infer_block_channels(block) for block in module.output_blocks]
        self.decoder_connectors = nn.ModuleList(
            build_connector(self.connector_type, channels) for channels in output_channels
        )
        _prepare_adapter_conditioning(self, module)
        self._prepared = True

    def _require_module(self) -> nn.Module:
        if self.base_model is None:
            raise RuntimeError("Adapter must be attached to a base model before use.")
        return _resolve_unet_module(self.base_model)


class ReplaceDecoderHiddenStateAdapter(Adapter):
    """Figure 3(e): replace the diffusion decoder with a trainable copy."""

    def __init__(
        self,
        cond_dim: int | None = None,
        cond_hidden_dim: int | None = None,
        use_adapter_conditioning: bool = True,
        output_mask: bool = False,
        output_kind: str = "prediction",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.use_adapter_conditioning = use_adapter_conditioning
        self.output_mask = output_mask
        self.output_kind = output_kind
        self._prepared = False
        self._feature_store = _UNetFeatureStore()

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = _resolve_unet_module(base_model)
        self._feature_store.attach(module)
        self._prepare_from_module(module)

    def clear_captured_base_features(self) -> None:
        self._feature_store.clear()

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        del base_output
        module = self._require_module()
        features = self._feature_store.require()
        emb, context, batch_size = _prepare_unet_runtime(module, x_t, t, cond, adapter=self)

        h = features.middle
        for index, block in enumerate(self.decoder_blocks):
            h = torch.cat([h, features.input_skips[-(index + 1)]], dim=1)
            h = block(h, emb, context=context, batch_size=batch_size)

        h = h.type(features.final_dtype)
        prediction = self.out_head(h)
        prediction = rearrange(prediction, "(b t) c h w -> b c t h w", b=batch_size)
        if self.mask_head is None:
            return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind)

        gate = self.mask_head(h)
        gate = rearrange(gate, "(b t) c h w -> b c t h w", b=batch_size)
        return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind, gate=gate)

    def _prepare_from_module(self, module: nn.Module) -> None:
        if self._prepared:
            return
        if not all(hasattr(module, name) for name in ("output_blocks", "out")):
            raise TypeError("ReplaceDecoderHiddenStateAdapter requires a U-Net style backbone.")
        self.decoder_blocks = nn.ModuleList(copy.deepcopy(module.output_blocks))
        self.out_head = copy.deepcopy(module.out)
        self.mask_head = copy.deepcopy(module.out_mask) if self.output_mask and hasattr(module, "out_mask") else None
        _prepare_adapter_conditioning(self, module)
        self._prepared = True

    def _require_module(self) -> nn.Module:
        if self.base_model is None:
            raise RuntimeError("Adapter must be attached to a base model before use.")
        return _resolve_unet_module(self.base_model)


class FullSkipLayerControlAdapter(Adapter):
    """Figure 3(c): full-network skip-layer ControlNet-style trainable replica."""

    def __init__(
        self,
        cond_dim: int | None = None,
        cond_hidden_dim: int | None = None,
        use_adapter_conditioning: bool = True,
        connector_type: str = "zeroconv",
        output_mask: bool = False,
        output_kind: str = "prediction",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.use_adapter_conditioning = use_adapter_conditioning
        self.connector_type = connector_type
        self.output_mask = output_mask
        self.output_kind = output_kind
        self._prepared = False
        self._feature_store = _UNetFeatureStore()

    def attach_base_model(self, base_model) -> None:
        super().attach_base_model(base_model)
        module = _resolve_unet_module(base_model)
        self._feature_store.attach(module)
        self._prepare_from_module(module)

    def clear_captured_base_features(self) -> None:
        self._feature_store.clear()

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        del base_output
        module = self._require_module()
        features = self._feature_store.require()
        emb, context, batch_size = _prepare_unet_runtime(module, x_t, t, cond, adapter=self)
        h = rearrange(x_t, "b c frames h w -> (b frames) c h w").type(module.dtype)

        replicated_skips: list[Tensor] = []
        for index, block in enumerate(self.input_blocks):
            h = block(h, emb, context=context, batch_size=batch_size)
            h = self.input_connectors[index](h, features.input_activations[index])
            replicated_skips.append(h)

        h = self.middle_block(h, emb, context=context, batch_size=batch_size)
        h = self.middle_connector(h, features.middle)

        for index, block in enumerate(self.output_blocks):
            h = torch.cat([h, replicated_skips.pop()], dim=1)
            h = block(h, emb, context=context, batch_size=batch_size)
            h = self.output_connectors[index](h, features.output_activations[index])

        h = h.type(features.final_dtype)
        prediction = self.out_head(h)
        prediction = rearrange(prediction, "(b t) c h w -> b c t h w", b=batch_size)
        if self.mask_head is None:
            return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind)

        gate = self.mask_head(h)
        gate = rearrange(gate, "(b t) c h w -> b c t h w", b=batch_size)
        return OutputAdapterResult(adapter_output=prediction, output_kind=self.output_kind, gate=gate)

    def _prepare_from_module(self, module: nn.Module) -> None:
        if self._prepared:
            return
        required = ("input_blocks", "middle_block", "output_blocks", "out", "input_block_chans")
        if not all(hasattr(module, name) for name in required):
            raise TypeError("FullSkipLayerControlAdapter requires a U-Net style backbone.")

        self.input_blocks = nn.ModuleList(copy.deepcopy(module.input_blocks))
        self.middle_block = copy.deepcopy(module.middle_block)
        self.output_blocks = nn.ModuleList(copy.deepcopy(module.output_blocks))
        self.out_head = copy.deepcopy(module.out)
        self.mask_head = copy.deepcopy(module.out_mask) if self.output_mask and hasattr(module, "out_mask") else None

        input_channels = list(module.input_block_chans)
        self.input_connectors = nn.ModuleList(build_connector(self.connector_type, channels) for channels in input_channels)
        self.middle_connector = build_connector(self.connector_type, _infer_middle_channels(module))
        output_channels = [_infer_block_channels(block) for block in module.output_blocks]
        self.output_connectors = nn.ModuleList(
            build_connector(self.connector_type, channels) for channels in output_channels
        )
        _prepare_adapter_conditioning(self, module)
        self._prepared = True

    def _require_module(self) -> nn.Module:
        if self.base_model is None:
            raise RuntimeError("Adapter must be attached to a base model before use.")
        return _resolve_unet_module(self.base_model)


class ZeroConvConnector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, target: Tensor, source: Tensor | None = None) -> Tensor:
        if source is None:
            source = target
        return target + self.proj(source)


class ZeroFTConnector(nn.Module):
    """A zero-initialized feature transform with additive and multiplicative paths."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.add = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.add.weight)
        nn.init.zeros_(self.add.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, target: Tensor, source: Tensor | None = None) -> Tensor:
        if source is None:
            source = target
        return target + self.add(source) + target * self.scale(source)


class _UNetFeatureStore:
    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.input_activations: list[Tensor] = []
        self.output_activations: list[Tensor] = []
        self.middle: Tensor | None = None
        self.final_dtype: torch.dtype = torch.float32

    def attach(self, module: nn.Module) -> None:
        self.clear_handles()

        def capture_input(_module, _args, output):
            self.input_activations.append(output.detach())
            self.final_dtype = output.dtype

        def capture_middle(_module, _args, output):
            self.middle = output.detach()
            self.final_dtype = output.dtype

        def capture_output(_module, _args, output):
            self.output_activations.append(output.detach())
            self.final_dtype = output.dtype

        for block in module.input_blocks:
            self._handles.append(block.register_forward_hook(capture_input))
        self._handles.append(module.middle_block.register_forward_hook(capture_middle))
        for block in module.output_blocks:
            self._handles.append(block.register_forward_hook(capture_output))

    def require(self) -> _UNetFeatures:
        if self.middle is None:
            raise RuntimeError("Base U-Net features were not captured. Run the frozen base model before the adapter.")
        return _UNetFeatures(
            input_activations=self.input_activations,
            input_skips=self.input_activations,
            middle=self.middle,
            output_activations=self.output_activations,
            final_dtype=self.final_dtype,
        )

    def clear(self) -> None:
        self.input_activations = []
        self.output_activations = []
        self.middle = None

    def clear_handles(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _UNetFeatures:
    def __init__(
        self,
        input_activations: list[Tensor],
        input_skips: list[Tensor],
        middle: Tensor,
        output_activations: list[Tensor],
        final_dtype: torch.dtype,
    ) -> None:
        self.input_activations = input_activations
        self.input_skips = input_skips
        self.middle = middle
        self.output_activations = output_activations
        self.final_dtype = final_dtype


def build_connector(connector_type: str, channels: int) -> nn.Module:
    connector_name = connector_type.lower()
    if connector_name in {"zeroconv", "zero_conv"}:
        return ZeroConvConnector(channels)
    if connector_name in {"zeroft", "zero_ft"}:
        return ZeroFTConnector(channels)
    raise ValueError(f"Unsupported connector type: {connector_type}")


def _resolve_unet_module(base_model) -> nn.Module:
    module = getattr(base_model, "module", base_model)
    return module


def _infer_middle_channels(module: nn.Module) -> int:
    middle_first = module.middle_block[0]
    return int(middle_first.channels)


def _infer_block_channels(block: nn.Module) -> int:
    for layer in block:
        if hasattr(layer, "out_channels"):
            return int(layer.out_channels)
    raise ValueError("Unable to infer block channels for connector construction.")


def _prepare_adapter_conditioning(adapter: Adapter, module: nn.Module) -> None:
    cond_dim = getattr(adapter, "cond_dim", None)
    if not getattr(adapter, "use_adapter_conditioning", False) or cond_dim is None or cond_dim <= 0:
        return
    if hasattr(adapter, "condition_proj") and hasattr(adapter, "emb_fuse"):
        return

    emb_dim = _infer_time_embedding_dim(module)
    cond_hidden_dim = int(getattr(adapter, "cond_hidden_dim", 0) or emb_dim)
    adapter.condition_proj = nn.Sequential(
        nn.Linear(cond_dim, cond_hidden_dim),
        nn.SiLU(),
        nn.Linear(cond_hidden_dim, emb_dim),
    )
    adapter.emb_fuse = nn.Sequential(
        nn.Linear(emb_dim * 2, emb_dim),
        nn.SiLU(),
        nn.Linear(emb_dim, emb_dim),
    )
    final = adapter.emb_fuse[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def _infer_time_embedding_dim(module: nn.Module) -> int:
    final_layer = module.time_embed[-1]
    if not hasattr(final_layer, "out_features"):
        raise ValueError("Unable to infer U-Net timestep embedding dimension.")
    return int(final_layer.out_features)


def _prepare_unet_runtime(
    module: nn.Module,
    x: Tensor,
    timesteps: Tensor,
    cond: object | None,
    adapter: Adapter | None = None,
    add_cond_embedding_time: bool = True,
) -> tuple[Tensor, Tensor, int]:
    if not isinstance(cond, Mapping):
        raise TypeError("Paper-aligned UniCon adapters expect a mapping condition with at least 'context'.")
    context = cond.get("context")
    if not isinstance(context, Tensor):
        raise KeyError("Condition mapping must provide a 'context' tensor for the U-Net backbone.")

    act = cond.get("act")
    fs = cond.get("fs")
    dropout_actions = bool(cond.get("dropout_actions", True))

    batch_size, _, frames, _, _ = x.shape
    t_emb = module.time_embed[0].weight.new_zeros((1,))  # placeholder to keep dtype/device tied to module params
    del t_emb
    #TODO: Integrate the timestep embedding in the code
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
        emb = emb.repeat_interleave(repeats=frames, dim=0) # Repeat along frame dimension
    else:
        act_drop_prob = module.action_dropout_prob if dropout_actions else 0.0
        time_emb = module.time_embed(t_emb)
        time_emb = time_emb.repeat_interleave(repeats=frames, dim=0)
        if act is not None:
            act_emb = module.action_embed(act)
            keep_mask = prob_mask_like((act.shape[0],), 1 - act_drop_prob, device=act.device)
            act_emb = torch.where(rearrange(keep_mask, "b -> b 1 1"), act_emb, module.null_action_emb)
            act_emb = rearrange(act_emb, "b t c -> (b t) c")
        else:
            act_emb = module.null_action_emb.repeat_interleave(repeats=frames * batch_size, dim=0)
        if not getattr(module, "add_act_time_emb", False):
            emb = torch.cat([time_emb, act_emb], dim=1)
        else:
            emb = time_emb + act_emb

    if getattr(module, "fs_condition", False):
        if fs is None:
            fs = torch.tensor([module.default_fs] * batch_size, dtype=torch.long, device=x.device)
        fs_emb = timestep_embedding(fs, module.model_channels, repeat_only=False).type(x.dtype)
        fs_embed = module.fps_embedding(fs_emb)
        fs_embed = fs_embed.repeat_interleave(repeats=frames, dim=0)
        emb = emb + fs_embed

    adapter_embedding = _prepare_adapter_embedding(adapter=adapter, cond=cond, batch_size=batch_size, frames=frames, dtype=x.dtype)
    if adapter_embedding is not None: # Merging timestep embedding + condition embedding
        emb = emb + adapter.emb_fuse(torch.cat([emb, adapter_embedding], dim=1))

    return emb, context, batch_size


def _prepare_adapter_embedding(
    adapter: Adapter | None,
    cond: object | None,
    batch_size: int,
    frames: int,
    dtype: torch.dtype,
) -> Tensor | None:
    if adapter is None or not getattr(adapter, "use_adapter_conditioning", False):
        return None
    cond_embedding = resolve_condition_embedding(cond)
    if cond_embedding is None:
        return None
    condition_proj = getattr(adapter, "condition_proj", None)
    if condition_proj is None:
        return None
    if cond_embedding.dim() == 2:
        cond_embedding = cond_embedding.repeat_interleave(repeats=frames, dim=0)
    elif cond_embedding.dim() == 3:
        if cond_embedding.shape[0] != batch_size or cond_embedding.shape[1] != frames:
            raise ValueError("Adapter conditioning tensors with rank 3 must be shaped [batch, frames, cond_dim].")
        cond_embedding = rearrange(cond_embedding, "b t c -> (b t) c")
    else:
        raise ValueError("Adapter conditioning embedding must have rank 2 or 3.")
    return condition_proj(cond_embedding).type(dtype)
