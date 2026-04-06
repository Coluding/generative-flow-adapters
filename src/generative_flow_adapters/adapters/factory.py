from __future__ import annotations

from generative_flow_adapters.adapters.output.dynamicrafter import DynamicCrafterOutputAdapter
from generative_flow_adapters.adapters.hidden_states.unicon import (
    FullSkipLayerControlAdapter,
    ReplaceDecoderHiddenStateAdapter,
    UniConHiddenStateAdapter,
)
from generative_flow_adapters.adapters.hidden_states.residual import ResidualConditioningAdapter
from generative_flow_adapters.adapters.hypernetworks.basic import HyperNetworkAdapter
from generative_flow_adapters.adapters.low_rank.lora import LoRAAdapter
from generative_flow_adapters.adapters.output.affine import AffineOutputAdapter
from generative_flow_adapters.config import AdapterConfig, ConditioningConfig, ModelConfig


def build_adapter(model: ModelConfig, adapter: AdapterConfig, conditioning: ConditioningConfig):
    adapter_type = adapter.type.lower()
    feature_dim = adapter.feature_dim or model.feature_dim
    cond_dim = conditioning.output_dim

    if adapter_type == "output":
        architecture = str(adapter.extra.get("architecture", "affine")).lower()
        if architecture == "affine":
            return AffineOutputAdapter(feature_dim=feature_dim, cond_dim=cond_dim, hidden_dim=adapter.hidden_dim)
        if architecture == "dynamicrafter":
            unet_config_path = adapter.extra.get("unet_config_path")
            if not isinstance(unet_config_path, str) or not unet_config_path:
                raise ValueError("DynamicCrafter output adapter requires adapter.extra.unet_config_path")
            return DynamicCrafterOutputAdapter(
                unet_config_path=unet_config_path,
                checkpoint_path=adapter.extra.get("checkpoint_path"),
                condition_on_base_outputs=bool(adapter.extra.get("condition_on_base_outputs", True)),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                strict_checkpoint=bool(adapter.extra.get("strict_checkpoint", False)),
            )
        if architecture == "unicon":
            return UniConHiddenStateAdapter(
                connector_type=str(adapter.extra.get("connector_type", "zeroft")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                connector_type=str(adapter.extra.get("connector_type", "zeroconv")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        raise ValueError(f"Unsupported output adapter architecture: {architecture}")
    if adapter_type in {"hidden", "hidden_state", "controlnet", "residual"}:
        architecture = str(adapter.extra.get("architecture", "residual")).lower()
        if architecture == "residual":
            return ResidualConditioningAdapter(feature_dim=feature_dim, cond_dim=cond_dim, hidden_dim=adapter.hidden_dim)
        if architecture == "unicon":
            return UniConHiddenStateAdapter(
                connector_type=str(adapter.extra.get("connector_type", "zeroft")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                connector_type=str(adapter.extra.get("connector_type", "zeroconv")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        raise ValueError(f"Unsupported hidden-state adapter architecture: {architecture}")
    if adapter_type in {"hyper", "hypernetwork"}:
        return HyperNetworkAdapter(feature_dim=feature_dim, cond_dim=cond_dim, hidden_dim=adapter.hidden_dim)
    if adapter_type == "lora":
        return LoRAAdapter(rank=adapter.rank, alpha=adapter.alpha, target_modules=adapter.target_modules)
    raise ValueError(f"Unsupported adapter type: {adapter.type}")
