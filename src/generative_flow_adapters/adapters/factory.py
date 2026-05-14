from __future__ import annotations

from generative_flow_adapters.adapters.output.dynamicrafter import DynamicCrafterOutputAdapter
from generative_flow_adapters.adapters.hidden_states.unicon import (
    FullSkipLayerControlAdapter,
    ReplaceDecoderHiddenStateAdapter,
    UniConHiddenStateAdapter,
)
from generative_flow_adapters.adapters.hidden_states.residual import ResidualConditioningAdapter
from generative_flow_adapters.adapters.hypernetworks import HyperAlignAdapter, SimpleHyperLoRAAdapter
from generative_flow_adapters.adapters.low_rank.common import PAPER_HYPERALIGN_TARGET_MODULES
from generative_flow_adapters.adapters.low_rank.lora import LoRAAdapter
from generative_flow_adapters.adapters.output.affine import AffineOutputAdapter
from generative_flow_adapters.config import AdapterConfig, ConditioningConfig, ModelConfig
from generative_flow_adapters.adapters.output.shortcut_direction import ShortcutDirectionOutputAdapter


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
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                allow_dummy_concat_condition=bool(
                    adapter.extra.get("allow_dummy_concat_condition", model.extra.get("allow_dummy_concat_condition", False))
                ),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
            )
        if architecture in {"shortcut_direction", "shortcut"}:
            return ShortcutDirectionOutputAdapter(
                feature_dim=feature_dim,
                cond_dim=cond_dim,
                hidden_dim=adapter.hidden_dim,
                include_x_t=bool(adapter.extra.get("include_x_t", True)),
                include_base_direction=bool(adapter.extra.get("include_base_direction", True)),
                include_step_size=bool(adapter.extra.get("include_step_size", True)),
                step_size_key=str(adapter.extra.get("step_size_key", conditioning.step_size_key)),
                normalize_base_direction=bool(adapter.extra.get("normalize_base_direction", True)),
            )
        if architecture == "unicon":
            return UniConHiddenStateAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                connector_type=str(adapter.extra.get("connector_type", "zeroft")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
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
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                connector_type=str(adapter.extra.get("connector_type", "zeroft")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                connector_type=str(adapter.extra.get("connector_type", "zeroconv")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
            )
        raise ValueError(f"Unsupported hidden-state adapter architecture: {architecture}")
    if adapter_type in {"hyper", "hypernetwork"}:
        architecture = str(adapter.extra.get("architecture", "hyperalign")).lower()
        if model.provider.lower() == "dynamicrafter":
            input_summary_dim = int(model.extra.get("latent_channels", 4))
        else:
            input_summary_dim = int(adapter.feature_dim or model.feature_dim)
        if architecture in {"hyper_lora_simple", "hyper_lora", "stepwise_lora", "simple"}:
            return SimpleHyperLoRAAdapter(
                rank=adapter.rank,
                alpha=adapter.alpha,
                target_modules=adapter.target_modules,
                cond_dim=cond_dim,
                hidden_dim=adapter.hidden_dim,
                input_summary_dim=input_summary_dim,
                use_base_output_summary=bool(adapter.extra.get("use_base_output_summary", False)),
                include_step_size=bool(adapter.extra.get("include_step_size", conditioning.include_step_size)),
                step_size_key=str(adapter.extra.get("step_size_key", conditioning.step_size_key)),
            )
        if architecture in {"hyperalign", "paper"}:
            if model.provider.lower() != "dynamicrafter":
                raise ValueError("Paper-aligned HyperAlign currently requires the dynamicrafter backbone provider.")
            aux_down_dim = int(adapter.extra.get("aux_down_dim", 16))
            aux_up_dim = int(adapter.extra.get("aux_up_dim", 16))
            target_modules = tuple(adapter.extra.get("target_modules", PAPER_HYPERALIGN_TARGET_MODULES))
            output_channels = adapter.extra.get("output_channels", model.extra.get("latent_channels"))
            return HyperAlignAdapter(
                rank=adapter.rank,
                alpha=adapter.alpha,
                target_modules=list(target_modules),
                cond_dim=cond_dim,
                hidden_dim=adapter.hidden_dim,
                input_summary_dim=input_summary_dim,
                cond_hidden_dim=adapter.extra.get("adapter_condition_hidden_dim"),
                aux_down_dim=aux_down_dim,
                aux_up_dim=aux_up_dim,
                num_decoder_layers=int(adapter.extra.get("num_decoder_layers", 4)),
                num_decoder_heads=int(adapter.extra.get("num_decoder_heads", 8)),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                use_factorized_memory_position=bool(adapter.extra.get("use_factorized_memory_position", True)),
                update_mode=str(adapter.extra.get("update_mode", "stepwise")),
                piecewise_progress_markers=tuple(adapter.extra.get("piecewise_progress_markers", (0.0, 0.05, 0.20))),
                condition_injection_mode=str(adapter.extra.get("condition_injection_mode", "memory_tokens")),
                condition_input_dim=adapter.extra.get("condition_input_dim"),
                condition_cross_attention_heads=int(adapter.extra.get("condition_cross_attention_heads", 4)),
                output_composition=adapter.composition,
                mask_mix_gate_kind=str(adapter.extra.get("mask_mix_gate_kind", "channel")),
                output_channels=output_channels,
            )
        raise ValueError(f"Unsupported hypernetwork adapter architecture: {architecture}")
    if adapter_type == "lora":
        return LoRAAdapter(rank=adapter.rank, alpha=adapter.alpha, target_modules=adapter.target_modules)
    raise ValueError(f"Unsupported adapter type: {adapter.type}")
