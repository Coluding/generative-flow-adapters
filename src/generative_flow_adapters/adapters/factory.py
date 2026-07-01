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
from generative_flow_adapters.adapters.output.format import normalize_granularity, normalize_output_format
from generative_flow_adapters.adapters.output.output_head import OutputHeadAdapter
from generative_flow_adapters.config import AdapterConfig, ConditioningConfig, ModelConfig
from generative_flow_adapters.adapters.output.shortcut_direction import ShortcutDirectionOutputAdapter


def build_adapter(model: ModelConfig, adapter: AdapterConfig, conditioning: ConditioningConfig):
    adapter_type = adapter.type.lower()
    feature_dim = adapter.feature_dim or model.feature_dim
    cond_dim = conditioning.output_dim

    if adapter_type == "output":
        # The default output adapter is the unified DynamiCrafter-backed one;
        # `extra.backbone` (unet | transformer | mlp) and `extra.output_format`
        # (direct | affine) choose its form. `affine`, `shortcut_direction`, and
        # the hidden-state architectures remain reachable as explicit legacy keys.
        architecture = str(adapter.extra.get("architecture", "dynamicrafter")).lower()
        if architecture == "affine":
            return AffineOutputAdapter(feature_dim=feature_dim, cond_dim=cond_dim, hidden_dim=adapter.hidden_dim)
        if architecture in {"dynamicrafter", "output", "output_v2", "output_head", "head", "default"}:
            return _build_output(model, adapter, conditioning, feature_dim=feature_dim, cond_dim=cond_dim)
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
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                connector_type=str(adapter.extra.get("connector_type", "zeroconv")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
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
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
            )
        if architecture in {"replace_decoder", "replace_diffusion_decoder"}:
            return ReplaceDecoderHiddenStateAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
            )
        if architecture in {"full_skip_controlnet", "skip_layer_controlnet"}:
            return FullSkipLayerControlAdapter(
                cond_dim=cond_dim,
                cond_hidden_dim=adapter.hidden_dim,
                use_adapter_conditioning=bool(adapter.extra.get("use_adapter_conditioning", True)),
                connector_type=str(adapter.extra.get("connector_type", "zeroconv")),
                output_mask=bool(adapter.extra.get("output_mask", False)),
                output_kind=str(adapter.extra.get("output_kind", "prediction")),
                use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
                step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
                step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
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
                step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
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


def _build_output(model: ModelConfig, adapter: AdapterConfig, conditioning: ConditioningConfig, *, feature_dim, cond_dim):
    """Unified output adapter.

    The form is chosen by ``extra``:
    - ``backbone``: ``unet`` (default — the DynamiCrafter 3D UNet), ``transformer``
      (DiT-style over patchified latents), or ``mlp`` (lightweight FiLM head).
    - ``output_format``: ``direct`` (emit the delta) or ``affine`` (emit
      ``scale, shift`` → ``base*scale + shift``, per-channel only); plus
      ``gate_kind`` for the head backbones.

    `unet` reuses ``DynamicCrafterOutputAdapter``; `transformer`/`mlp` use
    ``OutputHeadAdapter``. (The old ``architecture: output_v2`` key still routes
    here.)
    """
    backbone = str(adapter.extra.get("backbone", "unet")).lower()
    output_format = str(adapter.extra.get("output_format", "direct"))

    # Affine is channel-wise only (a dense per-element scale/shift map subsumes
    # the direct delta and collapses the format distinction — see format.py).
    # Reject a leftover affine_granularity so stale configs fail loud rather than
    # silently behaving as channel.
    requested_granularity = adapter.extra.get("affine_granularity")
    if (
        normalize_output_format(output_format) == "affine"
        and requested_granularity is not None
        and normalize_granularity(str(requested_granularity)) != "channel"
    ):
        raise ValueError(
            "output_format='affine' is channel-wise only; affine_granularity="
            f"{requested_granularity!r} is no longer supported (a dense per-element "
            "scale/shift map subsumes the direct delta). Remove affine_granularity "
            "or set it to 'channel'."
        )

    if backbone in {"wan", "wan2.1"}:
        from generative_flow_adapters.adapters.output.wan import Wan21OutputAdapter

        wan_adapter_config_path = adapter.extra.get("wan_adapter_config_path")
        if not isinstance(wan_adapter_config_path, str) or not wan_adapter_config_path:
            raise ValueError("output adapter with backbone='wan' requires adapter.extra.wan_adapter_config_path")
        # cond_dim is the fused embedding width from the condition encoder.
        if not cond_dim or cond_dim <= 0:
            raise ValueError("Wan output adapter needs a positive conditioning.output_dim (the fused embedding width).")
        # The latent channel count is a property of the base VAE (Wan2.1: 16,
        # Wan2.2-TI2V: 48), not the adapter tier file (which only sets width/depth).
        # Thread feature_dim through as in_dim/out_dim so the patch-embed matches
        # the real latents; falls back to the tier file's value when unset.
        latent_channels = int(feature_dim) if feature_dim else None
        return Wan21OutputAdapter.from_config(
            wan_adapter_config_path=wan_adapter_config_path,
            cond_dim=cond_dim,
            in_dim=latent_channels,
            out_dim=latent_channels,
            condition_on_base_outputs=adapter.extra.get("condition_on_base_outputs"),
            use_step_level=adapter.extra.get("use_step_level_conditioning"),
            step_level_key=adapter.extra.get("step_level_key"),
            step_level_transform=adapter.extra.get("step_level_transform"),
        )

    if backbone in {"unet", "dynamicrafter"}:
        unet_config_path = adapter.extra.get("unet_config_path")
        if not isinstance(unet_config_path, str) or not unet_config_path:
            raise ValueError("output adapter with backbone='unet' requires adapter.extra.unet_config_path")
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
            step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
            output_format=output_format,
        )

    # The head operates in latent space, so the channel count must be the true
    # latent width. Prefer an explicit override, then the adapter/model latent
    # channels, and only fall back to the generic feature_dim (which defaults to
    # 64 and would otherwise shadow a 4-channel video latent).
    channels = (
        adapter.extra.get("output_channels")
        or adapter.feature_dim
        or model.extra.get("latent_channels")
        or model.feature_dim
    )
    channels = int(channels) if channels else 0
    if not channels:
        raise ValueError(
            "output adapter with backbone='mlp'/'transformer' needs a channel count: set adapter.extra.output_channels, "
            "adapter.feature_dim, or model.extra.latent_channels."
        )
    return OutputHeadAdapter(
        feature_dim=int(channels),
        cond_dim=cond_dim,
        hidden_dim=adapter.hidden_dim,
        backbone=backbone,
        output_format=output_format,
        gate_kind=str(adapter.extra.get("gate_kind", "none")),
        condition_on_base_outputs=bool(adapter.extra.get("condition_on_base_outputs", True)),
        patch_size=int(adapter.extra.get("patch_size", 2)),
        num_layers=int(adapter.extra.get("num_layers", 4)),
        num_heads=int(adapter.extra.get("num_heads", 8)),
        use_step_level_conditioning=bool(adapter.extra.get("use_step_level_conditioning", False)),
        step_level_key=str(adapter.extra.get("step_level_key", "step_level")),
        step_level_hidden_dim=adapter.extra.get("step_level_hidden_dim"),
        step_level_transform=str(adapter.extra.get("step_level_transform", "linear")),
    )
