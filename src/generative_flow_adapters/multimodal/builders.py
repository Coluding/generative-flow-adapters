"""Build a multimodal experiment from config.

Mirrors :func:`generative_flow_adapters.training.builders.build_experiment`:
resolves a :class:`MultiModalExperimentConfig` into a frozen base + multi-stream
model + optimizer (and IO helpers). The video stream's adapter is the existing
output-adapter factory on a real backbone, or a lightweight head on the dummy
base; every other modality gets a prediction head, plus a video-adjustment head
when the fusion is compositional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from generative_flow_adapters.adapters.factory import build_adapter
from generative_flow_adapters.conditioning.encoders import build_condition_encoder
from generative_flow_adapters.models.base.factory import build_base_model
from generative_flow_adapters.multimodal.codecs import ModalityCodec, build_codec
from generative_flow_adapters.multimodal.config import MultiModalExperimentConfig
from generative_flow_adapters.multimodal.fusion import LearnedMaskFusion
from generative_flow_adapters.multimodal.model import MultiModalAdaptedModel
from generative_flow_adapters.multimodal.modality_adapter import ModalityPredictionHead
from generative_flow_adapters.multimodal.modality_encoder import ModalityEncoder, VideoReadout


@dataclass(slots=True)
class MultiModalComponents:
    model: MultiModalAdaptedModel
    optimizer: torch.optim.Optimizer
    codecs: dict[str, ModalityCodec] = field(default_factory=dict)


def build_codecs(config: MultiModalExperimentConfig, *, vae=None) -> dict[str, ModalityCodec]:
    """Construct one codec per modality. ``vae`` is injected into ``vae`` codecs."""
    codecs: dict[str, ModalityCodec] = {}
    for spec in config.output_modalities:
        kwargs = dict(spec.codec_kwargs)
        if spec.codec.lower() == "vae":
            if vae is None:
                raise ValueError(f"Modality {spec.name!r} uses a 'vae' codec but no VAE was supplied.")
            kwargs.setdefault("vae", vae)
        codecs[spec.name] = build_codec(spec.codec, **kwargs)
    return codecs


def build_multimodal_experiment(config: MultiModalExperimentConfig, *, vae=None) -> MultiModalComponents:
    base = config.base
    base_model = build_base_model(base.model)
    if base.model.freeze:
        base_model.freeze()

    condition_encoder = build_condition_encoder(base.conditioning)
    cond_dim = base.conditioning.output_dim
    video_spec = config.video_modality

    is_dummy = base.model.provider.lower() == "dummy"
    if is_dummy:
        video_adapter = ModalityPredictionHead(video_spec.feature_shape, cond_dim, hidden_dim=video_spec.hidden_dim)
    else:
        video_adapter = build_adapter(base.model, base.adapter, base.conditioning)

    fusion_kind = str(base.adapter.extra.get("fusion", "compositional")).lower()
    compositional = fusion_kind == "compositional"
    # On a REAL backbone, `compositional` is the contribution (docs/composite
    # (2).png): one AVID output adapter PER modality emits a per-modality video
    # adjustment Δ_m (its tokens injected only into that adapter's cross-attention
    # context), and a learned mask m ∈ ℝ^{n+2} blends {base, action-Δ, Δ_1…Δ_n}.
    # On the DUMMY base there is no cross-attention, so compositional falls back
    # to flat per-modality video-adjuster heads + the same learned mask (cheap
    # substrate check of the fusion math).
    real_compositional = compositional and not is_dummy

    modality_heads: dict[str, torch.nn.Module] = {}
    modality_video_adjusters: dict[str, torch.nn.Module] | None = (
        {} if (compositional and not real_compositional) else None
    )
    modality_video_adapters: dict[str, torch.nn.Module] | None = {} if real_compositional else None
    modality_encoders: dict[str, torch.nn.Module] | None = {} if real_compositional else None
    context_dim = int(base.conditioning.extra.get("context_dim", 1024))
    for spec in config.adapter_modalities:
        modality_heads[spec.name] = ModalityPredictionHead(spec.feature_shape, cond_dim, hidden_dim=spec.hidden_dim)
        if modality_video_adjusters is not None:
            modality_video_adjusters[spec.name] = ModalityPredictionHead(
                video_spec.feature_shape, cond_dim, hidden_dim=video_spec.hidden_dim
            )
        if real_compositional:
            # One AVID adapter per modality (separate weights) + a token encoder
            # feeding only that adapter's context.
            modality_video_adapters[spec.name] = build_adapter(base.model, base.adapter, base.conditioning)
            modality_encoders[spec.name] = ModalityEncoder(
                spec.feature_shape, context_dim=context_dim, hidden_dim=spec.hidden_dim
            )

    video_readout = None
    fusion = None
    if real_compositional:
        latent_channels = int(base.model.extra.get("latent_channels", 4))
        video_readout = VideoReadout(latent_channels, cond_dim)
        # m ∈ ℝ^{n+2}: base + action adapter + one Δ per modality.
        fusion = LearnedMaskFusion(1 + len(config.adapter_modalities))

    model = MultiModalAdaptedModel(
        base_model=base_model,
        video_adapter=video_adapter,
        modality_heads=modality_heads,
        modality_specs=config.output_modalities,
        condition_encoder=condition_encoder,
        fusion=fusion,
        modality_video_adjusters=modality_video_adjusters,
        modality_video_adapters=modality_video_adapters,
        modality_encoders=modality_encoders,
        video_readout=video_readout,
        video_name=video_spec.name,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=base.training.learning_rate, weight_decay=base.training.weight_decay
    )

    codecs = {}
    if any(s.codec.lower() != "vae" for s in config.output_modalities) or vae is not None:
        try:
            codecs = build_codecs(config, vae=vae)
        except ValueError:
            # VAE-using codecs deferred to the real-run builder; non-VAE codecs
            # only is fine for substrate bring-up.
            codecs = {}
    return MultiModalComponents(model=model, optimizer=optimizer, codecs=codecs)
