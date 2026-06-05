"""Multimodal compositional output adapters.

Parallel line to the single-stream adapter framework: a frozen diffusion base
plus trainable output adapters that predict *coupled* future modalities
(video + proprio + depth + tactile + ...). Imports the existing primitives
(adapters, conditioning, base-model loading, losses) and adds a multi-stream
model + trainer + configs without modifying ``AdaptedModel`` or ``trainer.py``.

See thesis-vault ``50_Decisions/open/multimodal-adapter-broadening.md``.
"""

from __future__ import annotations

from generative_flow_adapters.multimodal.codecs import (
    IdentityCodec,
    ModalityCodec,
    ResizeCodec,
    VaeCodec,
    build_codec,
)
from generative_flow_adapters.multimodal.config import (
    MultiModalExperimentConfig,
    load_multimodal_config,
)
from generative_flow_adapters.multimodal.spec import OutputModalitySpec

__all__ = [
    "OutputModalitySpec",
    "MultiModalExperimentConfig",
    "load_multimodal_config",
    "ModalityCodec",
    "IdentityCodec",
    "ResizeCodec",
    "VaeCodec",
    "build_codec",
]
