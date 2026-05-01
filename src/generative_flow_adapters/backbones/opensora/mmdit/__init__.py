"""Vendored Open-Sora MMDiT model implementation.

This module contains a vendored copy of the MMDiT (Multimodal Diffusion Transformer)
model from the Open-Sora project.

Source:
    Repository: https://github.com/hpcaitech/Open-Sora
    Path: opensora/models/mmdit/
    License: Apache License 2.0

The original code is modified from Flux (Black Forest Labs):
    Repository: https://github.com/black-forest-labs/flux
    License: Apache License 2.0

Modifications made for this vendored version:
    - Removed dependencies on mmengine registry
    - Simplified checkpoint utilities to remove colossalai dependency
    - Made flash_attn and liger_kernel optional with fallbacks
    - Adapted imports for the new package structure
"""

from generative_flow_adapters.backbones.opensora.mmdit.model import (
    MMDiTConfig,
    MMDiTModel,
)
from generative_flow_adapters.backbones.opensora.mmdit.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    EmbedND,
    LigerEmbedND,
    MLPEmbedder,
    LastLayer,
    timestep_embedding,
)

__all__ = [
    "MMDiTConfig",
    "MMDiTModel",
    "DoubleStreamBlock",
    "SingleStreamBlock",
    "EmbedND",
    "LigerEmbedND",
    "MLPEmbedder",
    "LastLayer",
    "timestep_embedding",
]