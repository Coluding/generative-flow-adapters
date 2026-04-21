"""Open-Sora backbone integration for generative flow adapters.

This module provides integration with the Open-Sora video generation model,
which uses an MMDiT (Multimodal Diffusion Transformer) architecture based on Flux.

Key differences from DynamicCrafter U-Net:
- Transformer-based (not convolutional)
- Flow matching (velocity prediction)
- Joint self-attention for text and image (not cross-attention)
- No skip connections (sequential processing)
"""

from generative_flow_adapters.backbones.opensora.common import (
    create_image_ids,
    create_text_ids,
    pack_latents,
    unpack_latents,
)
from generative_flow_adapters.backbones.opensora.model import OpenSoraModelWrapper

__all__ = [
    "OpenSoraModelWrapper",
    "pack_latents",
    "unpack_latents",
    "create_image_ids",
    "create_text_ids",
]