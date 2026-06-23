# Vendored from https://github.com/Wan-Video/Wan2.1 (Apache-2.0).
#
# Trimmed to the DiT backbone only. The upstream aggregator also re-exported
# T5/VAE/tokenizer/vace symbols, which trigger heavy third-party imports at
# package-load time. Those modules still live alongside this file and can be
# imported directly when needed.
from .attention import attention, flash_attention
from .model import WanModel

__all__ = [
    "WanModel",
    "attention",
    "flash_attention",
]
