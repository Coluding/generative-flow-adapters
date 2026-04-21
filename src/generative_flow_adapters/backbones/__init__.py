"""First-party backbone implementations and vendored model code.

Supported backbones:
- dynamicrafter: DynamicCrafter 3D U-Net for video generation (diffusion)
- opensora: Open-Sora MMDiT transformer for video generation (flow matching)
"""

# Lazy imports to avoid heavy dependencies at package import time
__all__ = ["dynamicrafter", "opensora"]


def __getattr__(name: str):
    if name == "dynamicrafter":
        from generative_flow_adapters.backbones import dynamicrafter
        return dynamicrafter
    if name == "opensora":
        from generative_flow_adapters.backbones import opensora
        return opensora
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
