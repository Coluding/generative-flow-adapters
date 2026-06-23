# Vendored from https://github.com/Wan-Video/Wan2.1 (Apache-2.0).
#
# The upstream package __init__ eagerly imports the full inference pipelines
# (text2video/image2video/vace) which in turn pull in T5, CLIP, the Wan-VAE and
# easydict configs. For adapter research we only need the bare `WanModel` DiT,
# so this aggregator is intentionally left minimal. Import the heavy modules
# explicitly (e.g. `from ...backbones.wan.modules.vae import WanVAE`) when a
# pixel-space decode path is actually needed.
