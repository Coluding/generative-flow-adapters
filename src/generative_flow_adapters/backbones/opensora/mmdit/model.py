"""MMDiT (Multimodal Diffusion Transformer) model.

Vendored from Open-Sora (https://github.com/hpcaitech/Open-Sora)
Original source: opensora/models/mmdit/model.py
License: Apache License 2.0

Modified from Flux (Black Forest Labs):
    https://github.com/black-forest-labs/flux
    License: Apache License 2.0

Modifications:
    - Removed mmengine registry dependency
    - Removed colossalai checkpoint dependency
    - Simplified checkpoint loading to use safetensors/torch.load
    - Adapted imports for vendored package structure
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from generative_flow_adapters.backbones.opensora.mmdit.checkpoint import auto_grad_checkpoint
from generative_flow_adapters.backbones.opensora.mmdit.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class MMDiTConfig:
    """Configuration for MMDiT model."""

    model_type: str = "MMDiT"
    from_pretrained: str = ""
    cache_dir: str = ""
    in_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list[int] | tuple[int, ...] = (16, 56, 56)
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed: bool = True
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2

    def get(self, attribute_name: str, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name: str) -> bool:
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    """Multimodal Diffusion Transformer model.

    This is the core transformer architecture used in Open-Sora for video generation.
    It processes image/video latents jointly with text embeddings using:
    - DoubleStreamBlocks: Parallel processing of image and text with cross-attention
    - SingleStreamBlocks: Combined processing after streams are merged

    Args:
        config: Model configuration
    """

    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )

        pe_dim = config.hidden_size // config.num_heads
        axes_dim = list(config.axes_dim) if isinstance(config.axes_dim, tuple) else config.axes_dim
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(
            dim=pe_dim, theta=config.theta, axes_dim=axes_dim
        )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if config.guidance_embed
            else nn.Identity()
        )
        self.cond_in = (
            nn.Linear(
                self.in_channels + self.patch_size**2, self.hidden_size, bias=True
            )
            if config.cond_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.initialize_weights()

        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False

    def initialize_weights(self):
        if self.config.cond_embed:
            nn.init.zeros_(self.cond_in.weight)
            nn.init.zeros_(self.cond_in.bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,  # t5 encoded vec
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,  # clip encoded vec
        cond: Tensor | None = None,
        guidance: Tensor | None = None,
    ):
        """Prepare inputs for transformer blocks.

        Args:
            img: Projected noisy image latent (B, seq_len, in_channels)
            img_ids: Image position IDs (B, seq_len, 3)
            txt: T5 text embeddings (B, txt_len, context_in_dim)
            txt_ids: Text position IDs (B, txt_len, 3)
            timesteps: Diffusion timesteps (B,)
            y_vec: CLIP vector embeddings (B, vec_in_dim)
            cond: Optional I2V conditioning (B, seq_len, in_channels + patch_size^2)
            guidance: Optional guidance scale (B,)

        Returns:
            Tuple of (img, txt, vec, pe) ready for transformer blocks
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        if self.config.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            img = img + self.cond_in(cond)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)

        txt = self.txt_in(txt)

        # concat: 4096 + t*h*2/4
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            # we only apply lora to double/single blocks, thus we only need to enable grad for these inputs
            img.requires_grad_()
            txt.requires_grad_()

        return img, txt, vec, pe

    def enable_input_require_grads(self):
        """Enable input gradients for LoRA training."""
        self._input_requires_grad = True

    def forward_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor | None = None,
        guidance: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward with gradient checkpointing on all blocks."""
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        for block in self.double_blocks:
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = auto_grad_checkpoint(block, img, vec, pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_selective_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor | None = None,
        guidance: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward with selective gradient checkpointing."""
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        for block in self.double_blocks[:ckpt_depth_double]:
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

        for block in self.double_blocks[ckpt_depth_double:]:
            img, txt = block(img, txt, vec, pe)

        ckpt_depth_single = self.config.grad_ckpt_settings[1]
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks[:ckpt_depth_single]:
            img = auto_grad_checkpoint(block, img, vec, pe)
        for block in self.single_blocks[ckpt_depth_single:]:
            img = block(img, vec, pe)

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> nn.Module:
    """Load checkpoint into model.

    Supports:
    - .safetensors files
    - .pt/.pth PyTorch checkpoints

    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        device: Device to load weights to
        strict: Whether to strictly enforce key matching

    Returns:
        Model with loaded weights
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            state_dict = load_file(str(path), device=str(device))
        except ImportError:
            raise ImportError("safetensors required for .safetensors files")
    else:
        state_dict = torch.load(str(path), map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Handle potential key prefixes from different training frameworks
    prefixes = ["model.", "module.", ""]
    for prefix in prefixes:
        if prefix and any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
            break

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    return model


def build_mmdit(
    config: MMDiTConfig,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    strict: bool = False,
) -> MMDiTModel:
    """Build an MMDiT model from config.

    Args:
        config: Model configuration
        device: Target device
        dtype: Model dtype
        checkpoint_path: Optional path to pretrained weights
        strict: Whether to strictly enforce key matching when loading

    Returns:
        Initialized MMDiTModel
    """
    with torch.device(device):
        model = MMDiTModel(config).to(dtype)

    if checkpoint_path:
        model = load_checkpoint(model, checkpoint_path, device=device, strict=strict)

    return model
