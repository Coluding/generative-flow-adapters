"""Open-Sora MMDiT model wrapper for the adapter framework.

This module provides a wrapper around Open-Sora's MMDiTModel that implements
the BaseGenerativeModel interface, enabling use with the adapter framework.

Key features:
- Automatic packing/unpacking of video latents
- Text encoding via T5 and CLIP
- Flow matching training support
- Compatible with existing adapter implementations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from generative_flow_adapters.backbones.opensora.common import (
    create_image_ids,
    create_text_ids,
    pack_latents,
    unpack_latents,
)
from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel


@dataclass
class OpenSoraConfig:
    """Configuration for Open-Sora model wrapper."""

    # Model architecture
    in_channels: int = 64
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 19  # Number of DoubleStreamBlocks
    depth_single_blocks: int = 38  # Number of SingleStreamBlocks
    mlp_ratio: float = 4.0
    patch_size: int = 2
    axes_dim: tuple[int, int, int] = (16, 56, 56)  # RoPE dimensions for (t, h, w)
    theta: int = 10000  # RoPE theta

    # Text encoder dimensions
    vec_in_dim: int = 768  # CLIP embedding dimension
    context_in_dim: int = 4096  # T5 embedding dimension

    # Model behavior
    guidance_embed: bool = True
    cond_embed: bool = True  # For I2V conditioning
    fused_qkv: bool = False  # Disable for cleaner LoRA injection
    qkv_bias: bool = True

    # Checkpointing
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False


class OpenSoraModelWrapper(BaseGenerativeModel):
    """Wrapper around Open-Sora's MMDiT for the adapter framework.

    This wrapper:
    1. Implements the BaseGenerativeModel interface (x_t, t, cond) -> output
    2. Handles packing/unpacking of video latents automatically
    3. Manages text encoding via T5 and CLIP
    4. Supports both T2V and I2V conditioning

    The wrapper expects video latents in (B, C, T, H, W) format and handles
    all the internal transformations required by MMDiT.
    """

    def __init__(
        self,
        model: nn.Module,
        config: OpenSoraConfig,
        t5_encoder: nn.Module | None = None,
        clip_encoder: nn.Module | None = None,
    ) -> None:
        """Initialize the Open-Sora model wrapper.

        Args:
            model: The MMDiTModel instance
            config: Model configuration
            t5_encoder: Optional T5 text encoder (for encoding prompts)
            clip_encoder: Optional CLIP encoder (for vector embeddings)
        """
        super().__init__(model_type="flow", prediction_type="velocity")
        self.model = model
        self.config = config
        self.t5_encoder = t5_encoder
        self.clip_encoder = clip_encoder

        # Cache for encoded text (to avoid re-encoding during adapter forward passes)
        self._cached_txt: Tensor | None = None
        self._cached_txt_ids: Tensor | None = None
        self._cached_y_vec: Tensor | None = None
        self._cache_key: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: OpenSoraConfig | None = None,
        t5_model_path: str | None = None,
        clip_model_path: str | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        strict: bool = False,
    ) -> "OpenSoraModelWrapper":
        """Load a pretrained Open-Sora model.

        Args:
            checkpoint_path: Path to model checkpoint (.safetensors or .pt)
            config: Model configuration (uses defaults if None)
            t5_model_path: Path to T5 encoder weights
            clip_model_path: Path to CLIP encoder weights
            device: Target device
            dtype: Model dtype
            strict: Whether to strictly load checkpoint

        Returns:
            Initialized OpenSoraModelWrapper
        """
        config = config or OpenSoraConfig()

        # Build the MMDiT model
        model = cls._build_mmdit(config, device=device, dtype=dtype)

        # Load checkpoint
        if checkpoint_path:
            model = cls._load_checkpoint(model, checkpoint_path, device=device, strict=strict)

        # Build text encoders if paths provided
        t5_encoder = None
        clip_encoder = None
        if t5_model_path:
            t5_encoder = cls._build_t5_encoder(t5_model_path, device=device, dtype=dtype)
        if clip_model_path:
            clip_encoder = cls._build_clip_encoder(clip_model_path, device=device, dtype=dtype)

        return cls(model=model, config=config, t5_encoder=t5_encoder, clip_encoder=clip_encoder)

    @staticmethod
    def _build_mmdit(
        config: OpenSoraConfig,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Build the MMDiT model from config.

        Uses the vendored MMDiT implementation from this package.
        """
        from generative_flow_adapters.backbones.opensora.mmdit import (
            MMDiTConfig,
            MMDiTModel,
        )

        mmdit_config = MMDiTConfig(
            in_channels=config.in_channels,
            vec_in_dim=config.vec_in_dim,
            context_in_dim=config.context_in_dim,
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            num_heads=config.num_heads,
            depth=config.depth,
            depth_single_blocks=config.depth_single_blocks,
            axes_dim=list(config.axes_dim),
            theta=config.theta,
            qkv_bias=config.qkv_bias,
            guidance_embed=config.guidance_embed,
            cond_embed=config.cond_embed,
            fused_qkv=config.fused_qkv,
            grad_ckpt_settings=config.grad_ckpt_settings,
            use_liger_rope=config.use_liger_rope,
            patch_size=config.patch_size,
        )

        with torch.device(device):
            model = MMDiTModel(mmdit_config).to(dtype)

        return model

    @staticmethod
    def _load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        strict: bool = False,
    ) -> nn.Module:
        """Load checkpoint into model."""
        path = Path(checkpoint_path)

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

        # Handle potential key prefixes
        prefixes = ["model.", "module.", ""]
        for prefix in prefixes:
            if prefix:
                filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                if filtered:
                    state_dict = filtered
                    break

        model.load_state_dict(state_dict, strict=strict)
        return model

    @staticmethod
    def _build_t5_encoder(
        model_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Build T5 text encoder."""
        try:
            from opensora.models.text.conditioner import HFEmbedder
            return HFEmbedder(
                model_path,
                max_length=512,
                torch_dtype=dtype,
                device=device,
            )
        except ImportError:
            # Fallback to transformers directly
            from transformers import T5EncoderModel, T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5EncoderModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
            model.eval()
            return _T5EncoderWrapper(model, tokenizer, device=device)

    @staticmethod
    def _build_clip_encoder(
        model_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Build CLIP text encoder."""
        try:
            from opensora.models.text.conditioner import HFEmbedder
            return HFEmbedder(
                model_path,
                max_length=77,
                torch_dtype=dtype,
                device=device,
            )
        except ImportError:
            from transformers import CLIPTextModel, CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(model_path)
            model = CLIPTextModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
            model.eval()
            return _CLIPEncoderWrapper(model, tokenizer, device=device)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None = None,
    ) -> Tensor:
        """Forward pass through the Open-Sora model.

        Args:
            x_t: Noisy video latents of shape (B, C, T, H, W)
            t: Timesteps of shape (B,) in range [0, 1] for flow matching
            cond: Conditioning dict with keys:
                - txt: Pre-encoded T5 embeddings (B, seq_len, 4096), or
                - prompt: List of text prompts to encode
                - txt_ids: Position IDs for text tokens
                - y_vec: CLIP embeddings (B, 768)
                - cond: I2V conditioning latents (B, C+patch_size^2, T, H, W)
                - guidance: CFG guidance scale

        Returns:
            Predicted velocity of shape (B, C, T, H, W)
        """
        if x_t.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got {x_t.dim()}D")

        batch_size = x_t.shape[0]
        num_frames = x_t.shape[2]
        height = x_t.shape[3] // self.config.patch_size
        width = x_t.shape[4] // self.config.patch_size
        device = x_t.device
        dtype = x_t.dtype

        # Pack latents to sequence format
        img = pack_latents(x_t, patch_size=self.config.patch_size)

        # Create image position IDs
        img_ids = create_image_ids(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

        # Extract conditioning
        txt, txt_ids, y_vec, i2v_cond, guidance = self._prepare_conditioning(
            cond=cond,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Forward through MMDiT
        output = self.model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t,
            y_vec=y_vec,
            cond=i2v_cond,
            guidance=guidance,
        )

        # Unpack back to video format
        return unpack_latents(
            output,
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.config.patch_size,
        )

    def _prepare_conditioning(
        self,
        cond: object | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Prepare conditioning inputs for the model.

        Returns:
            txt: T5 text embeddings (B, seq_len, context_in_dim)
            txt_ids: Text position IDs (B, seq_len, 3)
            y_vec: CLIP vector embeddings (B, vec_in_dim)
            i2v_cond: I2V conditioning (packed) or None
            guidance: Guidance scale tensor or None
        """
        if cond is None:
            cond = {}

        if not isinstance(cond, Mapping):
            raise TypeError(f"Expected dict conditioning, got {type(cond)}")

        # Text embeddings
        txt = cond.get("txt")
        txt_ids = cond.get("txt_ids")
        prompts = cond.get("prompt")

        if txt is None and prompts is not None:
            # Encode prompts
            txt, txt_ids = self._encode_text(prompts, device=device, dtype=dtype)
        elif txt is None:
            # Use empty/null text
            seq_len = 512
            txt = torch.zeros(batch_size, seq_len, self.config.context_in_dim, device=device, dtype=dtype)
            txt_ids = create_text_ids(batch_size, seq_len, device=device, dtype=dtype)

        if txt_ids is None:
            txt_ids = create_text_ids(batch_size, txt.shape[1], device=device, dtype=dtype)

        # CLIP embeddings
        y_vec = cond.get("y_vec")
        if y_vec is None:
            if prompts is not None and self.clip_encoder is not None:
                y_vec = self._encode_clip(prompts, device=device, dtype=dtype)
            else:
                y_vec = torch.zeros(batch_size, self.config.vec_in_dim, device=device, dtype=dtype)

        # I2V conditioning
        i2v_cond = cond.get("cond")
        if i2v_cond is not None:
            i2v_cond = pack_latents(i2v_cond, patch_size=self.config.patch_size)

        # Guidance scale
        guidance = cond.get("guidance")
        if guidance is not None and not isinstance(guidance, Tensor):
            guidance = torch.full((batch_size,), guidance, device=device, dtype=dtype)

        return txt, txt_ids, y_vec, i2v_cond, guidance

    def _encode_text(
        self,
        prompts: list[str] | str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Encode text prompts using T5."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.t5_encoder is None:
            raise RuntimeError("T5 encoder not loaded. Provide txt directly or load T5 model.")

        with torch.no_grad():
            txt = self.t5_encoder(prompts)
            if isinstance(txt, tuple):
                txt = txt[0]
            txt = txt.to(device=device, dtype=dtype)

        txt_ids = create_text_ids(len(prompts), txt.shape[1], device=device, dtype=dtype)
        return txt, txt_ids

    def _encode_clip(
        self,
        prompts: list[str] | str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Encode text prompts using CLIP for vector embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.clip_encoder is None:
            raise RuntimeError("CLIP encoder not loaded. Provide y_vec directly or load CLIP model.")

        with torch.no_grad():
            y_vec = self.clip_encoder(prompts)
            if isinstance(y_vec, tuple):
                # Get pooled output
                y_vec = y_vec[1] if len(y_vec) > 1 else y_vec[0].mean(dim=1)
            y_vec = y_vec.to(device=device, dtype=dtype)

        return y_vec

    def freeze(self) -> "OpenSoraModelWrapper":
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        if self.t5_encoder is not None:
            for param in self.t5_encoder.parameters():
                param.requires_grad_(False)
            self.t5_encoder.eval()

        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad_(False)
            self.clip_encoder.eval()

        return self

    def get_target_modules_for_lora(self) -> list[str]:
        """Get recommended module names for LoRA injection.

        These are the attention projections that work well with LoRA.
        Note: Requires fused_qkv=False in config for separate projections.
        """
        if self.config.fused_qkv:
            return [
                # DoubleStreamBlocks - fused QKV
                "double_blocks.*.img_attn.qkv",
                "double_blocks.*.img_attn.proj",
                "double_blocks.*.txt_attn.qkv",
                "double_blocks.*.txt_attn.proj",
                # SingleStreamBlocks - fused
                "single_blocks.*.linear1",
                "single_blocks.*.linear2",
            ]
        else:
            return [
                # DoubleStreamBlocks - separate projections
                "double_blocks.*.img_attn.q_proj",
                "double_blocks.*.img_attn.k_proj",
                "double_blocks.*.img_attn.v_proj",
                "double_blocks.*.img_attn.proj",
                "double_blocks.*.txt_attn.q_proj",
                "double_blocks.*.txt_attn.k_proj",
                "double_blocks.*.txt_attn.v_proj",
                "double_blocks.*.txt_attn.proj",
                # SingleStreamBlocks - separate projections
                "single_blocks.*.q_proj",
                "single_blocks.*.k_proj",
                "single_blocks.*.linear2",
            ]


class _T5EncoderWrapper(nn.Module):
    """Simple wrapper for T5 encoder with tokenizer."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: torch.device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, prompts: list[str]) -> Tensor:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class _CLIPEncoderWrapper(nn.Module):
    """Simple wrapper for CLIP text encoder with tokenizer."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: torch.device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, outputs.pooler_output