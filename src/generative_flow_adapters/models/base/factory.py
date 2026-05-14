from __future__ import annotations

from generative_flow_adapters.config import ModelConfig
from generative_flow_adapters.models.base.dynamicrafter import DynamicCrafterUNetWrapper
from generative_flow_adapters.models.base.diffusers import DiffusersUNetWrapper
from generative_flow_adapters.models.base.dummy import DummyVectorField


def build_base_model(config: ModelConfig):
    provider = config.provider.lower()
    if provider == "dummy":
        model = DummyVectorField(
            model_type=config.type,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            prediction_type=config.prediction_type,
        )
    elif provider == "diffusers":
        if not config.pretrained_model_name_or_path:
            raise ValueError("diffusers provider requires pretrained_model_name_or_path")
        model = DiffusersUNetWrapper.from_pretrained(
            model_type=config.type,
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            subfolder=config.subfolder,
            prediction_type=config.prediction_type,
        )
    elif provider == "dynamicrafter":
        unet_config_path = config.extra.get("unet_config_path")
        if not isinstance(unet_config_path, str) or not unet_config_path:
            raise ValueError("dynamicrafter provider requires model.extra.unet_config_path")
        model = DynamicCrafterUNetWrapper.from_config(
            model_type=config.type,
            unet_config_path=unet_config_path,
            checkpoint_path=config.pretrained_model_name_or_path,
            prediction_type=config.prediction_type,
            strict_checkpoint=bool(config.extra.get("strict_checkpoint", False)),
            allow_missing_checkpoint=bool(config.extra.get("allow_missing_checkpoint", False)),
            allow_dummy_concat_condition=bool(config.extra.get("allow_dummy_concat_condition", False)),
            load_first_stage_model=bool(config.extra.get("load_first_stage_model", False)),
        )
    elif provider == "opensora":
        model = _build_opensora_model(config)
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")

    if config.freeze:
        model.freeze()
    return model


def _build_opensora_model(config: ModelConfig):
    """Build an Open-Sora MMDiT model from config.

    Required config.extra keys:
        - None required (uses defaults)

    Optional config.extra keys:
        - in_channels: int (default 64)
        - hidden_size: int (default 3072)
        - num_heads: int (default 24)
        - depth: int (default 19)
        - depth_single_blocks: int (default 38)
        - patch_size: int (default 2)
        - fused_qkv: bool (default False for cleaner LoRA)
        - t5_model_path: str (path to T5 encoder)
        - clip_model_path: str (path to CLIP encoder)
        - device: str (default "cuda")
        - dtype: str (default "bfloat16")
        - use_stub: bool (default False) - use stub model for testing
    """
    from generative_flow_adapters.backbones.opensora.model import OpenSoraConfig, OpenSoraModelWrapper
    import torch

    extra = config.extra

    # Parse dtype
    dtype_str = extra.get("dtype", "float32")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Build config
    opensora_config = OpenSoraConfig(
        in_channels=extra.get("in_channels", 64),
        hidden_size=extra.get("hidden_size", 3072),
        num_heads=extra.get("num_heads", 24),
        depth=extra.get("depth", 19),
        depth_single_blocks=extra.get("depth_single_blocks", 38),
        mlp_ratio=extra.get("mlp_ratio", 4.0),
        patch_size=extra.get("patch_size", 2),
        axes_dim=tuple(extra.get("axes_dim", [16, 56, 56])),
        theta=extra.get("theta", 10000),
        vec_in_dim=extra.get("vec_in_dim", 768),
        context_in_dim=extra.get("context_in_dim", 4096),
        guidance_embed=extra.get("guidance_embed", True),
        cond_embed=extra.get("cond_embed", True),
        fused_qkv=extra.get("fused_qkv", False),
        qkv_bias=extra.get("qkv_bias", True),
        grad_ckpt_settings=extra.get("grad_ckpt_settings"),
        use_liger_rope=extra.get("use_liger_rope", False),
    )

    # Use stub model for testing without real MMDiT
    if extra.get("use_stub", False):
        return _build_opensora_stub(opensora_config)

    return OpenSoraModelWrapper.from_pretrained(
        checkpoint_path=config.pretrained_model_name_or_path or "",
        config=opensora_config,
        t5_model_path=extra.get("t5_model_path"),
        clip_model_path=extra.get("clip_model_path"),
        device=extra.get("device", "cuda"),
        dtype=dtype,
        strict=bool(extra.get("strict_checkpoint", False)),
    )


def _build_opensora_stub(config):
    """Build a stub OpenSora model for testing without the real MMDiT."""
    from generative_flow_adapters.backbones.opensora.model import OpenSoraModelWrapper
    from torch import nn, Tensor

    class StubMMDiT(nn.Module):
        """Minimal stub that mimics MMDiT interface for testing."""

        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

        def __call__(
            self,
            img: Tensor,
            img_ids: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            timesteps: Tensor,
            y_vec: Tensor,
            cond: Tensor = None,
            guidance: Tensor = None,
        ) -> Tensor:
            # Return packed latents unchanged; keeps tensor shapes consistent for tests.
            del img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
            return img

    stub = StubMMDiT(config)
    return OpenSoraModelWrapper(model=stub, config=config)
