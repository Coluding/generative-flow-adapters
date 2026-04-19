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
        )
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")

    if config.freeze:
        model.freeze()
    return model
