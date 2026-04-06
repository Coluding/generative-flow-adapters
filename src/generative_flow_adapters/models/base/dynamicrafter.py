from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel, infer_prediction_type


class DynamicCrafterUNetWrapper(BaseGenerativeModel):
    """Thin wrapper around the vendored DynamiCrafter 3D UNet."""

    def __init__(self, module: torch.nn.Module, model_type: str, prediction_type: str | None = None) -> None:
        super().__init__(model_type=model_type, prediction_type=infer_prediction_type(model_type, prediction_type))
        self.module = module

    @classmethod
    def from_config(
        cls,
        model_type: str,
        unet_config_path: str,
        checkpoint_path: str | None = None,
        prediction_type: str | None = None,
        strict_checkpoint: bool = False,
    ) -> "DynamicCrafterUNetWrapper":
        try:
            from external_deps.lvdm.modules.networks.openaimodel3d import UNetModel
        except ImportError as exc:
            raise RuntimeError(
                "DynamiCrafter UNet dependencies are unavailable. "
                "Install optional deps such as `einops`, `numpy`, and `pytorch-lightning`."
            ) from exc

        unet_params = _load_unet_params(unet_config_path)
        module = UNetModel(**unet_params)
        wrapper = cls(module=module, model_type=model_type, prediction_type=prediction_type)
        if checkpoint_path:
            wrapper.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict_checkpoint)
        return wrapper

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict: dict[str, Tensor] = state.get("state_dict", state)
        candidate_prefixes = (
            "model.diffusion_model.",
            "diffusion_model.",
            "model.model.diffusion_model.",
            "",
        )

        selected = state_dict
        for prefix in candidate_prefixes:
            filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if prefix and k.startswith(prefix)}
            if filtered:
                selected = filtered
                break

        self.module.load_state_dict(selected, strict=strict)

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        context = None
        act = None
        fs = None
        dropout_actions = True

        if isinstance(cond, dict):
            context = cond.get("context")
            act = cond.get("act")
            fs = cond.get("fs")
            if "dropout_actions" in cond:
                dropout_actions = bool(cond["dropout_actions"])

        return self.module(
            x_t,
            timesteps=t,
            context=context,
            act=act,
            dropout_actions=dropout_actions,
            fs=fs,
        )


def _load_unet_params(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    model_cfg = raw.get("model", {})
    model_params = model_cfg.get("params", {})
    unet_cfg = model_params.get("unet_config", {})
    unet_params = unet_cfg.get("params")
    if not isinstance(unet_params, dict):
        raise ValueError(f"Could not find `model.params.unet_config.params` in {path}")
    return unet_params
