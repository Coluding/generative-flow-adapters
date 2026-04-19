from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig
from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel, infer_prediction_type


class DynamicCrafterUNetWrapper(BaseGenerativeModel):
    """Thin wrapper around the vendored DynamiCrafter 3D UNet."""

    def __init__(
        self,
        module: torch.nn.Module,
        model_type: str,
        prediction_type: str | None = None,
        allow_dummy_concat_condition: bool = False,
        diffusion_schedule_config: DiffusionScheduleConfig | None = None,
    ) -> None:
        super().__init__(model_type=model_type, prediction_type=infer_prediction_type(model_type, prediction_type))
        self.module = module
        self.allow_dummy_concat_condition = allow_dummy_concat_condition
        self._diffusion_schedule_config = diffusion_schedule_config

    @classmethod
    def from_config(
        cls,
        model_type: str,
        unet_config_path: str,
        checkpoint_path: str | None = None,
        prediction_type: str | None = None,
        strict_checkpoint: bool = False,
        allow_missing_checkpoint: bool = False,
        allow_dummy_concat_condition: bool = False,
    ) -> "DynamicCrafterUNetWrapper":
        from generative_flow_adapters.backbones.dynamicrafter.modules.networks.openaimodel3d import UNetModel

        unet_params = _load_unet_params(unet_config_path)
        diffusion_schedule_config = _load_diffusion_schedule_config(unet_config_path)
        module = UNetModel(**unet_params)
        wrapper = cls(
            module=module,
            model_type=model_type,
            prediction_type=prediction_type,
            allow_dummy_concat_condition=allow_dummy_concat_condition,
            diffusion_schedule_config=diffusion_schedule_config,
        )
        if checkpoint_path:
            if allow_missing_checkpoint and not Path(checkpoint_path).exists():
                return wrapper
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
        model_input = x_t
        context = None
        act = None
        fs = None
        dropout_actions = True

        if isinstance(cond, dict):
            context = cond.get("context")
            act = cond.get("act")
            fs = cond.get("fs")
            concat = cond.get("concat")
            if "dropout_actions" in cond:
                dropout_actions = bool(cond["dropout_actions"])
            if isinstance(concat, Tensor):
                model_input = torch.cat([model_input, concat.to(device=x_t.device, dtype=x_t.dtype)], dim=1)

        if self.allow_dummy_concat_condition:
            expected_channels = getattr(self.module, "in_channels", model_input.shape[1])
            missing_channels = expected_channels - model_input.shape[1]
            if missing_channels > 0:
                dummy_concat = torch.zeros(
                    model_input.shape[0],
                    missing_channels,
                    *model_input.shape[2:],
                    device=model_input.device,
                    dtype=model_input.dtype,
                )
                model_input = torch.cat([model_input, dummy_concat], dim=1)

        return self.module(
            model_input,
            timesteps=t,
            context=context,
            act=act,
            dropout_actions=dropout_actions,
            fs=fs,
        )

    @property
    def diffusion_schedule_config(self) -> DiffusionScheduleConfig | None:
        return self._diffusion_schedule_config


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


def _load_diffusion_schedule_config(config_path: str) -> DiffusionScheduleConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    model_cfg = raw.get("model", {})
    model_params = model_cfg.get("params", {})
    return DiffusionScheduleConfig(
        timesteps=int(model_params.get("timesteps", 1000)),
        beta_schedule=str(model_params.get("beta_schedule", "linear")),
        linear_start=float(model_params.get("linear_start", 8.5e-4)),
        linear_end=float(model_params.get("linear_end", 1.2e-2)),
        rescale_betas_zero_snr=bool(model_params.get("rescale_betas_zero_snr", False)),
    )
