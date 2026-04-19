from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from generative_flow_adapters.conditioning.utils.dynamicrafter_conditioning import prepare_dynamicrafter_condition
from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult
from generative_flow_adapters.backbones.dynamicrafter.modules.networks.openaimodel3d import UNetModel
from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig

class DynamicCrafterOutputAdapter(OutputAdapterInterface):
    """Output adapter backed by the AVID/DynamiCrafter 3D UNet architecture."""

    def __init__(
        self,
        unet_config_path: str,
        checkpoint_path: str | None = None,
        condition_on_base_outputs: bool = True,
        output_mask: bool = False,
        strict_checkpoint: bool = False,
        cond_dim: int | None = None,
        cond_hidden_dim: int | None = None,
        use_adapter_conditioning: bool = True,
        allow_dummy_concat_condition: bool = False,
        use_step_level_conditioning: bool = False,
        step_level_key: str = "step_level",
        step_level_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()


        params = _load_unet_params(unet_config_path)
        self.condition_on_base_outputs = condition_on_base_outputs
        self.output_mask = output_mask
        self.cond_dim = cond_dim
        self.cond_hidden_dim = cond_hidden_dim
        self.use_adapter_conditioning = use_adapter_conditioning
        self.allow_dummy_concat_condition = allow_dummy_concat_condition
        self.use_step_level_conditioning = use_step_level_conditioning
        self.step_level_key = step_level_key
        self.step_level_hidden_dim = int(step_level_hidden_dim or (cond_hidden_dim or cond_dim or 128))
        if self.use_adapter_conditioning and self.cond_dim is not None and self.cond_dim > 0:
            params["adapter_condition_dim"] = self.cond_dim
            params["adapter_condition_hidden_dim"] = self.cond_hidden_dim
        if self.condition_on_base_outputs:
            params["in_channels"] = int(params["in_channels"]) + int(params["out_channels"])
        if self.output_mask:
            params["output_mask"] = True
        self.module = UNetModel(**params)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict_checkpoint)
        self._diffusion_schedule_config = _load_diffusion_schedule_config(unet_config_path)
        self.step_level_embed: nn.Module | None = None
        if self.use_step_level_conditioning:
            if self.cond_dim is None or self.cond_dim <= 0:
                raise ValueError("Step-level conditioning requires a positive cond_dim.")
            self.step_level_embed = nn.Sequential(
                nn.Linear(1, self.step_level_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.step_level_hidden_dim, int(self.cond_dim)),
            )

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict: dict[str, Tensor] = state.get("state_dict", state)
        candidate_prefixes = (
            "action_cond_model.",
            "model.diffusion_model.",
            "diffusion_model.",
            "",
        )

        selected = state_dict
        for prefix in candidate_prefixes:
            filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if prefix and k.startswith(prefix)}
            if filtered:
                selected = filtered
                break

        self.module.load_state_dict(selected, strict=strict)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: object | None,
        base_output: Tensor | None = None,
    ) -> OutputAdapterResult:
        cond = prepare_dynamicrafter_condition(
            cond,
            x_t=x_t,
            use_step_level_conditioning=self.use_step_level_conditioning,
            step_level_key=self.step_level_key,
            step_level_embed=self.step_level_embed,
        )
        context = None
        act = None
        fs = None
        dropout_actions = True

        if isinstance(cond, dict):
            context = cond.get("context")
            act = cond.get("act")
            fs = cond.get("fs")
            adapter_embedding = cond.get("embedding")
            concat = cond.get("concat")
            if "dropout_actions" in cond:
                dropout_actions = bool(cond["dropout_actions"])
        else:
            adapter_embedding = None
            concat = None

        if self.condition_on_base_outputs and base_output is not None:
            adapter_input = torch.cat([x_t, base_output], dim=1)
        else:
            adapter_input = x_t
        if isinstance(concat, Tensor):
            adapter_input = torch.cat([adapter_input, concat.to(device=x_t.device, dtype=x_t.dtype)], dim=1)
        if self.allow_dummy_concat_condition:
            expected_channels = getattr(self.module, "in_channels", adapter_input.shape[1])
            missing_channels = expected_channels - adapter_input.shape[1]
            if missing_channels > 0:
                dummy_concat = torch.zeros(
                    adapter_input.shape[0],
                    missing_channels,
                    *adapter_input.shape[2:],
                    device=adapter_input.device,
                    dtype=adapter_input.dtype,
                )
                adapter_input = torch.cat([adapter_input, dummy_concat], dim=1)

        output = self.module(
            adapter_input,
            timesteps=t,
            context=context,
            act=act,
            dropout_actions=dropout_actions,
            fs=fs,
            adapter_embedding=adapter_embedding,
        )
        if self.output_mask:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Expected DynamicCrafter output adapter with output_mask=True to return (prediction, mask).")
            prediction, gate = output
            return OutputAdapterResult(adapter_output=prediction, output_kind="prediction", gate=gate)
        return OutputAdapterResult(adapter_output=output, output_kind="prediction")

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
    return dict(unet_params)


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
