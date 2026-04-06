from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from generative_flow_adapters.adapters.output.interface import OutputAdapterInterface, OutputAdapterResult


class DynamicCrafterOutputAdapter(OutputAdapterInterface):
    """Output adapter backed by the AVID/DynamiCrafter 3D UNet architecture."""

    def __init__(
        self,
        unet_config_path: str,
        checkpoint_path: str | None = None,
        condition_on_base_outputs: bool = True,
        output_mask: bool = False,
        strict_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        try:
            from external_deps.lvdm.modules.networks.openaimodel3d import UNetModel
        except ImportError as exc:
            raise RuntimeError(
                "DynamiCrafter adapter dependencies are unavailable. "
                "Install optional deps such as `einops`, `numpy`, and `pytorch-lightning`."
            ) from exc

        params = _load_unet_params(unet_config_path)
        self.condition_on_base_outputs = condition_on_base_outputs
        self.output_mask = output_mask
        if self.condition_on_base_outputs:
            params["in_channels"] = int(params["in_channels"]) + int(params["out_channels"])
        if self.output_mask:
            params["output_mask"] = True
        self.module = UNetModel(**params)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

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

        if self.condition_on_base_outputs and base_output is not None:
            adapter_input = torch.cat([x_t, base_output], dim=1)
        else:
            adapter_input = x_t

        output = self.module(
            adapter_input,
            timesteps=t,
            context=context,
            act=act,
            dropout_actions=dropout_actions,
            fs=fs,
        )
        if self.output_mask:
            return OutputAdapterResult(adapter_output=output[:, :-1], output_kind="prediction", gate=output[:, -1:])
        return OutputAdapterResult(adapter_output=output, output_kind="prediction")


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
