from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from generative_flow_adapters.losses.consistency import local_consistency_loss, multistep_self_consistency_loss
from generative_flow_adapters.losses.diffusion import diffusion_loss
from generative_flow_adapters.losses.flow_matching import flow_matching_loss


class LossRegistry:
    _base_losses: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
        "diffusion": diffusion_loss,
        "flow": flow_matching_loss,
        "flow_matching": flow_matching_loss,
    }
    _consistency_losses: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
        "local_consistency": local_consistency_loss,
        "multistep_self_consistency": multistep_self_consistency_loss,
    }

    @classmethod
    def get_loss(cls, model_type: str):
        try:
            return cls._base_losses[model_type]
        except KeyError as exc:
            raise ValueError(f"Unknown loss key: {model_type}") from exc

    @classmethod
    def get_consistency_loss(cls, name: str):
        try:
            return cls._consistency_losses[name]
        except KeyError as exc:
            raise ValueError(f"Unknown consistency loss key: {name}") from exc
