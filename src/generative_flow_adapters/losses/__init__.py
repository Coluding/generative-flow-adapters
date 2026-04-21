from .consistency import local_consistency_loss, multistep_self_consistency_loss, shortcut_direction_loss
from .diffusion import DiffusionTrainingObjective, diffusion_loss
from .flow_matching import FlowMatchingTrainingObjective, flow_matching_loss
from .registry import LossRegistry

__all__ = [
    "LossRegistry",
    "local_consistency_loss",
    "shortcut_direction_loss",
    "multistep_self_consistency_loss",
    "diffusion_loss",
    "flow_matching_loss",
    "DiffusionTrainingObjective",
    "FlowMatchingTrainingObjective",
]
