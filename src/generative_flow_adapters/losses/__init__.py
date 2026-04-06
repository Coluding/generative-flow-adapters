from .consistency import local_consistency_loss, multistep_self_consistency_loss
from .registry import LossRegistry

__all__ = ["LossRegistry", "local_consistency_loss", "multistep_self_consistency_loss"]
