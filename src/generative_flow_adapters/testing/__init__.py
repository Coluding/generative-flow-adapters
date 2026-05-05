from .fake_data import FakeAdapterDataset, build_fake_dataloader, infer_fake_batch_spec

# Re-export from production location for backward compatibility
from generative_flow_adapters.training.shortcut_targets import attach_shortcut_targets_from_base

__all__ = ["FakeAdapterDataset", "build_fake_dataloader", "infer_fake_batch_spec", "attach_shortcut_targets_from_base"]
