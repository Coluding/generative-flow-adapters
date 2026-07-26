"""Standalone evaluation probes that run against a trained adapter checkpoint.

Distinct from the in-training eval cycle in :mod:`..training.trainer`: these are
post-hoc diagnostics answering one question each, run once against a checkpoint
rather than on a cadence during training.
"""

from generative_flow_adapters.evaluation.action_sensitivity import (
    ACTION_KEYS,
    VARIANTS,
    ActionSensitivityResult,
    format_report,
    perturb_cond,
    result_to_dict,
    run_action_sensitivity,
)

__all__ = [
    "ACTION_KEYS",
    "VARIANTS",
    "ActionSensitivityResult",
    "format_report",
    "perturb_cond",
    "result_to_dict",
    "run_action_sensitivity",
]
