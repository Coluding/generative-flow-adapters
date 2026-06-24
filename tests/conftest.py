"""Shared pytest fixtures / environment setup.

Several configs now enable wandb video logging (`training.extra.wandb.enable:
true`), and tests that call `build_experiment` on them would otherwise spin up a
real wandb run (network + credentials + junk runs in the user's project). Force
wandb into `disabled` mode for the whole test session so `wandb.init` is a
no-op; real runs set their own `WANDB_MODE` outside pytest.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_wandb() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
