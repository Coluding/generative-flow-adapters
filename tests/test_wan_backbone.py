"""Smoke tests for the vendored Wan2.1 DiT backbone wrapper.

Uses the tiny CPU config (``configs/base/wan2.1_tiny.yaml``) and random weights,
so it runs anywhere without flash-attn or the real checkpoint.
"""

from __future__ import annotations

import torch

from generative_flow_adapters.config import ModelConfig
from generative_flow_adapters.models.base.factory import build_base_model

_TINY_CONFIG = "configs/base/wan2.1_tiny.yaml"


def _build(freeze: bool = True):
    cfg = ModelConfig(
        type="flow",
        provider="wan2.1",
        prediction_type="velocity",
        freeze=freeze,
        extra={"wan_config_path": _TINY_CONFIG, "allow_missing_checkpoint": True},
    )
    return build_base_model(cfg)


def test_flow_velocity_metadata():
    base = _build()
    assert base.model_type == "flow"
    assert base.prediction_type == "velocity"


def test_freeze_disables_grads():
    base = _build(freeze=True)
    assert all(not p.requires_grad for p in base.parameters())


def test_forward_shape_roundtrip():
    base = _build()
    x = torch.randn(2, 16, 4, 8, 8)
    t = torch.tensor([120.0, 880.0])
    with torch.no_grad():
        v = base(x, t)
    assert v.shape == x.shape  # velocity lives in latent space
    assert v.dtype == torch.float32


def test_scalar_timestep_broadcasts():
    base = _build()
    x = torch.randn(3, 16, 2, 8, 8)
    with torch.no_grad():
        v = base(x, torch.tensor(500.0))
    assert v.shape == x.shape


def test_null_context_matches_explicit_zero_context():
    base = _build()
    x = torch.randn(2, 16, 4, 8, 8)
    t = torch.tensor([10.0, 990.0])
    with torch.no_grad():
        v_null = base(x, t)
        v_zero = base(x, t, cond={"context": torch.zeros(1, base.module.text_dim)})
    assert torch.allclose(v_null, v_zero, atol=1e-5)


def test_additive_composition_identity():
    """base + 0 * delta == base (the composition contract adapters rely on)."""
    base = _build()
    x = torch.randn(2, 16, 4, 8, 8)
    t = torch.tensor([300.0, 700.0])
    with torch.no_grad():
        v = base(x, t)
        composed = base(x, t) + 0.0 * torch.randn_like(v)
    assert torch.allclose(v, composed)


def test_adapted_model_composition():
    """Full AdaptedModel: frozen Wan base + trainable output-head adapter.

    Verifies the frozen base contributes no trainable params, the adapter does,
    and the zero-initialised delta makes the composition the identity at init.
    """
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    experiment = build_experiment(load_config("configs/wan_output_adapter.yaml"))
    model = experiment.model

    assert sum(p.numel() for p in model.base_model.parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in model.adapter.parameters() if p.requires_grad) > 0

    x = torch.randn(2, 16, 4, 16, 16)
    t = torch.rand(2)
    cond = {"action": torch.randn(2, 4)}
    model.eval()
    with torch.no_grad():
        base_out = model.base_model(x, t)
        full = model(x, t, cond)
    assert torch.allclose(full, base_out, atol=1e-5)


def test_flow_micro_step_and_self_consistency_target():
    """Flow Euler micro-step + self-consistency target math (no model needed)."""
    from generative_flow_adapters.training.shortcut_targets import (
        compute_self_consistency_target_v_flow,
        flow_micro_step_v,
    )

    x = torch.randn(2, 16, 2, 8, 8)
    v = torch.randn_like(x)
    d = torch.tensor([0.25, 0.25])
    # x_{t-d} = x - d * v (generative-direction Euler step).
    expected = x - 0.25 * v
    assert torch.allclose(flow_micro_step_v(x=x, v=v, d=d), expected)

    # With a constant-velocity dummy model, both calls return the same v, so the
    # averaged target equals that v.
    class _ConstV(torch.nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def forward(self, x, t, cond=None):
            return self.val

    target = compute_self_consistency_target_v_flow(
        model=_ConstV(v), x_t=x, t=torch.rand(2), cond_half={"action": torch.randn(2, 4)}, d=d
    )
    assert torch.allclose(target, v)


def test_wan_avid_adapter_tiers_and_composition():
    """The action-conditioned tiny-Wan adapter (AVID-style) builds at the
    expected param tier, composes as identity at init, and is conditioned on
    both action and step_level."""
    from generative_flow_adapters.adapters.output.wan import Wan21OutputAdapter
    from generative_flow_adapters.backbones.wan.modules.action_model import ActionWanModel
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    # Tier param counts land near 11M / 34M / 150M.
    counts = {
        "configs/base/wan_adapter_11m.yaml": (8e6, 16e6),
        "configs/base/wan_adapter_34m.yaml": (28e6, 42e6),
        "configs/base/wan_adapter_150m.yaml": (140e6, 172e6),
    }
    for path, (lo, hi) in counts.items():
        a = Wan21OutputAdapter.from_config(path, cond_dim=512)
        n = sum(p.numel() for p in a.parameters())
        assert lo < n < hi, f"{path}: {n/1e6:.1f}M outside ({lo/1e6}, {hi/1e6})"

    config = load_config("configs/diffusion_wan_avid_shortcut_metaworld.yaml")
    config.model.extra["wan_config_path"] = _TINY_CONFIG
    config.model.extra["dtype"] = "float32"
    config.model.extra["allow_missing_checkpoint"] = True
    model = build_experiment(config).model.eval()
    assert isinstance(model.adapter, Wan21OutputAdapter)
    assert sum(p.numel() for p in model.base_model.parameters() if p.requires_grad) == 0

    x = torch.randn(2, 16, 4, 16, 16)
    t = torch.tensor([120.0, 880.0])
    with torch.no_grad():
        base = model.base_model(x, t)
        full = model(x, t, {"action": torch.randn(2, 4), "step_level": torch.full((2,), 0.25)})
    assert torch.allclose(full, base, atol=1e-5)  # zero-init head -> identity

    # The adapter consumes the fused `embedding`; conditioning + step actually
    # move the delta (after nudging off zero-init).
    adapter = model.adapter
    cond_dim = adapter.module.cond_dim
    for p in adapter.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)

    def delta(emb, s):
        with torch.no_grad():
            return adapter(x, t, {"embedding": emb, "step_level": s}, base_output=torch.zeros_like(x)).adapter_output

    base_e = delta(torch.zeros(2, cond_dim), torch.full((2,), 0.25))
    assert (delta(torch.randn(2, cond_dim), torch.full((2,), 0.25)) - base_e).abs().max() > 1e-5
    assert (delta(torch.zeros(2, cond_dim), torch.full((2,), 1.0)) - base_e).abs().max() > 1e-5


def test_wan_preprocessor_feeds_native_timestep_scale():
    """The preprocessor builds x_t with sigma in [0,1] but feeds the model
    t = sigma * timestep_scale (Wan native [0,1000])."""
    from generative_flow_adapters.data import WanBatchPreprocessConfig, WanBatchPreprocessor

    class _FakeVAE:
        device = torch.device("cpu")

        def encode(self, vids):
            return [torch.randn(16, 2, 8, 8) for _ in vids]

    pp = WanBatchPreprocessor(
        vae=_FakeVAE(),
        config=WanBatchPreprocessConfig(target_height=64, target_width=64, timestep_scale=1000.0),
        condition_keys=("act",),
    )
    raw = {"video": (torch.rand(4, 8, 64, 64, 3) * 255).byte(), "act": torch.randn(4, 8, 4)}
    batch = pp(raw)
    assert batch["t"].max() <= 1000.0 and batch["t"].max() > 1.5  # scaled up, not raw sigma
    assert batch["x_t"].shape == batch["target"].shape


def test_trainer_selects_flow_inference_sampler():
    """A flow model gets the FlowInferenceSampler; a diffusion model does not."""
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.inference import FlowInferenceSampler
    from generative_flow_adapters.training.builders import build_experiment
    from generative_flow_adapters.training.trainer import Trainer

    experiment = build_experiment(load_config("configs/wan_output_adapter.yaml"))
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn,
                      load_config("configs/wan_output_adapter.yaml").training)
    assert isinstance(trainer.inference_sampler, FlowInferenceSampler)

    # A short rollout from a fake batch is finite and shape-preserving.
    batch = {"target": torch.randn(1, 16, 4, 16, 16), "cond": {"action": torch.randn(1, 4)}}
    out = trainer.inference_sampler.sample_from_batch(batch, num_inference_steps=4)
    assert out.shape == batch["target"].shape
    assert torch.isfinite(out).all()


def test_flow_shortcut_training_step_fires():
    """The trainer's flow-native shortcut branch produces a finite
    shortcut_direction_loss when step-level conditioning is enabled."""
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment
    from generative_flow_adapters.training.trainer import Trainer

    config = load_config("configs/diffusion_wan_shortcut_metaworld.yaml")
    config.model.extra["wan_config_path"] = _TINY_CONFIG  # tiny base for CPU
    config.model.extra["allow_missing_checkpoint"] = True
    config.model.extra["dtype"] = "float32"  # CPU test runs fp32 base
    config.model.pretrained_model_name_or_path = None
    config.training.extra["shortcut_anchor_prob"] = 0.0  # force a shortcut step
    config.training.extra["amp_dtype"] = "float32"  # CPU test runs fp32 (no bf16 autocast)
    experiment = build_experiment(config)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)

    z0 = torch.randn(2, 16, 2, 8, 8)
    noise = torch.randn_like(z0)
    t = torch.rand(2)
    t_b = t.view(2, 1, 1, 1, 1)
    batch = {
        "x_t": (1 - t_b) * z0 + t_b * noise,
        "t": t,
        "target": noise - z0,
        "cond": {"action": torch.randn(2, 4)},
    }
    metrics = trainer.training_step(batch)
    assert "shortcut_direction_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["shortcut_direction_loss"]))
    # The frozen base stays frozen; the adapter (incl. step_level_embed) trains.
    assert sum(p.numel() for p in experiment.model.base_model.parameters() if p.requires_grad) == 0
    assert experiment.model.adapter.step_level_embed is not None
