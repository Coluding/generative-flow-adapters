"""Phase 0–2 substrate tests for the multimodal compositional adapter.

Runs entirely on the lightweight ``DummyVectorField`` base (no DynamiCrafter /
GPU), so the multi-stream contract, codecs, per-stream noising, and summed loss
are validated cheaply. The video stream is modelled as just another stream with
``has_frozen_prior=True``.
"""

from __future__ import annotations

import torch

from generative_flow_adapters.multimodal.builders import build_multimodal_experiment
from generative_flow_adapters.multimodal.codecs import IdentityCodec, ResizeCodec
from generative_flow_adapters.multimodal.config import MultiModalExperimentConfig
from generative_flow_adapters.multimodal.fusion import LearnedMaskFusion
from generative_flow_adapters.multimodal.preprocessor import MultiModalBatchPreprocessor
from generative_flow_adapters.multimodal.spec import OutputModalitySpec
from generative_flow_adapters.multimodal.trainer import MultiModalTrainer


def _dummy_config_dict(fusion="trivial"):
    return {
        "name": "mm-substrate",
        "model": {"type": "diffusion", "provider": "dummy", "feature_dim": 8, "hidden_dim": 32},
        "adapter": {"type": "output", "extra": {"fusion": fusion}},
        "conditioning": {"type": "structured", "output_dim": 16,
                         "conditions": [{"key": "act", "input_dim": 4}]},
        "training": {"learning_rate": 2e-3, "diffusion_timesteps": 1000},
        "output_modalities": [
            {"name": "video", "kind": "vector", "feature_shape": [8], "has_frozen_prior": True,
             "loss_weight": 1.0, "hidden_dim": 64},
            {"name": "proprio", "kind": "vector", "feature_shape": [7], "loss_weight": 1.0, "hidden_dim": 64},
            {"name": "tactile", "kind": "map", "feature_shape": [2, 4, 4], "codec": "resize",
             "codec_kwargs": {"size": [4, 4]}, "loss_weight": 1.0, "hidden_dim": 64},
        ],
    }


def _overfit(fusion, steps=800):
    torch.manual_seed(0)
    config = MultiModalExperimentConfig.from_dict(_dummy_config_dict(fusion=fusion))
    components = build_multimodal_experiment(config)
    pre = MultiModalBatchPreprocessor(config.output_modalities, components.codecs, video_preprocessor=None)
    trainer = MultiModalTrainer(components.model, components.optimizer, config.base.training, config.output_modalities)

    raw = _dummy_batch(batch_size=2)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        history.append(trainer.training_step(pre(raw)))
    return components, history


# --------------------------------------------------------------------------- #
# Phase 2 — substrate model + trainer (TrivialFusion)
# --------------------------------------------------------------------------- #
def test_substrate_overfits_every_stream_trivial_fusion():
    _, history = _overfit("trivial", steps=500)
    for key in ("loss_video", "loss_proprio", "loss_tactile", "loss"):
        first = sum(h[key] for h in history[:25]) / 25
        last = sum(h[key] for h in history[-25:]) / 25
        assert last < 0.6 * first, f"{key}: {last:.4f} not < 0.6*{first:.4f}"


# --------------------------------------------------------------------------- #
# Phase 3 — compositional learned-mask fusion
# --------------------------------------------------------------------------- #
def test_compositional_fusion_learns_and_overfits():
    components, history = _overfit("compositional", steps=500)
    fusion = components.model.fusion
    assert isinstance(fusion, LearnedMaskFusion)
    # mask has n+2 slots: base + action + (proprio, tactile) video-adjustments
    assert fusion.logits.numel() == 4
    assert fusion.logits.grad is not None  # mask received gradient
    weights = fusion.mask_weights()
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    for key in ("loss_video", "loss_proprio", "loss_tactile"):
        first = sum(h[key] for h in history[:25]) / 25
        last = sum(h[key] for h in history[-25:]) / 25
        assert last < 0.7 * first, f"{key}: {last:.4f} not < 0.7*{first:.4f}"


def _dummy_modalities():
    """Video-as-vector substrate spec for the dummy base (feature_dim=8)."""
    return [
        OutputModalitySpec(name="video", kind="vector", feature_shape=[8], has_frozen_prior=True, loss_weight=1.0),
        OutputModalitySpec(name="proprio", kind="vector", feature_shape=[7], loss_weight=0.5),
        OutputModalitySpec(name="tactile", kind="map", feature_shape=[2, 8, 8], codec="resize",
                           codec_kwargs={"size": [8, 8]}, loss_weight=0.2),
    ]


def _dummy_batch(batch_size=4):
    return {
        "video": torch.randn(batch_size, 8),
        "proprio": torch.randn(batch_size, 7),
        "tactile": torch.randn(batch_size, 2, 8, 8),
        "act": torch.randn(batch_size, 4),
    }


# --------------------------------------------------------------------------- #
# Phase 0 — spec, codecs, config
# --------------------------------------------------------------------------- #
def test_output_modality_spec_validates_kind_and_shape():
    spec = OutputModalitySpec(name="proprio", kind="vector", feature_shape=[7], loss_weight=0.5)
    assert spec.feature_shape == (7,)
    assert spec.loss_weight == 0.5

    try:
        OutputModalitySpec(name="bad", kind="nonsense")
    except ValueError as exc:
        assert "kind" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown kind")

    try:
        OutputModalitySpec(name="proprio", kind="vector")  # missing feature_shape
    except ValueError as exc:
        assert "feature_shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing feature_shape")


def test_identity_codec_roundtrip():
    codec = IdentityCodec(mean=[0.0, 1.0, 2.0], std=[1.0, 2.0, 4.0])
    raw = torch.randn(2, 5, 3)  # (B, T, C)
    target = codec.encode(raw)
    recovered = codec.decode(target)
    assert torch.allclose(recovered, raw, atol=1e-5)


def test_resize_codec_roundtrip_shapes():
    codec = ResizeCodec(size=(8, 8))
    raw = torch.randn(2, 4, 16, 16)  # (B, T, H, W)
    target = codec.encode(raw)
    assert target.shape == (2, 4, 8, 8)
    recovered = codec.decode(target)
    assert recovered.shape == raw.shape


def test_multimodal_config_from_dict_partitions_streams():
    data = {
        "name": "mm-test",
        "model": {"type": "diffusion", "provider": "dummy", "feature_dim": 8, "hidden_dim": 16},
        "adapter": {"type": "output", "extra": {"backbone": "mlp", "output_channels": 8}},
        "conditioning": {"type": "action", "input_dim": 4, "output_dim": 16},
        "output_modalities": [
            {"name": "video", "kind": "video", "has_frozen_prior": True, "loss_weight": 1.0},
            {"name": "proprio", "kind": "vector", "feature_shape": [7], "loss_weight": 0.5},
            {"name": "tactile", "kind": "map", "feature_shape": [2, 8, 8], "loss_weight": 0.2},
        ],
    }
    config = MultiModalExperimentConfig.from_dict(data)
    assert config.video_modality.name == "video"
    assert [m.name for m in config.adapter_modalities] == ["proprio", "tactile"]
    assert config.base.model.provider == "dummy"


# --------------------------------------------------------------------------- #
# Phase 1 — preprocessor (dummy path, no DynamiCrafter)
# --------------------------------------------------------------------------- #
def test_preprocessor_builds_target_and_cond_dicts():
    specs = _dummy_modalities()
    codecs = {
        "video": IdentityCodec(),
        "proprio": IdentityCodec(),
        "tactile": ResizeCodec(size=(8, 8)),
    }
    pre = MultiModalBatchPreprocessor(specs, codecs, video_preprocessor=None, condition_keys=("act",))
    out = pre(_dummy_batch(batch_size=4), train=True)

    assert set(out["targets"]) == {"video", "proprio", "tactile"}
    assert out["targets"]["video"].shape == (4, 8)
    assert out["targets"]["proprio"].shape == (4, 7)
    assert out["targets"]["tactile"].shape == (4, 2, 8, 8)
    assert "act" in out["cond"]
    assert out["cond"]["act"].shape == (4, 4)
