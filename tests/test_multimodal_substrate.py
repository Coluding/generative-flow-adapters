"""Phase 0–2 substrate tests for the multimodal compositional adapter.

Runs entirely on the lightweight ``DummyVectorField`` base (no DynamiCrafter /
GPU), so the multi-stream contract, codecs, per-stream noising, and summed loss
are validated cheaply. The video stream is modelled as just another stream with
``has_frozen_prior=True``.
"""

from __future__ import annotations

import torch

from generative_flow_adapters.multimodal.codecs import IdentityCodec, ResizeCodec
from generative_flow_adapters.multimodal.config import MultiModalExperimentConfig
from generative_flow_adapters.multimodal.spec import OutputModalitySpec


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
