"""Tests for the output-format ablation machinery.

Covers the shared ``format.py`` math (direct vs. affine, dense vs. channel
granularity) and the new ``OutputHeadAdapter`` (mlp / transformer backbones),
end-to-end through ``AdaptedModel``. The DynamiCrafter ``unet`` backbone is
exercised by the existing DynamiCrafter tests; here we only verify the format
helper it now delegates to.
"""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.output.format import (
    build_output_result,
    normalize_affine_granularity,
    normalize_output_format,
    output_channel_multiplier,
)
from generative_flow_adapters.adapters.output.output_head import OutputHeadAdapter
from generative_flow_adapters.models.adapted_model import AdaptedModel


class _FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_type = "diffusion"
        self.prediction_type = "velocity"
        self._scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        return x_t * 2.0  # an arbitrary deterministic "base prediction"


def _video_inputs():
    x_t = torch.randn(2, 4, 3, 8, 8)
    cond = {"embedding": torch.randn(2, 16)}
    t = torch.full((2,), 500, dtype=torch.long)
    return x_t, t, cond


class FormatNormalizationTest(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_output_format("delta"), "direct")
        self.assertEqual(normalize_output_format("scale_shift"), "affine")
        self.assertEqual(normalize_affine_granularity("per_channel"), "channel")
        self.assertEqual(normalize_affine_granularity("spatial"), "dense")

    def test_multiplier(self):
        self.assertEqual(output_channel_multiplier("direct"), 1)
        self.assertEqual(output_channel_multiplier("affine"), 2)

    def test_rejects_unknown(self):
        with self.assertRaises(ValueError):
            normalize_output_format("bogus")


class FormatMathTest(unittest.TestCase):
    def test_direct_passes_raw_through_as_delta(self):
        raw = torch.randn(2, 4, 3, 8, 8)
        result = build_output_result(raw, reference=None, output_format="direct")
        self.assertEqual(result.output_kind, "delta")
        self.assertTrue(torch.equal(result.adapter_output, raw))

    def test_affine_dense_computes_scale_shift(self):
        reference = torch.randn(2, 4, 3, 8, 8)
        scale = torch.randn(2, 4, 3, 8, 8)
        shift = torch.randn(2, 4, 3, 8, 8)
        raw = torch.cat([scale, shift], dim=1)
        result = build_output_result(raw, reference, output_format="affine", affine_granularity="dense")
        self.assertEqual(result.output_kind, "delta")
        self.assertTrue(torch.allclose(result.adapter_output, reference * scale + shift))

    def test_affine_channel_pools_spatial(self):
        reference = torch.randn(2, 4, 3, 8, 8)
        scale = torch.randn(2, 4, 3, 8, 8)
        shift = torch.randn(2, 4, 3, 8, 8)
        raw = torch.cat([scale, shift], dim=1)
        result = build_output_result(raw, reference, output_format="affine", affine_granularity="channel")
        pooled_scale = scale.mean(dim=(2, 3, 4), keepdim=True)
        pooled_shift = shift.mean(dim=(2, 3, 4), keepdim=True)
        self.assertTrue(torch.allclose(result.adapter_output, reference * pooled_scale + pooled_shift))

    def test_affine_requires_reference(self):
        raw = torch.randn(2, 8, 3, 8, 8)
        with self.assertRaises(ValueError):
            build_output_result(raw, None, output_format="affine")

    def test_affine_rejects_odd_channels(self):
        raw = torch.randn(2, 3, 3, 8, 8)
        with self.assertRaises(ValueError):
            build_output_result(raw, torch.randn(2, 3, 3, 8, 8), output_format="affine")


class OutputHeadAdapterTest(unittest.TestCase):
    def _run(self, backbone: str, output_format: str, affine_granularity: str = "dense", condition_on_base_outputs: bool = True):
        adapter = OutputHeadAdapter(
            feature_dim=4,
            cond_dim=16,
            hidden_dim=32,
            backbone=backbone,
            output_format=output_format,
            affine_granularity=affine_granularity,
            condition_on_base_outputs=condition_on_base_outputs,
            patch_size=2,
            num_layers=2,
            num_heads=4,
        )
        base = _FakeBaseModel()
        model = AdaptedModel(base_model=base, adapter=adapter, output_composition="add")
        x_t, t, cond = _video_inputs()
        return adapter, base, model, x_t, t, cond

    def test_mlp_affine_shape_and_identity_at_init(self):
        adapter, base, model, x_t, t, cond = self._run("mlp", "affine", "channel")
        out = model(x_t, t, cond)
        self.assertEqual(tuple(out.shape), tuple(x_t.shape))
        # Zero-init head => delta 0 => composed output equals the base prediction.
        self.assertTrue(torch.allclose(out, base(x_t, t, cond), atol=1e-6))

    def test_mlp_direct_shape(self):
        _, _, model, x_t, t, cond = self._run("mlp", "direct")
        self.assertEqual(tuple(model(x_t, t, cond).shape), tuple(x_t.shape))

    def test_transformer_affine_dense_shape_and_identity(self):
        adapter, base, model, x_t, t, cond = self._run("transformer", "affine", "dense")
        out = model(x_t, t, cond)
        self.assertEqual(tuple(out.shape), tuple(x_t.shape))
        self.assertTrue(torch.allclose(out, base(x_t, t, cond), atol=1e-6))

    def test_transformer_direct_shape(self):
        _, _, model, x_t, t, cond = self._run("transformer", "direct")
        self.assertEqual(tuple(model(x_t, t, cond).shape), tuple(x_t.shape))

    def test_transformer_without_base_output_conditioning(self):
        _, _, model, x_t, t, cond = self._run("transformer", "affine", "dense", condition_on_base_outputs=False)
        self.assertEqual(tuple(model(x_t, t, cond).shape), tuple(x_t.shape))

    def test_affine_channel_count_doubles(self):
        affine = OutputHeadAdapter(feature_dim=4, cond_dim=16, hidden_dim=32, backbone="mlp", output_format="affine")
        direct = OutputHeadAdapter(feature_dim=4, cond_dim=16, hidden_dim=32, backbone="mlp", output_format="direct")
        self.assertEqual(affine.out_channels, 8)
        self.assertEqual(direct.out_channels, 4)

    def test_trains_a_step(self):
        # Backward should reach the adapter (and not the frozen base).
        adapter, base, model, x_t, t, cond = self._run("transformer", "affine", "dense")
        out = model(x_t, t, cond)
        loss = (out - torch.randn_like(out)).pow(2).mean()
        loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_rejects_unknown_backbone(self):
        with self.assertRaises(ValueError):
            OutputHeadAdapter(feature_dim=4, cond_dim=16, hidden_dim=32, backbone="unet")

    def test_rejects_unknown_gate_kind(self):
        with self.assertRaises(ValueError):
            OutputHeadAdapter(feature_dim=4, cond_dim=16, hidden_dim=32, backbone="mlp", gate_kind="bogus")


class GatedCompositionTest(unittest.TestCase):
    """The mixing layer offers two blends on top of add/replace; both read the
    adapter's gate. gated_residual treats the output as a contribution
    (base + g·Δ); mask_mix treats it as a standalone prediction."""

    def _adapter(self, gate_kind: str, output_format: str = "affine"):
        return OutputHeadAdapter(
            feature_dim=4,
            cond_dim=16,
            hidden_dim=32,
            backbone="transformer",
            output_format=output_format,
            gate_kind=gate_kind,
            patch_size=2,
            num_layers=2,
            num_heads=4,
        )

    def test_head_emits_gate_channels(self):
        adapter = self._adapter("dense", output_format="affine")
        # affine (2C) + gate (C) = 3C for C=4
        self.assertEqual(adapter.out_channels, 12)
        x_t, t, cond = _video_inputs()
        result = adapter(x_t, t, cond)
        self.assertIsNotNone(result.gate)
        self.assertEqual(result.gate.shape[1], 4)

    def test_channel_gate_is_pooled(self):
        adapter = self._adapter("channel", output_format="direct")
        x_t, t, cond = _video_inputs()
        result = adapter(x_t, t, cond)
        # channel gate pooled over T,H,W -> singleton spatial/temporal dims
        self.assertEqual(tuple(result.gate.shape), (2, 4, 1, 1, 1))

    def test_gated_residual_identity_at_init(self):
        adapter = self._adapter("dense")
        base = _FakeBaseModel()
        model = AdaptedModel(base_model=base, adapter=adapter, output_composition="gated_residual")
        x_t, t, cond = _video_inputs()
        out = model(x_t, t, cond)
        self.assertEqual(tuple(out.shape), tuple(x_t.shape))
        # Δ≈0 at init ⇒ base + g·0 = base, regardless of gate value.
        self.assertTrue(torch.allclose(out, base(x_t, t, cond), atol=1e-6))

    def test_gated_residual_requires_gate(self):
        adapter = self._adapter("none")  # no gate emitted
        base = _FakeBaseModel()
        model = AdaptedModel(base_model=base, adapter=adapter, output_composition="gated_residual")
        x_t, t, cond = _video_inputs()
        with self.assertRaises(ValueError):
            model(x_t, t, cond)


if __name__ == "__main__":
    unittest.main()
