"""Architecture tests for the UniCon hidden-state adapter family.

Mirrors ``test_hyperalign_architecture.py``: builds a tiny stand-in U-Net
that exposes the same surface UniCon expects (``input_blocks``,
``middle_block``, ``output_blocks``, ``out`` / ``out_mask``,
``input_block_chans``, ``time_embed``, etc.), then drives each variant
through ``AdaptedModel`` to verify shape, composition modes, the new
step-level conditioning branch, and that the captured-features path
behaves end-to-end.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.hidden_states.unicon import (
    FullSkipLayerControlAdapter,
    ReplaceDecoderHiddenStateAdapter,
    UniConHiddenStateAdapter,
    ZeroConvConnector,
    ZeroFTConnector,
)
from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.models.adapted_model import AdaptedModel


# ----------------------------------------------------------------------------
# Fake U-Net surface
# ----------------------------------------------------------------------------


_TIME_EMBED_DIM = 16


class FakeUNetBlock(nn.Module):
    """A U-Net block stand-in. Applies a Conv2d and adds a learned projection
    of the timestep embedding so the block actually depends on ``emb``
    (otherwise step-level conditioning has no observable effect on the
    output). Exposes ``out_channels`` for ``_infer_block_channels`` and
    ``channels`` for ``_infer_middle_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(_TIME_EMBED_DIM, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = in_channels

    def forward(self, h: Tensor, emb=None, context=None, batch_size=None) -> Tensor:
        del context, batch_size
        out = self.conv(h)
        if isinstance(emb, Tensor):
            out = out + self.emb_proj(emb).view(emb.shape[0], -1, 1, 1)
        return out


class FakeBlockWrapper(nn.Module):
    """Mimics a DynamiCrafter TimestepEmbedSequential: callable with the U-Net
    block signature AND iterable so the adapter's channel inference can walk
    sublayers looking for ``out_channels``."""

    def __init__(self, layer: FakeUNetBlock) -> None:
        super().__init__()
        self.layer = layer

    def __iter__(self):
        yield self.layer

    def forward(self, h: Tensor, emb=None, context=None, batch_size=None) -> Tensor:
        return self.layer(h, emb=emb, context=context, batch_size=batch_size)


class FakeMiddleBlock(nn.Module):
    """Iterable, indexable middle block. ``middle_block[0].channels`` is the
    only attribute UniCon's ``_infer_middle_channels`` reads."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = FakeUNetBlock(channels, channels)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self.block

    def forward(self, h: Tensor, emb=None, context=None, batch_size=None) -> Tensor:
        return self.block(h, emb=emb, context=context, batch_size=batch_size)


class FakeOutputHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, h: Tensor) -> Tensor:
        return self.conv(h)


class FakeVideoUNet(nn.Module):
    """Minimal U-Net surface the UniCon adapters rely on.

    Shapes are tiny:
      input_blocks:  [Conv(4→8), Conv(8→16)]
      middle_block:  Conv(16→16)
      output_blocks: [Conv(16+16→8), Conv(8+8→4)]
      out:           Conv1x1(4 → 4)
      out_mask:      Conv1x1(4 → 4)
      input_block_chans: [8, 16]   (channels at each input block's output)
    """

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = 8
        self.action_conditioned = False
        self.action_dropout_prob = 0.0
        self.fs_condition = False

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, _TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(_TIME_EMBED_DIM, _TIME_EMBED_DIM),
        )

        self.input_blocks = nn.ModuleList(
            [
                FakeBlockWrapper(FakeUNetBlock(4, 8)),
                FakeBlockWrapper(FakeUNetBlock(8, 16)),
            ]
        )
        self.input_block_chans = [8, 16]
        self.middle_block = FakeMiddleBlock(16)
        self.output_blocks = nn.ModuleList(
            [
                FakeBlockWrapper(FakeUNetBlock(16 + 16, 8)),
                FakeBlockWrapper(FakeUNetBlock(8 + 8, 4)),
            ]
        )
        self.out = FakeOutputHead(4, 4)
        self.out_mask = FakeOutputHead(4, 4)
        self.in_channels = 4
        self.out_channels = 4

    def forward(self, x_t: Tensor, timesteps: Tensor, **kwargs) -> Tensor:
        """Flatten frames into the batch axis, run encoder/middle/decoder with
        skip connections (so the hooks capture matching activations), then
        un-flatten to ``[B, C, T, H, W]``."""
        del kwargs
        batch_size, channels, frames, height, width = x_t.shape
        h = x_t.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        emb = torch.zeros(batch_size * frames, self.time_embed[-1].out_features, dtype=x_t.dtype, device=x_t.device)
        skips: list[Tensor] = []
        for block in self.input_blocks:
            h = block(h, emb)
            skips.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, emb)
        out = self.out(h)
        return out.reshape(batch_size, frames, self.out_channels, height, width).permute(0, 2, 1, 3, 4)


class FakeBaseModel(nn.Module):
    """The thin wrapper UniCon's ``_resolve_unet_module`` walks through to get
    at the inner ``module``."""

    def __init__(self) -> None:
        super().__init__()
        self.module = FakeVideoUNet()
        self.diffusion_schedule_config = SimpleNamespace(timesteps=1000)
        self.model_type = "diffusion"
        self.prediction_type = "velocity"

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        if not isinstance(cond, dict):
            raise ValueError("UniCon tests expect a mapping condition with a 'context' tensor.")
        return self.module(x_t, timesteps=t, context=cond.get("context"))


def _build_inputs():
    x_t = torch.randn(2, 4, 3, 8, 8)
    cond = {"context": torch.randn(2, 5, 12)}
    return x_t, cond


def _attach(adapter, base=None):
    base = base if base is not None else FakeBaseModel()
    adapter.attach_base_model(base)
    return adapter, base


# ----------------------------------------------------------------------------
# Connectors (the adapter-side fusion machinery)
# ----------------------------------------------------------------------------


class ConnectorInitializationTest(unittest.TestCase):
    """Zero-init connectors are the safety guarantee that lets us attach a
    fresh adapter to a frozen base without distorting the base's behaviour at
    step 0. If these inits drift the whole family loses its 'identity at
    init' property."""

    def test_zero_conv_starts_as_identity(self):
        connector = ZeroConvConnector(channels=4)
        target = torch.randn(2, 4, 8, 8)
        source = torch.randn(2, 4, 8, 8)
        self.assertTrue(torch.allclose(connector(target, source), target))

    def test_zero_ft_starts_as_identity(self):
        connector = ZeroFTConnector(channels=4)
        target = torch.randn(2, 4, 8, 8)
        source = torch.randn(2, 4, 8, 8)
        self.assertTrue(torch.allclose(connector(target, source), target))


# ----------------------------------------------------------------------------
# UniCon decoder-focused variant — the canonical Figure-3(d) replication
# ----------------------------------------------------------------------------


class UniConDecoderArchitectureTest(unittest.TestCase):
    def test_attaches_and_replicates_decoder_only(self):
        adapter, _ = _attach(UniConHiddenStateAdapter(cond_dim=12, output_kind="prediction"))
        # Decoder + heads only; no encoder replica.
        self.assertEqual(len(adapter.decoder_blocks), 2)
        self.assertEqual(len(adapter.skip_connectors), len(adapter._require_module().input_block_chans))
        self.assertEqual(len(adapter.decoder_connectors), 2)
        self.assertIsNotNone(adapter.out_head)
        self.assertIsNone(adapter.mask_head)

    def test_forward_returns_prediction_shape(self):
        adapter, base = _attach(UniConHiddenStateAdapter(cond_dim=12, output_kind="prediction"))
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)

        with torch.no_grad():
            base(x_t, t, cond=cond)  # populate the feature store via hooks
        result = adapter(x_t, t, cond)

        self.assertIsInstance(result, OutputAdapterResult)
        self.assertEqual(result.output_kind, "prediction")
        self.assertIsNone(result.gate)
        self.assertEqual(tuple(result.adapter_output.shape), tuple(x_t.shape))

    def test_output_mask_emits_gate_of_prediction_shape(self):
        adapter, base = _attach(UniConHiddenStateAdapter(cond_dim=12, output_mask=True))
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)

        with torch.no_grad():
            base(x_t, t, cond=cond)
        result = adapter(x_t, t, cond)

        self.assertIsInstance(result, OutputAdapterResult)
        self.assertIsNotNone(result.gate)
        self.assertEqual(tuple(result.gate.shape), tuple(x_t.shape))

    def test_feature_store_clears_on_request(self):
        adapter, base = _attach(UniConHiddenStateAdapter(cond_dim=12))
        x_t, cond = _build_inputs()
        with torch.no_grad():
            base(x_t, torch.full((2,), 999, dtype=torch.long), cond=cond)
        self.assertEqual(len(adapter._feature_store.input_activations), 2)
        adapter.clear_captured_base_features()
        self.assertEqual(len(adapter._feature_store.input_activations), 0)
        self.assertIsNone(adapter._feature_store.middle)


# ----------------------------------------------------------------------------
# Composition modes via AdaptedModel — same contract used by HyperAlign/AVID.
# ----------------------------------------------------------------------------


class UniConCompositionTest(unittest.TestCase):
    """UniCon plays through AdaptedModel's ``add``/``replace``/``mask_mix``
    just like the other families. We don't re-test the composition logic
    itself (that's AdaptedModel's job) — only that UniCon emits the right
    shape/kind for each."""

    def _run_adapted(self, *, composition: str, output_mask: bool) -> Tensor:
        base = FakeBaseModel()
        adapter = UniConHiddenStateAdapter(cond_dim=12, output_mask=output_mask)
        model = AdaptedModel(base_model=base, adapter=adapter, output_composition=composition)
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)
        return model(x_t, t, cond)

    def test_add_composition_returns_full_shape(self):
        prediction = self._run_adapted(composition="add", output_mask=False)
        self.assertEqual(tuple(prediction.shape), (2, 4, 3, 8, 8))

    def test_replace_composition_returns_adapter_only(self):
        prediction = self._run_adapted(composition="replace", output_mask=False)
        self.assertEqual(tuple(prediction.shape), (2, 4, 3, 8, 8))

    def test_mask_mix_blends_base_and_adapted(self):
        prediction = self._run_adapted(composition="mask_mix", output_mask=True)
        self.assertEqual(tuple(prediction.shape), (2, 4, 3, 8, 8))
        # Backward must run cleanly through the gate path.
        prediction.sum().backward()


# ----------------------------------------------------------------------------
# Step-level conditioning — the shortcut training entry point.
# ----------------------------------------------------------------------------


class UniConStepLevelConditioningTest(unittest.TestCase):
    def test_disabled_by_default_no_step_level_embed(self):
        adapter = UniConHiddenStateAdapter(cond_dim=12)
        self.assertIsNone(adapter.step_level_embed)

    def test_enabled_builds_linear_embed_of_correct_shape(self):
        adapter = UniConHiddenStateAdapter(
            cond_dim=12, use_step_level_conditioning=True, step_level_hidden_dim=8
        )
        self.assertIsNotNone(adapter.step_level_embed)
        scalar = torch.tensor([[0.5]])
        embedded = adapter.step_level_embed(scalar)
        self.assertEqual(tuple(embedded.shape), (1, 12))

    def test_requires_positive_cond_dim_when_enabled(self):
        with self.assertRaises(ValueError):
            UniConHiddenStateAdapter(cond_dim=None, use_step_level_conditioning=True)
        with self.assertRaises(ValueError):
            UniConHiddenStateAdapter(cond_dim=0, use_step_level_conditioning=True)

    def test_step_level_flows_into_adapter_embedding(self):
        """The structured-condition embedding (cond['embedding']) plus the
        step-level scalar must both reach the adapter's ``emb_fuse`` head.
        We mark the step_level_embed weights and check the resulting fused
        embedding changes when step_level changes."""
        adapter, base = _attach(UniConHiddenStateAdapter(
            cond_dim=12, use_step_level_conditioning=True, step_level_hidden_dim=8,
            step_level_transform="linear",
        ))
        # Initialise the *last* layer of step_level_embed away from zero so its
        # output materially perturbs the fused embedding.
        with torch.no_grad():
            for parameter in adapter.step_level_embed[-1].parameters():
                parameter.fill_(0.1)
            # emb_fuse final layer is zero-init by _prepare_adapter_conditioning;
            # nudge it so the step_level signal can propagate.
            adapter.emb_fuse[-1].weight.fill_(0.05)
            adapter.emb_fuse[-1].bias.fill_(0.0)

        x_t, cond = _build_inputs()
        cond["embedding"] = torch.randn(2, 12)
        t = torch.full((2,), 999, dtype=torch.long)

        with torch.no_grad():
            base(x_t, t, cond=cond)
        cond_low = dict(cond, step_level=torch.tensor([0.01, 0.01]))
        result_low = adapter(x_t, t, cond_low).adapter_output

        adapter.clear_captured_base_features()
        with torch.no_grad():
            base(x_t, t, cond=cond)
        cond_high = dict(cond, step_level=torch.tensor([1.0, 1.0]))
        result_high = adapter(x_t, t, cond_high).adapter_output

        self.assertFalse(torch.allclose(result_low, result_high))


# ----------------------------------------------------------------------------
# Sibling variants — share most infrastructure but worth a smoke test each.
# ----------------------------------------------------------------------------


class ReplaceDecoderArchitectureTest(unittest.TestCase):
    def test_attaches_with_no_connectors(self):
        adapter, _ = _attach(ReplaceDecoderHiddenStateAdapter(cond_dim=12))
        # No skip_connectors / decoder_connectors — the captured features go
        # straight into the trainable decoder copy.
        self.assertFalse(hasattr(adapter, "skip_connectors"))
        self.assertFalse(hasattr(adapter, "decoder_connectors"))
        self.assertEqual(len(adapter.decoder_blocks), 2)

    def test_forward_returns_prediction_shape(self):
        adapter, base = _attach(ReplaceDecoderHiddenStateAdapter(cond_dim=12))
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)

        with torch.no_grad():
            base(x_t, t, cond=cond)
        result = adapter(x_t, t, cond)

        self.assertEqual(tuple(result.adapter_output.shape), tuple(x_t.shape))


class FullSkipControlNetArchitectureTest(unittest.TestCase):
    def test_attaches_with_full_replica(self):
        adapter, _ = _attach(FullSkipLayerControlAdapter(cond_dim=12))
        # Full ControlNet-style replica: encoder, middle, decoder all
        # trainable copies of the base.
        self.assertEqual(len(adapter.input_blocks), 2)
        self.assertIsNotNone(adapter.middle_block)
        self.assertEqual(len(adapter.output_blocks), 2)
        # One connector per encoder block, the middle, and each decoder block.
        self.assertEqual(len(adapter.input_connectors), 2)
        self.assertIsNotNone(adapter.middle_connector)
        self.assertEqual(len(adapter.output_connectors), 2)

    def test_forward_returns_prediction_shape(self):
        adapter, base = _attach(FullSkipLayerControlAdapter(cond_dim=12))
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)

        with torch.no_grad():
            base(x_t, t, cond=cond)
        result = adapter(x_t, t, cond)

        self.assertEqual(tuple(result.adapter_output.shape), tuple(x_t.shape))


if __name__ == "__main__":
    unittest.main()
