"""Contract tests for the 2026-07 base-parity countermeasures:

- ``sigma_shift`` (Wan22DiffusionForcingPreprocessor): train-only timestep
  shift toward high noise.
- ``gate_cap`` (AdaptedModel, mask_mix): upper clamp on the post-sigmoid gate
  so the adapter branch keeps >= (1-cap) of the gradient.
- ``ACWMPhysTranslator``: emits the MetaWorld clip contract from the
  mp4+metadata.pt release layout.

No GPU, no VAE, no 5B — pure wiring contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from generative_flow_adapters.data.wan_batch_preprocessor import WanBatchPreprocessConfig


def _shift(sigma: torch.Tensor, s: float) -> torch.Tensor:
    return s * sigma / (1.0 + (s - 1.0) * sigma)


class TestSigmaShift:
    def test_config_field_default_none(self):
        cfg = WanBatchPreprocessConfig(target_height=64, target_width=64)
        assert cfg.sigma_shift is None

    def test_shift_moves_mass_toward_high_sigma(self):
        torch.manual_seed(0)
        u = torch.rand(200_000)
        sh = _shift(u, 5.0)
        assert 0.80 < sh.median() < 0.87  # analytic median at s=5: 5/6
        assert (sh > 0.7).float().mean() > 0.6
        assert sh.min() >= 0 and sh.max() <= 1

    def test_shift_of_one_is_identity(self):
        u = torch.rand(1000)
        assert torch.allclose(_shift(u, 1.0), u)

    def test_preprocessor_applies_shift_train_only(self):
        """The wan22 preprocessor's __call__ shifts sigma when train=True and
        sigma_shift is set, and leaves eval batches U(0,1). We test the sigma
        branch in isolation by monkeypatching the encode away."""
        from generative_flow_adapters.data.wan22_batch_preprocessor import (
            Wan22DiffusionForcingPreprocessor,
        )

        cfg = WanBatchPreprocessConfig(target_height=32, target_width=32, sigma_shift=5.0)
        pre = Wan22DiffusionForcingPreprocessor.__new__(Wan22DiffusionForcingPreprocessor)
        pre.config = cfg
        pre.cond_frames = 1
        pre._cond_values = None
        pre._cond_weights = None
        pre.condition_keys = ()
        z0 = torch.randn(64, 4, 5, 8, 8)
        pre._encode_z0 = lambda *a, **k: z0
        pre._build_condition = lambda *a, **k: {}

        torch.manual_seed(0)
        train_t = Wan22DiffusionForcingPreprocessor.__call__(
            pre, {"video": torch.zeros(64, 1, 1, 1, 1)}, train=True
        )["t"]
        torch.manual_seed(0)
        eval_t = Wan22DiffusionForcingPreprocessor.__call__(
            pre, {"video": torch.zeros(64, 1, 1, 1, 1)}, train=False
        )["t"]
        # t = frame_mask * sigma * 1000; frame 0 is the clean obs frame.
        train_sigma = train_t[:, 1:].amax(dim=1) / 1000.0
        eval_sigma = eval_t[:, 1:].amax(dim=1) / 1000.0
        # Same seed => train sigmas are exactly shift(eval sigmas).
        assert torch.allclose(train_sigma, _shift(eval_sigma, 5.0), atol=1e-5)
        assert train_sigma.median() > eval_sigma.median()


class TestGateCap:
    def _model(self, gate_cap):
        from generative_flow_adapters.models.adapted_model import AdaptedModel

        m = AdaptedModel.__new__(AdaptedModel)
        m.output_composition = "mask_mix"
        m.gate_bias = 0.0
        m.gate_cap = gate_cap
        m._last_gate = None
        return m

    def _result(self, gate_logit: float):
        from generative_flow_adapters.adapters.output.interface import OutputAdapterResult

        return OutputAdapterResult(
            adapter_output=torch.ones(2, 3, 4),
            output_kind="prediction",
            gate=torch.full((2, 3, 4), gate_logit),
        )

    def test_uncapped_gate_can_saturate(self):
        m = self._model(None)
        out = m._compose(base_output=torch.zeros(2, 3, 4), adapter_result=self._result(8.0))
        assert m._last_gate.max() > 0.99
        assert out.abs().max() < 0.01  # ~all base (zeros)

    def test_cap_clamps_gate_and_preserves_adapter_share(self):
        m = self._model(0.9)
        out = m._compose(base_output=torch.zeros(2, 3, 4), adapter_result=self._result(8.0))
        assert m._last_gate.max() <= 0.9 + 1e-6
        # adapter contributes (1-gate) >= 0.1 of its ones-output
        assert out.min() >= 0.1 - 1e-6

    def test_cap_inactive_below_threshold(self):
        capped = self._model(0.9)
        uncapped = self._model(None)
        res = self._result(0.0)  # sigmoid(0)=0.5 < cap
        base = torch.randn(2, 3, 4)
        assert torch.allclose(
            capped._compose(base_output=base, adapter_result=res),
            uncapped._compose(base_output=base, adapter_result=res),
        )

    def test_config_plumbing(self):
        from generative_flow_adapters.config import AdapterConfig

        assert AdapterConfig(type="output").gate_cap is None


class TestBaseCosineDiagnostics:
    def test_masked_cosine_uses_predicted_frames_only(self):
        from generative_flow_adapters.training.trainer import Trainer

        a = torch.zeros(1, 2, 3, 2, 2)
        b = torch.zeros(1, 2, 3, 2, 2)
        # frame 0 (obs): orthogonal junk that would drag the cosine down.
        a[:, 0, 0], b[:, 1, 0] = 5.0, 5.0
        # frames 1-2 (predicted): identical -> cosine 1.
        a[:, :, 1:] = 1.0
        b[:, :, 1:] = 1.0
        batch = {"frame_mask": torch.tensor([[0.0, 1.0, 1.0]])}
        assert Trainer._masked_cosine(a, b, batch) == pytest.approx(1.0)
        assert Trainer._masked_cosine(a, b, {}) < 0.9  # unmasked includes the junk

    def test_compose_captures_raw_adapter_branch(self):
        from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
        from generative_flow_adapters.models.adapted_model import AdaptedModel

        m = AdaptedModel.__new__(AdaptedModel)
        m.output_composition = "mask_mix"
        m.gate_bias = 0.0
        m.gate_cap = None
        m._last_gate = None
        m._last_adapter_out = None
        pred = torch.randn(2, 3, 4)
        res = OutputAdapterResult(adapter_output=pred, output_kind="prediction",
                                  gate=torch.full((2, 3, 4), 8.0))
        composed = m._compose(base_output=torch.zeros(2, 3, 4), adapter_result=res)
        # Gate ~1 -> composed ≈ base (zeros), but the captured branch is the raw pred.
        assert composed.abs().max() < 0.05 * pred.abs().max()
        assert torch.equal(m._last_adapter_out, pred)
        assert not m._last_adapter_out.requires_grad


class TestACWMPhysTranslator:
    @pytest.fixture()
    def split_dir(self, tmp_path):
        try:
            import imageio.v3 as iio
        except ImportError:
            pytest.skip("imageio not available")
        d = tmp_path / "rigid_dynamics" / "push_block" / "ind_train"
        d.mkdir(parents=True)
        frames = (np.random.rand(20, 64, 64, 3) * 255).astype(np.uint8)
        iio.imwrite(d / "episode_0.mp4", frames, fps=10)
        meta = [{
            "video_path": "episode_0.mp4",
            "actions": torch.randn(20, 2),
            "length": 20,
        }]
        torch.save(meta, d / "metadata.pt")
        return d

    def test_contract(self, split_dir):
        from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator

        tr = ACWMPhysTranslator(str(split_dir))
        eps = tr.list_episodes()
        assert len(eps) == 1 and eps[0].length == 20
        assert tr.env_name == "push_block-ind_train"

        clip = tr.load_clip(eps[0], start=2, length=8, stride=1)
        assert clip["video"].shape[0] == 8 and clip["video"].dtype == np.uint8
        assert clip["act"].shape == (8, 2) and clip["act"].dtype == torch.float32
        # latent-cache identity fields present and correct
        assert clip["env_name"] == "push_block-ind_train"
        assert clip["episode_idx"] == 0 and clip["start_idx"] == 2 and clip["frame_stride"] == 1

    def test_stride_sums_actions(self, split_dir):
        from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator

        tr = ACWMPhysTranslator(str(split_dir))
        ep = tr.list_episodes()[0]
        clip = tr.load_clip(ep, start=0, length=5, stride=2)
        raw = tr._meta[0]["actions"][:10].to(torch.float32)
        expected = raw.reshape(5, 2, -1).sum(dim=1)
        assert torch.allclose(clip["act"], expected)

    def test_out_of_range_raises(self, split_dir):
        from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator

        tr = ACWMPhysTranslator(str(split_dir))
        ep = tr.list_episodes()[0]
        with pytest.raises(IndexError):
            tr.load_clip(ep, start=15, length=8, stride=1)

    def test_dataset_windows(self, split_dir):
        from generative_flow_adapters.data.dataset import TranslatedClipDataset
        from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator

        tr = ACWMPhysTranslator(str(split_dir))
        ds = TranslatedClipDataset(tr, window_width=8, num_windows=4, sampling="random")
        enum = ds.fixed_window_enumeration()
        assert len(enum) == 4
        starts = [s for _, s in enum._pairs]
        assert starts == sorted(set(starts))  # deterministic, distinct, sorted
