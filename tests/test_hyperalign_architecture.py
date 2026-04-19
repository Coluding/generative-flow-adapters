from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from generative_flow_adapters.adapters.hypernetworks.hyperalign import HyperAlignAdapter
from generative_flow_adapters.adapters.low_rank.common import PAPER_HYPERALIGN_TARGET_MODULES


class FakeInputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, emb, context=None, batch_size=None):
        del emb, context, batch_size
        return self.conv(x)


class FakeAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Identity())

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.to_out(self.to_q(x) + self.to_k(context) + self.to_v(context))


class FakeVideoUNet(nn.Module):
    def __init__(self, channels: int = 4, model_channels: int = 8, context_dim: int = 12) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.model_channels = model_channels
        self.action_conditioned = False
        self.action_dropout_prob = 0.0
        self.fs_condition = False
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 2),
            nn.SiLU(),
            nn.Linear(model_channels * 2, model_channels * 2),
        )
        self.input_blocks = nn.ModuleList(
            [
                FakeInputBlock(channels, model_channels),
                FakeInputBlock(model_channels, context_dim),
            ]
        )
        self.attn_a = FakeAttention(context_dim)
        self.attn_b = FakeAttention(context_dim)
        self.output_proj = nn.Linear(context_dim, channels, bias=False)

    def forward(self, x, timesteps, context=None, act=None, dropout_actions=True, fs=None, **kwargs):
        del timesteps, act, dropout_actions, fs, kwargs
        batch_size, channels, frames, height, width = x.shape
        h = x.reshape(batch_size * frames, channels, height, width)
        emb = torch.zeros(batch_size * frames, self.time_embed[-1].out_features, device=x.device, dtype=x.dtype)
        repeated_context = context.repeat_interleave(frames, dim=0)
        for block in self.input_blocks:
            h = block(h, emb, context=repeated_context, batch_size=batch_size)
        pooled = h.mean(dim=(2, 3))
        context_summary = repeated_context.mean(dim=1)
        pooled = pooled + self.attn_a(pooled, context_summary) + self.attn_b(pooled, context_summary)
        output = self.output_proj(pooled)
        output = output.reshape(batch_size, frames, channels).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return output.expand(batch_size, channels, frames, height, width)


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = FakeVideoUNet()
        self.diffusion_schedule_config = SimpleNamespace(timesteps=1000)

    def forward(self, x_t, t, cond=None):
        if cond is None:
            raise ValueError("Fake HyperAlign tests expect a condition mapping.")
        return self.module(x_t, timesteps=t, context=cond["context"], act=cond.get("act"), fs=cond.get("fs"))


def _build_adapter(
    update_mode: str = "stepwise",
    *,
    cond_dim: int = 12,
    use_step_level_conditioning: bool = False,
) -> HyperAlignAdapter:
    adapter = HyperAlignAdapter(
        rank=4,
        alpha=1.0,
        target_modules=list(PAPER_HYPERALIGN_TARGET_MODULES),
        cond_dim=cond_dim,
        hidden_dim=128,
        input_summary_dim=4,
        aux_down_dim=16,
        aux_up_dim=16,
        num_decoder_layers=2,
        num_decoder_heads=8,
        use_step_level_conditioning=use_step_level_conditioning,
        update_mode=update_mode,
        piecewise_progress_markers=(0.0, 0.2),
    )
    adapter.attach_base_model(FakeBaseModel())
    return adapter


def _build_inputs():
    x_t = torch.randn(2, 4, 3, 8, 8)
    cond = {"context": torch.randn(2, 5, 12)}
    return x_t, cond


class HyperAlignArchitectureTest(unittest.TestCase):
    def test_hyperalign_uses_exact_paper_targets_and_zero_initialized_b_aux(self):
        adapter = _build_adapter()
        self.assertEqual(len(adapter._handles), 8)
        self.assertEqual([handle.qualified_name.rsplit(".", maxsplit=1)[-1] for handle in adapter._handles].count("to_q"), 2)
        for handle in adapter._handles:
            self.assertEqual(handle.wrapped.hyper_aux_dims, (16, 16))
            self.assertIsNotNone(handle.wrapped.up_aux)
            self.assertEqual(int(torch.count_nonzero(handle.wrapped.up_aux).item()), 0)

    def test_hyperalign_generates_fixed_shape_hyper_factors_and_prediction_delta(self):
        adapter = _build_adapter()
        x_t, cond = _build_inputs()
        t = torch.full((2,), 999, dtype=torch.long)

        factor_tokens = adapter.build_hyper_input(x_t=x_t, t=t, cond=cond)
        hyper_down, hyper_up = adapter._split_hyper_factors(factor_tokens)

        self.assertEqual(tuple(factor_tokens.shape), (2, len(adapter._handles), 128))
        self.assertEqual(tuple(hyper_down.shape), (2, len(adapter._handles), 16, 4))
        self.assertEqual(tuple(hyper_up.shape), (2, len(adapter._handles), 4, 16))

        delta = adapter(x_t, t, cond)
        self.assertEqual(tuple(delta.shape), tuple(x_t.shape))

    def test_hyperalign_initial_mode_reuses_factors_until_a_new_trajectory_starts(self):
        adapter = _build_adapter(update_mode="initial")
        x_t, cond = _build_inputs()
        calls = {"count": 0}
        original = adapter.build_hyper_input

        def counted_build_hyper_input(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        adapter.build_hyper_input = counted_build_hyper_input

        adapter(x_t, torch.full((2,), 999, dtype=torch.long), cond)
        adapter(x_t, torch.full((2,), 980, dtype=torch.long), cond)
        self.assertEqual(calls["count"], 1)

        adapter(x_t, torch.full((2,), 999, dtype=torch.long), cond)
        self.assertEqual(calls["count"], 2)

    def test_hyperalign_piecewise_mode_refreshes_only_when_stage_changes(self):
        adapter = _build_adapter(update_mode="piecewise")
        x_t, cond = _build_inputs()
        calls = {"count": 0}
        original = adapter.build_hyper_input

        def counted_build_hyper_input(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        adapter.build_hyper_input = counted_build_hyper_input

        adapter(x_t, torch.full((2,), 999, dtype=torch.long), cond)
        adapter(x_t, torch.full((2,), 950, dtype=torch.long), cond)
        self.assertEqual(calls["count"], 1)

        adapter(x_t, torch.full((2,), 700, dtype=torch.long), cond)
        self.assertEqual(calls["count"], 2)

    def test_hyperalign_accepts_dynamicrafter_style_generic_embeddings(self):
        adapter = _build_adapter(use_step_level_conditioning=True)
        x_t, cond = _build_inputs()
        cond["embedding"] = torch.randn(2, 3, 12)
        cond["step_level"] = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

        factor_tokens = adapter.build_hyper_input(x_t=x_t, t=torch.full((2,), 999, dtype=torch.long), cond=cond)
        self.assertEqual(tuple(factor_tokens.shape), (2, len(adapter._handles), 128))

    def test_hyperalign_respects_custom_target_modules(self):
        adapter = HyperAlignAdapter(
            rank=4,
            alpha=1.0,
            target_modules=["to_q", "to_out.0"],
            cond_dim=12,
            hidden_dim=128,
            input_summary_dim=4,
            aux_down_dim=16,
            aux_up_dim=16,
            num_decoder_layers=2,
            num_decoder_heads=8,
            update_mode="stepwise",
            piecewise_progress_markers=(0.0, 0.2),
        )
        adapter.attach_base_model(FakeBaseModel())
        names = [handle.qualified_name for handle in adapter._handles]
        self.assertTrue(any(name.endswith("to_q") for name in names))
        self.assertTrue(any(name.endswith("to_out.0") for name in names))
        self.assertFalse(any(name.endswith("to_k") for name in names))
        self.assertFalse(any(name.endswith("to_v") for name in names))


if __name__ == "__main__":
    unittest.main()
