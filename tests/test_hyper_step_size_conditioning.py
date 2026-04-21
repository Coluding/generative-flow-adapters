from __future__ import annotations

import unittest

import torch

from generative_flow_adapters.adapters.hypernetworks.stepwise_lora import SimpleHyperLoRAAdapter


class HyperStepSizeConditioningTest(unittest.TestCase):
    def test_step_size_conditioning_builds_hyper_input(self):
        adapter = SimpleHyperLoRAAdapter(
            rank=4,
            alpha=1.0,
            target_modules=["net.0", "net.2"],
            cond_dim=64,
            hidden_dim=128,
            input_summary_dim=32,
            include_step_size=True,
            step_size_key="step_size",
        )
        x_t = torch.randn(2, 32)
        t = torch.tensor([10, 20], dtype=torch.long)
        cond = {
            "embedding": torch.randn(2, 64),
            "step_size": torch.tensor([1.0, 2.0]),
        }
        hyper_input = adapter.build_hyper_input(x_t=x_t, t=t, cond=cond)
        self.assertEqual(tuple(hyper_input.shape), (2, 128))

    def test_step_size_conditioning_requires_step_size_tensor(self):
        adapter = SimpleHyperLoRAAdapter(
            rank=4,
            alpha=1.0,
            target_modules=["net.0", "net.2"],
            cond_dim=64,
            hidden_dim=128,
            input_summary_dim=32,
            include_step_size=True,
            step_size_key="step_size",
        )
        x_t = torch.randn(2, 32)
        t = torch.tensor([10, 20], dtype=torch.long)
        cond = {"embedding": torch.randn(2, 64)}
        with self.assertRaises(KeyError):
            _ = adapter.build_hyper_input(x_t=x_t, t=t, cond=cond)


if __name__ == "__main__":
    unittest.main()
