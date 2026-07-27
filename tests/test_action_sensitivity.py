"""Tests for the action-sensitivity probe.

The probe's failure mode is *silent*: an unpaired comparison, a leaking base, or
a degenerate perturbation all produce plausible-looking numbers. These tests pin
the two verdicts (blind vs sensitive) and the guards that keep them honest.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from generative_flow_adapters.evaluation.action_sensitivity import (
    perturb_cond,
    run_action_sensitivity,
    result_to_dict,
)


class _FakeTrainer:
    """Minimal stand-in for ``Trainer._forward_and_loss``.

    Builds ``x_t`` from the global RNG exactly as the real trainer does, so the
    probe's RNG-forking discipline is exercised for real: if the probe forgot to
    reseed per variant, ``x_t`` would differ and the paired-comparison guard
    would fire.
    """

    def __init__(self, model, *, redraw_noise: bool = False):
        self.model = model
        self.redraw_noise = redraw_noise
        self.calls = 0

    def _forward_and_loss(self, batch):
        target = batch["target"]
        self.calls += 1
        noise = torch.randn(target.shape)
        if self.redraw_noise:
            # Simulates a backbone whose noise is NOT reproducible under a fixed
            # seed — the exact bug the pairing guard exists to catch. Uses a call
            # counter, not the RNG, so reseeding cannot mask it.
            noise = noise + 1e-3 * self.calls
        x_t = target + 0.1 * noise
        t = torch.zeros(target.shape[0])
        prediction, base = self.model(x_t, t, batch.get("cond"), return_base=True)
        loss = torch.nn.functional.mse_loss(prediction, target)
        base_loss = torch.nn.functional.mse_loss(base, target)
        components = {
            "base_loss": float(loss),
            "denoise_base_only": float(base_loss),
            "adapter_rel_contribution": float(
                (prediction - base).norm() / base.norm().clamp_min(1e-8)
            ),
        }
        return loss, components, x_t, t, batch.get("cond"), prediction, dict(batch)


class _FakeModel(torch.nn.Module):
    """Adapter whose action-dependence is dialled by ``action_weight``."""

    supports_return_base = True

    def __init__(self, action_weight: float):
        super().__init__()
        self.action_weight = action_weight
        self.condition_drop_prob = 0.0

    def forward(self, x_t: Tensor, t, cond, return_base: bool = False):
        base = 0.5 * x_t
        adapter = 0.1 * x_t
        if self.action_weight and isinstance(cond, dict) and isinstance(cond.get("action"), Tensor):
            action = cond["action"].float().mean(dim=tuple(range(1, cond["action"].dim())))
            adapter = adapter + self.action_weight * action.view(-1, *([1] * (x_t.dim() - 1)))
        out = base + adapter
        return (out, base) if return_base else out


def _batch(seed: int, *, batch_size: int = 2) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "target": torch.randn(batch_size, 3, 4, 4, generator=gen),
        "cond": {"action": torch.randn(batch_size, 5, generator=gen)},
    }


def _run(model, *, redraw_noise: bool = False, num_draws: int = 3):
    return run_action_sensitivity(
        trainer=_FakeTrainer(model, redraw_noise=redraw_noise),
        model=model,
        batches=[_batch(0), _batch(1), _batch(2)],
        variants=("shuffle", "zero"),
        num_draws=num_draws,
        progress=lambda _msg: None,
    )


def test_action_blind_model_reports_zero_effect():
    result = _run(_FakeModel(action_weight=0.0))
    for name in ("shuffle", "zero"):
        effects = result.variants[name].action_effect_rel
        assert effects, f"no samples recorded for {name}"
        assert max(effects) == pytest.approx(0.0, abs=1e-6), (
            f"{name}: an action-independent model must be exactly invariant"
        )


def test_action_sensitive_model_reports_nonzero_effect():
    result = _run(_FakeModel(action_weight=1.0))
    shuffle = result.variants["shuffle"].action_effect_rel
    assert min(shuffle) > 1e-3, "a strongly action-dependent model must move the prediction"


def test_frozen_base_null_control_holds():
    """The frozen base ignores actions, so its loss must not move across variants."""
    result = _run(_FakeModel(action_weight=1.0))
    assert result.base_null_violation == pytest.approx(0.0, abs=1e-9)


def test_unpaired_noise_is_an_error_not_a_silent_number():
    """A non-reproducible noise draw must fail loudly — an unpaired comparison
    reports noise variance as action sensitivity."""
    with pytest.raises(RuntimeError, match="unpaired comparison"):
        _run(_FakeModel(action_weight=0.0), redraw_noise=True)


def test_condition_drop_prob_is_restored():
    model = _FakeModel(action_weight=1.0)
    model.condition_drop_prob = 0.3
    _run(model)
    assert model.condition_drop_prob == pytest.approx(0.3), (
        "the probe must restore training-time condition dropout it disabled"
    )


@pytest.mark.parametrize("variant", ["zero", "roll", "gauss", "shuffle"])
def test_perturbations_change_the_action(variant):
    cond = {"action": torch.arange(12, dtype=torch.float32).view(2, 6)}
    donor = {"action": torch.full((2, 6), 99.0)}
    out = perturb_cond(cond, variant, donor=donor, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(out["action"], cond["action"]), f"{variant} left the action unchanged"
    assert cond["action"].shape == out["action"].shape


def test_perturbation_covers_every_action_key():
    """Missing one key lets the model fall back to the surviving one, which
    silently under-reports sensitivity."""
    cond = {
        "action": torch.ones(2, 4),
        "action_seq": torch.ones(2, 3, 4),
        "context": torch.ones(2, 7),
    }
    out = perturb_cond(cond, "zero", donor=None)
    assert out["action"].abs().sum() == 0
    assert out["action_seq"].abs().sum() == 0
    assert torch.equal(out["context"], cond["context"]), "non-action conditioning must be untouched"


def test_missing_action_key_is_an_error_not_an_action_blind_verdict():
    """A preprocessor emitting actions under an unexpected key must fail, not
    silently produce 0.0 and a confident ACTION-BLIND verdict."""
    model = _FakeModel(action_weight=1.0)
    batch = _batch(0)
    batch["cond"] = {"act_tokens": batch["cond"]["action"]}  # unrecognised name
    with pytest.raises(RuntimeError, match="no action tensor found"):
        run_action_sensitivity(
            trainer=_FakeTrainer(model),
            model=model,
            batches=[batch, _batch(1)],
            variants=("shuffle",),
            num_draws=1,
            progress=lambda _msg: None,
        )


def test_custom_action_keys_are_perturbed():
    """A backbone emitting actions under a non-default name works once named."""
    model = _FakeModel(action_weight=0.0)  # weight irrelevant; we check the key plumbing
    batch_a, batch_b = _batch(0), _batch(1)
    for batch in (batch_a, batch_b):
        batch["cond"] = {"act_tokens": batch["cond"]["action"]}
    result = run_action_sensitivity(
        trainer=_FakeTrainer(model), model=model, batches=[batch_a, batch_b],
        variants=("zero",), num_draws=1, action_keys=("act_tokens",),
        progress=lambda _msg: None,
    )
    assert any("act_tokens" in note for note in result.notes)


def test_explicitly_named_missing_key_aborts():
    """A typo in --action-keys must abort, not silently narrow the perturbation."""
    model = _FakeModel(action_weight=1.0)
    with pytest.raises(RuntimeError, match="are not in the batch"):
        run_action_sensitivity(
            trainer=_FakeTrainer(model), model=model, batches=[_batch(0), _batch(1)],
            variants=("zero",), num_draws=1,
            action_keys=("action", "action_seq"),  # action_seq absent from the fake batch
            require_all_keys=True,
            progress=lambda _msg: None,
        )


def test_inconsistent_keys_across_batches_abort():
    """A batch missing the action would be silently unperturbed and drag the
    mean toward 'action-blind'."""
    model = _FakeModel(action_weight=1.0)
    good, odd = _batch(0), _batch(1)
    odd["cond"] = {"action": odd["cond"]["action"], "action_seq": torch.ones(2, 3, 5)}
    with pytest.raises(RuntimeError, match="exposes action keys"):
        run_action_sensitivity(
            trainer=_FakeTrainer(model), model=model, batches=[good, odd],
            variants=("zero",), num_draws=1, progress=lambda _msg: None,
        )


def test_summary_is_json_serialisable():
    import json

    summary = result_to_dict(_run(_FakeModel(action_weight=0.5)))
    json.dumps(summary)
    assert "shuffle" in summary["variants"]
    assert len(summary["variants"]["shuffle"]["action_effect_rel_ci95"]) == 2
