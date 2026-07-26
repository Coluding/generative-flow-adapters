"""Action-sensitivity probe — does the adapter's prediction depend on the action?

The measurement behind requirement **R1** of the thesis storyline: given a
trained adapter checkpoint, perturb the action conditioning and ask whether the
prediction moves at all. A model whose prediction is invariant to the action is
**action-blind** — it may still converge beautifully and show a healthy gate, but
it is useless as a world model and cannot support planning (every candidate
action sequence yields the same rollout).

Backbone-agnostic: this module knows nothing about DynamiCrafter / Wan / SkyReels.
It needs a :class:`Trainer` (for the backbone's ``x_t``/``t`` construction), a
composed :class:`AdaptedModel`, and preprocessed batches.

Why the metrics are shaped the way they are
-------------------------------------------
*Prediction-space, not just loss.* The direct question is whether the prediction
moves, and ``‖pred_true − pred_perturbed‖`` answers it without reference to a
target. A loss *gap* additionally tells you whether the movement is in a
*useful* direction, so both are reported — but the prediction delta is primary
because a model can have a ~0 loss gap while still being action-sensitive (it
moves, just not helpfully), and that distinction matters diagnostically.

*Paired draws.* All variants for a given batch/draw share the same ``x_t`` and
``t``. This is enforced, not assumed — :func:`_assert_paired` fails loudly if the
backbone's noise draw leaked across variants, because an unpaired comparison
silently turns noise variance into "action sensitivity".

*A built-in null control.* The frozen base ignores the action by construction, so
its loss must be **identical** across variants. If it is not, the harness is
wrong (typically ``pass_cond_to_base`` leaking actions into the base, or
condition dropout still active) — reported as ``base_null_violation`` rather
than silently inflating the result.

*Relative to what the adapter does at all.* ``action_effect_rel`` is normalised
by ‖pred‖, and ``adapter_rel_contribution`` says how far the adapter moved the
output from the frozen base. An adapter that barely moves the output at all has
a trivially small action effect, and reading that as "action-blind" would be a
misdiagnosis — the ratio of the two is what separates "does nothing" from "does
something, but nothing action-driven".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

# Keys the preprocessors emit for action conditioning. ``action_seq`` is the
# per-frame token sequence used by the cross-attention adapters; ``action`` is
# the aggregated vector. Both must be perturbed together or the model falls back
# to whichever survived and the probe under-reports.
ACTION_KEYS: tuple[str, ...] = ("action", "action_seq")

VARIANTS: tuple[str, ...] = ("shuffle", "zero", "roll", "gauss")


@dataclass
class VariantStats:
    """Per-variant accumulation across (batch, draw) pairs."""

    name: str
    action_effect_rel: list[float] = field(default_factory=list)
    cos_true_variant: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    loss_gap: list[float] = field(default_factory=list)
    base_loss_delta: list[float] = field(default_factory=list)


@dataclass
class ActionSensitivityResult:
    variants: dict[str, VariantStats]
    true_loss: list[float]
    adapter_rel_contribution: list[float]
    gate_mean: list[float]
    num_samples: int
    base_null_violation: float
    notes: list[str]


def _masked(tensor: Tensor, batch: Mapping[str, object]) -> Tensor:
    """Flatten ``tensor`` to the predicted frames only.

    Diffusion-forcing batches mark observation frames with ``frame_mask == 0``;
    those frames are clamped clean and carry no prediction, so including them
    dilutes every norm below toward zero and makes an action-blind model look
    merely "weakly sensitive".
    """
    frame_mask = batch.get("frame_mask")
    if not isinstance(frame_mask, Tensor):
        return tensor.float().flatten()
    m = frame_mask.to(device=tensor.device, dtype=torch.bool)
    m = m.view(m.shape[0], 1, m.shape[1], 1, 1).expand_as(tensor)
    return tensor.float()[m]


def _rel_delta(a: Tensor, b: Tensor, batch: Mapping[str, object]) -> float:
    """‖a − b‖ / ‖a‖ over predicted frames. 0 ⇒ the perturbation changed nothing."""
    av, bv = _masked(a, batch), _masked(b, batch)
    return float(((av - bv).norm() / av.norm().clamp_min(1e-8)).cpu())


def _cosine(a: Tensor, b: Tensor, batch: Mapping[str, object]) -> float:
    av, bv = _masked(a, batch), _masked(b, batch)
    return float(torch.nn.functional.cosine_similarity(av, bv, dim=0).cpu())


def perturb_cond(
    cond: object,
    variant: str,
    *,
    donor: object | None,
    generator: torch.Generator | None = None,
) -> object:
    """Return ``cond`` with its action entries replaced according to ``variant``.

    - ``shuffle`` — actions from a *different clip* (``donor``). The cleanest
      test: the actions stay on-distribution, only their correspondence to this
      clip's dynamics is broken. Falls back to ``zero`` when no donor is
      available, which is reported rather than silently substituted.
    - ``zero`` — null actions. Cheap, but a model trained with condition dropout
      has *seen* zeros and may treat them as a valid "no action" token, so a
      small zero-gap is weaker evidence than a small shuffle-gap.
    - ``roll`` — this clip's own actions, rolled by half the sequence. Identical
      marginal statistics, wrong temporal alignment: separates "uses action
      magnitude" from "uses action *timing*".
    - ``gauss`` — Gaussian noise matched to the batch's per-key mean/std.
    """
    if not isinstance(cond, Mapping):
        return cond
    out = dict(cond)
    for key in ACTION_KEYS:
        value = out.get(key)
        if not isinstance(value, Tensor):
            continue
        if variant == "zero":
            out[key] = torch.zeros_like(value)
        elif variant == "roll":
            # Roll along the time axis when there is one, else along the batch.
            axis = 1 if value.dim() >= 2 and value.shape[1] > 1 else 0
            out[key] = torch.roll(value, shifts=max(1, value.shape[axis] // 2), dims=axis)
        elif variant == "gauss":
            noise = torch.randn(
                value.shape, generator=generator, device=value.device, dtype=value.dtype
            )
            out[key] = value.mean() + value.std().clamp_min(1e-6) * noise
        elif variant == "shuffle":
            donor_value = donor.get(key) if isinstance(donor, Mapping) else None
            if isinstance(donor_value, Tensor) and donor_value.shape == value.shape:
                out[key] = donor_value.to(device=value.device, dtype=value.dtype)
            else:
                # Within-batch roll is the honest fallback: still a *different
                # clip's* actions when batch>1. For batch==1 it degenerates to
                # the identity, which the caller detects via a ~0 effect on ALL
                # variants and the "shuffle degenerate" note.
                out[key] = torch.roll(value, shifts=1, dims=0)
        else:
            raise ValueError(f"unknown action perturbation variant: {variant!r}")
    return out


def _assert_actions_present(cond: object) -> tuple[str, ...]:
    """Fail if the batch carries no action tensor under a key we perturb.

    This is the probe's most dangerous silent failure: a preprocessor that emits
    actions under an unexpected key means :func:`perturb_cond` changes nothing,
    every variant equals the reference, and the report confidently declares the
    model ACTION-BLIND. The measurement would be of *nothing*. So a missing key
    is an error, never a zero.
    """
    if not isinstance(cond, Mapping):
        raise RuntimeError(
            f"batch['cond'] is {type(cond).__name__}, not a mapping — the probe cannot "
            "locate or perturb the action conditioning."
        )
    found = tuple(k for k in ACTION_KEYS if isinstance(cond.get(k), Tensor))
    if not found:
        raise RuntimeError(
            f"no action tensor found in cond under any of {ACTION_KEYS}. "
            f"Available keys: {sorted(str(k) for k in cond)}. "
            "This preprocessor emits actions under a different name — add it to "
            "ACTION_KEYS, or the probe would report a meaningless 0.0 and the "
            "report would call the model action-blind."
        )
    return found


def _assert_paired(reference: Tensor, other: Tensor, variant: str) -> None:
    """Guard against an unpaired comparison.

    If the backbone re-draws noise per forward, ``x_t`` differs across variants
    and the measured "action effect" is really noise variance. That failure mode
    produces a *plausible non-zero* number, so it must be an error, not a warning.
    """
    if reference.shape != other.shape or not torch.equal(reference, other):
        raise RuntimeError(
            f"unpaired comparison: x_t differs between the true and '{variant}' "
            "forward. All variants must share one noise draw — the probe is "
            "measuring noise variance, not action sensitivity."
        )


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0, iters: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI. Returns (lo, hi); (nan, nan) for <2 samples."""
    n = len(values)
    if n < 2:
        return (math.nan, math.nan)
    tensor = torch.tensor(values, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (iters, n), generator=gen)
    means = tensor[idx].mean(dim=1)
    lo = float(means.quantile(0.025))
    hi = float(means.quantile(0.975))
    return (lo, hi)


def run_action_sensitivity(
    *,
    trainer,
    model,
    batches: Iterable[Mapping[str, object]],
    variants: Sequence[str] = VARIANTS,
    num_draws: int = 4,
    seed: int = 0,
    progress: Callable[[str], None] = print,
) -> ActionSensitivityResult:
    """Measure whether perturbing the action changes the adapter's prediction.

    ``batches`` must be *preprocessed* batches (post ``_call_preprocessor``).
    They are materialised into a list so the shuffle variant can borrow a donor
    clip's actions.

    ``num_draws`` repeats each batch with fresh noise; the spread across draws
    and batches is what the bootstrap CI is computed over.
    """
    pool = list(batches)
    if not pool:
        raise ValueError("no batches supplied to the action-sensitivity probe.")

    notes: list[str] = []
    action_keys = _assert_actions_present(pool[0].get("cond"))
    notes.append(f"perturbing action keys: {', '.join(action_keys)}")
    stats = {name: VariantStats(name=name) for name in variants}
    result = ActionSensitivityResult(
        variants=stats,
        true_loss=[],
        adapter_rel_contribution=[],
        gate_mean=[],
        num_samples=0,
        base_null_violation=0.0,
        notes=notes,
    )

    # Condition dropout is a *training* augmentation; leaving it on would blank
    # the action on a random subset of forwards and blur every variant together.
    previous_drop = getattr(model, "condition_drop_prob", None)
    if previous_drop:
        model.condition_drop_prob = 0.0
        notes.append(f"condition_drop_prob forced 0.0 for the probe (was {previous_drop}).")

    if not getattr(model, "supports_return_base", False):
        notes.append(
            "model does not support return_base — base-loss null control unavailable; "
            "a harness bug that leaks actions into the frozen base would go undetected."
        )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(pool):
            donor = pool[(batch_index + 1) % len(pool)].get("cond") if len(pool) > 1 else None
            if donor is None and "shuffle" in variants:
                notes.append(
                    "only one batch supplied: the 'shuffle' variant degenerates to a "
                    "within-batch roll (identity at batch size 1)."
                )
            for draw in range(num_draws):
                draw_seed = seed + 1000 * batch_index + draw
                base_losses: list[float] = []

                # --- true actions: the reference forward -----------------------
                with torch.random.fork_rng(devices=_rng_devices()), torch.no_grad():
                    torch.manual_seed(draw_seed)
                    (
                        loss_true,
                        comps_true,
                        x_t_ref,
                        _t,
                        _cond,
                        pred_true,
                        _b,
                    ) = trainer._forward_and_loss(batch)
                result.true_loss.append(float(loss_true.detach().cpu()))
                if "adapter_rel_contribution" in comps_true:
                    result.adapter_rel_contribution.append(comps_true["adapter_rel_contribution"])
                if "adapter_gate_mean" in comps_true:
                    result.gate_mean.append(comps_true["adapter_gate_mean"])
                if "denoise_base_only" in comps_true:
                    base_losses.append(comps_true["denoise_base_only"])

                # --- each perturbation, same noise draw -------------------------
                for name in variants:
                    perturbed = dict(batch)
                    perturbed["cond"] = perturb_cond(
                        batch.get("cond"), name, donor=donor, generator=gen
                    )
                    with torch.random.fork_rng(devices=_rng_devices()), torch.no_grad():
                        torch.manual_seed(draw_seed)  # identical noise/timesteps
                        (
                            loss_v,
                            comps_v,
                            x_t_v,
                            _t_v,
                            _c_v,
                            pred_v,
                            _b_v,
                        ) = trainer._forward_and_loss(perturbed)
                    _assert_paired(x_t_ref, x_t_v, name)

                    entry = stats[name]
                    entry.action_effect_rel.append(_rel_delta(pred_true, pred_v, batch))
                    entry.cos_true_variant.append(_cosine(pred_true, pred_v, batch))
                    entry.loss.append(float(loss_v.detach().cpu()))
                    entry.loss_gap.append(float(loss_v.detach().cpu()) - result.true_loss[-1])
                    if "denoise_base_only" in comps_v and base_losses:
                        # The frozen base ignores actions: this must be ~0.
                        entry.base_loss_delta.append(comps_v["denoise_base_only"] - base_losses[0])

                result.num_samples += 1
            progress(f"  batch {batch_index + 1}/{len(pool)} done ({num_draws} draws)")
    finally:
        if previous_drop:
            model.condition_drop_prob = previous_drop
        if was_training:
            model.train()

    violations = [
        abs(v) for entry in stats.values() for v in entry.base_loss_delta
    ]
    result.base_null_violation = max(violations) if violations else 0.0
    return result


def _rng_devices() -> list[int]:
    return [torch.cuda.current_device()] if torch.cuda.is_available() else []


def format_report(result: ActionSensitivityResult, *, threshold: float = 0.01) -> str:
    """Human-readable verdict.

    ``threshold`` is the relative prediction change below which the adapter is
    called action-blind. 1% is a deliberately *generous* bar — it is far below
    anything that could drive a planner, so failing it is unambiguous while
    passing it is not by itself evidence of useful action-following.
    """
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("ACTION SENSITIVITY — does perturbing the action change the prediction?")
    add("=" * 78)
    add(f"samples: {result.num_samples} (batch x draw)")

    if result.adapter_rel_contribution:
        mean_contrib = sum(result.adapter_rel_contribution) / len(result.adapter_rel_contribution)
        add(f"adapter_rel_contribution : {mean_contrib:.5f}   (|pred-base|/|base|; 0 = pure base clone)")
    if result.gate_mean:
        add(f"gate mean                : {sum(result.gate_mean) / len(result.gate_mean):.4f}")
    if result.true_loss:
        add(f"loss (true actions)      : {sum(result.true_loss) / len(result.true_loss):.6f}")

    add("")
    add(f"{'variant':>9} {'rel_delta':>11} {'95% CI':>21} {'cos':>8} {'loss_gap':>11} {'base_null':>10}")
    for name, entry in result.variants.items():
        if not entry.action_effect_rel:
            continue
        n = len(entry.action_effect_rel)
        mean_rel = sum(entry.action_effect_rel) / n
        lo, hi = _bootstrap_ci(entry.action_effect_rel)
        mean_cos = sum(entry.cos_true_variant) / n
        mean_gap = sum(entry.loss_gap) / n
        null = max((abs(v) for v in entry.base_loss_delta), default=0.0)
        add(
            f"{name:>9} {mean_rel:>11.6f} {f'[{lo:.6f}, {hi:.6f}]':>21} "
            f"{mean_cos:>8.5f} {mean_gap:>+11.6f} {null:>10.2e}"
        )

    add("")
    add("  rel_delta  |pred_true - pred_perturbed| / |pred_true|, predicted frames only.")
    add("             This is the primary readout: 0 = the action changed nothing.")
    add("  cos        cosine(pred_true, pred_perturbed). 1.0 = identical direction.")
    add("  loss_gap   perturbed loss - true loss. >0 = true actions genuinely help.")
    add("  base_null  |base loss shift| across variants. MUST be ~0 (frozen base")
    add("             ignores actions); non-zero means the harness leaks actions.")

    add("")
    if result.base_null_violation > 1e-6:
        add(f"!! HARNESS ERROR: base loss moved by {result.base_null_violation:.3e} across variants.")
        add("   The frozen base cannot see actions — check pass_cond_to_base and")
        add("   condition dropout. Do NOT report these numbers until this is 0.")
    else:
        add("null control OK: the frozen base's loss is invariant across variants.")

    shuffle = result.variants.get("shuffle")
    if shuffle and shuffle.action_effect_rel:
        mean_rel = sum(shuffle.action_effect_rel) / len(shuffle.action_effect_rel)
        lo, hi = _bootstrap_ci(shuffle.action_effect_rel)
        add("")
        if hi < threshold:
            add(f"VERDICT: ACTION-BLIND. Shuffling actions moves the prediction by "
                f"{mean_rel:.2%} (CI upper bound {hi:.2%} < {threshold:.0%}).")
            add("         This model cannot support planning: different action sequences")
            add("         produce effectively the same rollout.")
        elif lo > threshold:
            add(f"VERDICT: ACTION-SENSITIVE. Shuffling actions moves the prediction by "
                f"{mean_rel:.2%} (CI lower bound {lo:.2%} > {threshold:.0%}).")
            add("         Note: sensitivity is necessary but not sufficient — check the")
            add("         loss_gap sign to see whether the true actions are actually better.")
        else:
            add(f"VERDICT: INCONCLUSIVE. Shuffle effect {mean_rel:.2%}, CI [{lo:.2%}, {hi:.2%}] "
                f"straddles the {threshold:.0%} threshold.")
            add("         Increase --num-draws / --num-batches before drawing a conclusion.")

    for note in result.notes:
        add(f"note: {note}")
    add("=" * 78)
    return "\n".join(lines)


def result_to_dict(result: ActionSensitivityResult) -> dict:
    """JSON-serialisable summary for the vault / wandb."""
    out: dict = {
        "num_samples": result.num_samples,
        "base_null_violation": result.base_null_violation,
        "notes": result.notes,
        "true_loss_mean": (
            sum(result.true_loss) / len(result.true_loss) if result.true_loss else None
        ),
        "adapter_rel_contribution_mean": (
            sum(result.adapter_rel_contribution) / len(result.adapter_rel_contribution)
            if result.adapter_rel_contribution
            else None
        ),
        "gate_mean": (
            sum(result.gate_mean) / len(result.gate_mean) if result.gate_mean else None
        ),
        "variants": {},
    }
    for name, entry in result.variants.items():
        if not entry.action_effect_rel:
            continue
        n = len(entry.action_effect_rel)
        lo, hi = _bootstrap_ci(entry.action_effect_rel)
        out["variants"][name] = {
            "action_effect_rel_mean": sum(entry.action_effect_rel) / n,
            "action_effect_rel_ci95": [lo, hi],
            "cos_true_variant_mean": sum(entry.cos_true_variant) / n,
            "loss_mean": sum(entry.loss) / n,
            "loss_gap_mean": sum(entry.loss_gap) / n,
            "base_loss_delta_max": max((abs(v) for v in entry.base_loss_delta), default=0.0),
        }
    return out
