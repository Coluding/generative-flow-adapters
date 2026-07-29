# Experiment runbooks — what to run, for what, and the metric to watch

Operator's map of the thesis experiments. Each runbook says: the **question**, the
**exact commands**, and the **metric that answers it** (with healthy-vs-failure
values). Metric definitions live in [`metrics.md`](metrics.md).

## One-time cluster setup
- `git pull` on the cluster gets everything under `jobs/`, `configs/`, `src/`.
- **`external_repos/` is gitignored** → the AVID repo (its probe/converter/configs)
  reaches the cluster only via `rsync`. See [`avid-controls.md`](avid-controls.md).
- Base ckpt `ckpts/Wan2.2-TI2V-5B` + DC ckpt `ckts/dynami512.ckpt` must be present.

## The experiments

| Runbook | Question | Headline metric | Status |
|---|---|---|---|
| [`d2-action-following.md`](d2-action-following.md) | Do our adapters follow actions? | `eval_action_effect_rel` (blind ≈0.003) | ACWM = **blind**; RT-1 test staged |
| [`d3-shortcut.md`](d3-shortcut.md) | Does few-step shortcut work on flow vs diffusion? | `eval_stepsize_effect_rel` + few-step N-sweep quality + consistency loss | Wan learning; DC consistency ~65× higher |
| [`avid-controls.md`](avid-controls.md) | Does the *original* recipe follow actions? (control) | `action_effect_rel` on AVID ckpts | RT-1 **follows** (0.0495); ACWM-arm pending |
| [`metrics.md`](metrics.md) | What each logged metric means | — | reference |

## The through-line
D2 (action) went **blind on OOD synthetic ACWM** across all bases, but the AVID
control **follows actions on in-distribution RT-1** → the blindness is a **data**
problem. So the pivotal open runs are: **our adapter on RT-1** (does it flip
positive?) and the **AVID-on-ACWM-robot-arm** control (does the working recipe
also go blind on our data?). D3 (shortcut) is the independent, D2-free line where
flow is the headline.
