# D2 — do the adapters follow actions?

**Question:** does the action-conditioned adapter actually use the action, or
collapse to an action-independent base-clone (base-parity)?

**Headline metric:** `eval_action_effect_rel` — ‖pred(true action) − pred(shuffled
action)‖ / ‖pred‖, logged every eval cycle. **≈0 ⇒ action-blind.** Trust it only
when `eval_action_base_null_violation ≈ 0` (frozen base must be action-invariant).
Supporting: `condition_grad_norm` / `action_inject_grad_norm` (does the action
pathway get gradient), `adapter_pred_base_cosine` (→1 = cloning). See
[`metrics.md`](metrics.md).

---

## Exp A — ACWM base × dataset matrix  (RESULT: all action-blind)

Wan/DC/SkyReels × Push Cube / Robot Arm. Prereqs (login node, per dataset):
`infra/download_acwmphys*.sh` → `infra/submit_precompute_acwmphys*.sh` →
`infra/submit_precompute_prompts_acwm_*.sh` (T5 contexts, required at startup).

```bash
sbatch jobs/experiments_cluster/acwm_phys/wan/submit_train_wan_robotarm.sh
sbatch jobs/experiments_cluster/acwm_phys/dc/submit_train_dc_robotarm.sh
sbatch jobs/experiments_cluster/acwm_phys/skyreels/submit_train_skyreels_robotarm.sh
# push_block variants + the intervention run alongside
```
Measured: Wan 0.0056 · DC 0.0034 · SkyReels 0.0013 (all blind; null 0).
Tickets: `20_Tickets/experiments/exp-backbone-*`. Note: DC logs `condition_grad_norm`
(its action route), Wan/SkyReels log `action_inject_grad_norm` (cross-attn route).

## Exp B — intervention (gate_cap 0.5 + AVID warmup), Wan · Push Cube

Can a harder gate cap + warmup rescue the adapter on the near-zero-residual flat
domain? Reuses the pushblock latents (no new precompute).
```bash
sbatch jobs/experiments_cluster/acwm_phys/wan/submit_train_wan_pushblock_cap50_warmup.sh
```
Watch: `adapter_pred_base_cosine` (still ~0.85 ⇒ still cloning) + `eval_action_effect_rel`.

## Exp C — RT-1 in-distribution action test  (THE D2-flip experiment)

Does our adapter follow actions on **real** robot video, where it went blind on
OOD synthetic ACWM? Full pipeline + interpretation:
**→ [`jobs/experiments_cluster/rt1/README.md`](../experiments_cluster/rt1/README.md)**
```bash
bash   jobs/experiments_cluster/rt1/convert_rt1.sh
sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents.sh
sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_captions.sh
sbatch jobs/experiments_cluster/rt1/submit_train_wan_rt1_action.sh
```
**Success = `eval_action_effect_rel` comes off the ~0.003 blind floor** (compare
the AVID-RT1 control's 0.0495). If it does, D2 is positive. Actions are per-dim
std-normalized (converter default) so a weak result isn't a scale artifact.
