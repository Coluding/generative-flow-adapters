# Metrics glossary — what each logged number means

All logged to wandb by the shared trainer (`src/generative_flow_adapters/training/trainer.py`).
`train/*` = per optimizer step; `eval_*` = per eval cycle (`eval_every_n_steps`).

## Action following (D2) — is the adapter using the action?
| metric | meaning | healthy | failure |
|---|---|---|---|
| `eval_action_effect_rel` | ‖pred(true) − pred(shuffled action)‖/‖pred‖ | > ~0.01 (moves) | **≈0 = action-blind** |
| `eval_action_cos` | cosine(true, shuffled) | < 1 | ≈1 = blind |
| `eval_action_loss_gap` | loss change when action perturbed | > 0 (action helps) | ≈0 |
| `eval_action_effect_vs_adapter` | effect ÷ adapter_rel_contribution | high (movement is action-driven) | ~0 (adapter moves, but not for actions) |
| `eval_action_base_null_violation` | frozen base's effect (must be action-invariant) | **0** (trust) | >0 = harness leak, distrust |
| `condition_grad_norm` | grad on the condition encoder (aggregated/adaLN action route; **DC's** route) | > 0 | ≈0 = action path starved |
| `action_inject_grad_norm` | grad on the cross-attn action tokens (**Wan/SkyReels** route; absent on DC) | > 0 | ≈0 = cross-attn action starved |

Config: `training.extra.action_sensitivity_probe` (default on; off on shortcut configs).
Reference: ACWM runs 0.001–0.006 (blind); AVID-RT1 0.0495 (follows).

## Shortcut / few-step (D3) — is the adapter step-conditioned, or collapsed?
| metric | meaning | healthy | failure |
|---|---|---|---|
| `shortcut_direction_loss` / `multistep_consistency_loss` | the shortcut + self-consistency objective | > 0 and falling | =0 (inert `anchor_prob 1.0`) |
| `train/shortcut_direction_loss/N001..N050` | per-step-size direction loss | falling; ordered (coarse N harder) | rising/diverging |
| `eval_stepsize_effect_rel` | ‖pred(d) − pred(d_ref)‖/‖pred(d_ref)‖ over step-sizes | > 0 (step-conditioned) | **≈0 = step-size-blind (collapse)** |
| `eval_stepsize_base_null_violation` | base's step-size effect (must be invariant) | **0** | >0 = leak |
| `eval_step_grid/*` (videos) | N∈{1,2,4,8,25,50} × gt|base|adapted | adapted@small-N ≈ base@50 | adapted@small-N degrades |
| `eval/adapted/{fid,fvd_i3d,psnr,ssim,lpips}` vs `eval/base/*` | quality vs base | adapted ≥ base at few steps | worse |

Config: `stepsize_sensitivity_probe` (default on when shortcut active); `eval_step_schedule`.
Note: flow uses `v_average`, diffusion uses `endpoint_inversion` consistency targets.

## Composition / copy diagnostics (both lines)
| metric | meaning | reading |
|---|---|---|
| `adapter_pred_base_cosine` | raw adapter branch vs frozen base | →1 = cloning the base (D2 failure). For shortcut, high at the anchor step is benign — cross-check with `eval_stepsize_effect_rel` |
| `adapter_base_cosine` | composed output vs base | high when the gate is base-heavy |
| `adapter_rel_contribution` | ‖pred−base‖/‖base‖ | how far the adapter moves the output off base (0 = clone) |
| `adapter_gate_mean` / `_std` | mask-mix gate σ(gate) | saturated (→cap, std→0) = adapter starved; stuck at 0.5 = gate not learning |
| `denoise_base_only` / `denoise_adapter_delta` | base loss / (base − adapted) | delta > 0 = adapter helps; **for shortcut, negative single-step delta is EXPECTED** (trades single-step for few-step) |
| `adapter_grad_norm` | grad on the whole adapter | healthy = learning; ~0 = starved |

## Rules
- A metric is only trustworthy when its `*_null_violation` companion ≈ 0.
- Low training loss ≠ success — a degenerate adapter can satisfy the loss while
  being action- or step-blind. Always cross-check the effect/probe metric.
- Cite every number with wandb run id + step (thesis hard rules 7–8).
