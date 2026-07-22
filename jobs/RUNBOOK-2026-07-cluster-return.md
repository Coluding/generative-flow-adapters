# Runbook — cluster return (prepared 2026-07-22 during the outage)

Everything below is staged in this repo. Thesis-vault context:
`20_Tickets/experiments/exp-adapter-{replace,gatelow}-nobase-overfit.md`,
`exp-adapter-gatelow-cap-sigmashift-metaworld-run.md`,
`50_Decisions/open/second-dataset-action-informativeness.md`.

## 0. One-time (login node)

```bash
cd ~/generative-flow-adapters
git status          # check for drifted local edits (e.g. wan22_replace_no_shortcut.yaml) — stash/commit them first
git pull
bash jobs/download_acwmphys.sh          # ~120 MB, minutes; verifies all 3 splits
```

What the pull brings: eval `action_seq` fix + fail-loud adapter guard (the
generation-noise root cause), `gate_cap`, `sigma_shift`, `--dataset
acwm_phys` / `--data-dir` / `--eval-data-dir` / `--temporal-length` train
flags, `ACWMPhysTranslator`, σ-sweep/action-probe tooling, 5 new configs, 4
job files.

NOTE — `trainer.py` still floors grad accumulation at `max(2, ...)`
(effective batch = 2×batch-size, half the optimizer steps). Left unchanged
DELIBERATELY: uxrst2k5/y1jrgxqp ran with it, and the triangle must be
comparable to them. Revisit only as its own controlled change.

## 1. Submit (in this order)

```bash
sbatch jobs/submit_overfit_triangle_capshift.sh   # MetaWorld: 3 overfit arms (1000 steps each) + full-data cap+shift until walltime
sbatch jobs/submit_precompute_acwmphys.sh         # ACWM Push Cube latents, all 3 splits -> shared cache (~13.6k windows)
# after the precompute job finishes:
sbatch jobs/submit_train_acwm_pushblock.sh        # first ACWM training run (gatelow + cap 0.9 + shift 5.0)
```

The two first jobs are independent — they can run concurrently if two GPUs
are free.

## 2. Readouts (what decides what)

| Run | Watch | Branch |
|---|---|---|
| replace-nobase overfit | denoise loss vs base ~0.05-0.08 floor | well below base ⇒ base-input was the trap; at base ⇒ 34M capacity limit |
| gatelow-nobase raw | `adapter_gate_mean`, `adapter_grad_norm` from step 1 | gate pins ~0.99 + gradnorm dies ⇒ gate trap alone suffices (arm can't speak to input question) |
| gatelow-nobase cap09 | same + does pred improve with guaranteed ≥10% gradient | pred improves & gate mixed ⇒ composition works without the crutch |
| full-data cap+shift (MetaWorld) | action probe on its checkpoint, NOT the loss delta | shuffle-gap > 0 ⇒ MetaWorld back in the claim-(a) race; ≈0 ⇒ dataset diagnosis locked |
| ACWM push_block train | denoise delta AND action probe on checkpoint | shuffle-gap > 0 here = first action-following signal of the project |

Action probe (local 3090 or cluster, on any checkpoint):

```bash
python scripts/generate_wan22_i2v_compare.py \
  --config <that run's config> --checkpoint <ckpt.pt> \
  --sigma-sweep --action-probe --loss-batches 0 \
  --temporal-length 41   # local only: hit the 41-frame cache; on cluster use the trained length
```

## 3. Optional local runs during/before (3090)

The overfit arms also run locally at 41-frame windows (latent-cache hits;
generation eval off — the 768² decode OOMs 24 GB):

```bash
python scripts/train_wan22_i2v_metaworld_external.py \
  --config configs/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml \
  --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
  --overfit-index 0 --num-windows 1 --temporal-length 41 \
  --steps 800 --batch-size 2 --no-eval-gen \
  --wandb-run-name local-overfit-replace-nobase-41f
# same pattern with ..._gatelow_nobase_overfit_... and ..._gatelow_nobase_gatecap_overfit_...
```

Caveat: 41f/batch-2/single-task-corner2 vs the cluster's 97f/batch-12/
five-task — trends (gate saturation, delta sign) transfer; absolute numbers
don't.
