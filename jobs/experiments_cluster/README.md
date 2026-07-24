# experiments_cluster/ — Snellius job scripts, grouped by dataset and base

Cluster (sbatch) submission scripts for the experiment queue. Local 3090 probes
live in `jobs/experiments/` and are not mirrored here.

```
metaworld/
  avid/   DynamiCrafter-base runs (AVID-repo UNet adapter, HyperAlign)
  wan/    Wan2.x-base runs (AVID-style adapters on frozen Wan2.2 TI2V-5B)
acwm_phys/  ACWM-Phys push_block
infra/      precompute / download / smoke test
```

**Submit from the repo root** — `#SBATCH --output=logs/%x-%j.out` resolves
against the submit-time CWD:

```bash
sbatch jobs/experiments_cluster/metaworld/wan/<script>.sh
```

All training scripts accept env overrides (`BATCH_SIZE`, `DS_PATH`) and pass
trailing `"$@"` through to the python entrypoint.

## Warm the latent cache before parallel submission

`LatentCache.put()` stages through a fixed per-key `<key>.tmp` before its atomic
rename, so two jobs encoding the **same** key concurrently can publish a corrupt
latent. Jobs that share a dataset therefore need a warm cache first:

```bash
DS_PATH=../scratch-shared/metaworld/three_task.hdf5 \
  sbatch jobs/experiments_cluster/infra/precompute_cache.sh
```

The `metaworld/wan/` base-parity arms **refuse to start** on a cold cache;
`ALLOW_COLD_CACHE=1` overrides, and is only safe when that job runs alone.

## metaworld/wan/ — frozen Wan2.2 TI2V-5B base

Base-parity campaign (2026-07-21) — four independent arms, submit all at once:

| Script | Config (`configs/wan22/`) | Vault ticket (`20_Tickets/experiments/`) | Bound |
|---|---|---|---|
| `submit_overfit_replace_nobase.sh` | `…xattn_replace_nobase_overfit_metaworld` | `exp-adapter-replace-nobase-overfit` | 1000 steps / 12h |
| `submit_overfit_gatelow_nobase.sh` | `…xattn_gatelow_nobase_overfit_metaworld` | `exp-adapter-gatelow-nobase-overfit` | 1000 steps / 12h |
| `submit_overfit_gatelow_nobase_cap09.sh` | `…xattn_gatelow_nobase_gatecap_overfit_metaworld` | `exp-adapter-gatelow-nobase-overfit` (cap09 arm) | 1000 steps / 12h |
| `submit_train_gatelow_cap_sigmashift.sh` | `…xattn_gatelow_cap_sigmashift_metaworld` | `exp-adapter-gatelow-cap-sigmashift-metaworld-run` | full data / 32h |

These four replace the old sequential `submit_overfit_triangle_capshift.sh`
(removed; recoverable from git history). Each arm keeps that script's settings —
batch 12, `--overfit-index 0 --num-windows 1 --steps 1000` for arms 1-3, and
`--num-windows 8 --steps 5000000` for arm 4 — and each config writes its own
`output_dir`, so parallel arms don't collide. Readouts per arm are in section 2
of `jobs/RUNBOOK-2026-07-cluster-return.md`.

Other Wan-base runs:

| Script | Config | Vault ticket | Status |
|---|---|---|---|
| `submit_train_adaln_gatelow.sh` | `wan22/diffusion_wan22_avid_gatelow_metaworld.yaml` | `exp-adapter-adaln-gatelow-metaworld-run` | controlled retest, 3 confound fixes in config |
| `submit_train_gatelow_noshortcut.sh` | `wan22/diffusion_wan22_avid_gatelow_noshortcut_metaworld.yaml` | `exp-shortcut-zero-weight-control-run` | shortcut-OFF control for the above |
| `submit_train_dcunet_gatelow.sh` | `wan22/diffusion_wan22_dcunet_output_metaworld.yaml` | `exp-adapter-dcunet-gatelow-capacity-run` | capacity-vs-injection test |
| `submit_train_shortcut_only.sh` | `wan22/flow_wan22_shortcut_only_metaworld.yaml` | `exp-shortcut-action-free-isolation` | action-free pure-D3 test |
| `submit_train_wan_shortcut.sh` | `wan22/diffusion_wan22_avid_i2v_metaworld.yaml` | — | pre-gate-fix baseline (job-name says avid-shortcut; copy-paste artifact) |
| `submit_train_wan22_avid_noshortcut.sh` | `wan22/diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml` | superseded by `exp-shortcut-zero-weight-control-run` (stale config: wrong script / 121f / 256px) | legacy |

The gatelow-family scripts default to `BATCH_SIZE=2` because their configs carry
`grad_accum_steps: 4` (effective batch 8, matching the AVID reference run
`pg3x72uc`). The base-parity arms default to `BATCH_SIZE=12`, as the original
campaign script did. Wan2.2 configs whose header says so must run via
`scripts/train_wan22_i2v_metaworld_external.py` (real pretrained weights).

## metaworld/avid/ — DynamiCrafter base

| Script | Config (`configs/dynamicrafter/`) | Vault ticket | Status |
|---|---|---|---|
| `submit_train_avid_shortcut_affine.sh` | `diffusion_avid_shortcut_affine_metaworld.yaml` | `exp-adapter-output-format-affine-vs-direct` (affine arm) | equal budget to the direct arm |
| `submit_train_avid_shortcut_action.sh` | `diffusion_avid_shortcut_action_metaworld.yaml` | `exp-conditioning-add-actions-to-shortcut-adapter` | verify config-header checklist before launch |
| `submit_train_avid_shortcut.sh` | `diffusion_avid_shortcut_metaworld.yaml` | `exp-adapter-output-format-affine-vs-direct` (direct arm) | ⚠ entrypoint mismatch — see below |
| `submit_train_hyperalign.sh` | (script defaults) | — | legacy baseline (was `jobs/submit_train.sh`) |

⚠ **Pre-existing mismatch, not fixed here:** `submit_train_avid_shortcut.sh`
pairs the DynamiCrafter config with `scripts/train_wan22_i2v_metaworld.py`
(which only builds the wan2.2 provider). The matching entrypoint is
`scripts/train_avid_shortcut_metaworld.py` (its default config is exactly this
file). Fix before submitting the direct arm of the affine-vs-direct ablation.

`submit_train_hyperalign.sh` is HyperAlign rather than AVID, but shares the
DynamiCrafter base, so it lives here rather than under `wan/`.

## acwm_phys/

| Script | Config | Vault ticket | Status |
|---|---|---|---|
| `submit_train_acwm_pushblock.sh` | `wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml` | (second-dataset move — `50_Decisions/open/second-dataset-action-informativeness`) | submit after `infra/submit_precompute_acwmphys.sh` finishes |

## infra/

| Script | Purpose |
|---|---|
| `download_acwmphys.sh` | LOGIN-node helper: HF download of ACWM-Phys push_block into `ds/acwm-phys/` |
| `submit_precompute_acwmphys.sh` | VAE-latent precompute for the 3 ACWM splits → `latents.shared` cache |
| `precompute_cache.sh` | MetaWorld latent precompute (`gpu_a100`); `DS_PATH` selects the dataset |
| `submit_test_wan22_i2v.sh` | Wan2.2 TI2V-5B GPU smoke test (pytest MSE check) |

## Open tickets intentionally NOT scripted here

| Ticket | Why |
|---|---|
| `exp-adapter-param-matched-comparison` | blocked on `feat-adapter-flops-per-step-estimator`; ~39–48-run sweep needs its own harness |
| `exp-adapter-avid-native-reference-run` | separate Poetry/torch-2.1 env (external AVID repo); running locally as `pg3x72uc` — cluster recipe lives in the ticket |
| `exp-adapter-xattn-gatelow-metaworld-run` | already ran (`bcipghvw`, crashed @624 steps, adapter cloned base); rerun decision pending |
| `exp-shortcut-scale-episodes-longer-train` | config lives in `data/results/20261706/…`; fold in reweighting + endpoint-inversion first |
| `exp-shortcut-vs-image-only-anchor-baseline` | same config as its counterpart, only `shortcut_anchor_prob` differs; gated on the per-stepsize diagnosis |
| `exp-data-single-env-sample-quality` | needs env/camera restriction choice on the current flow-shortcut setup first |
