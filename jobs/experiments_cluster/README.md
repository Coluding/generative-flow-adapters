# experiments_cluster/ — Snellius job scripts, grouped by dataset

Cluster (sbatch) submission scripts for the experiment queue, organised by
dataset. Local 3090 probes live in `jobs/experiments/` and are not mirrored
here.

**Submit from the repo root** — `#SBATCH --output=logs/%x-%j.out` resolves
against the submit-time CWD:

```bash
sbatch jobs/experiments_cluster/metaworld/<script>.sh
```

All training scripts accept env overrides (`BATCH_SIZE`, `DS_PATH` where
noted) and pass trailing `"$@"` through to the python entrypoint.

## metaworld/

| Script | Config | Vault ticket (`20_Tickets/experiments/`) | Status |
|---|---|---|---|
| `submit_train_hyperalign.sh` | (script defaults, `train_hyperalign_metaworld.py`) | — | legacy baseline (was `jobs/submit_train.sh`) |
| `submit_train_avid_shortcut.sh` | `diffusion_avid_shortcut_metaworld.yaml` | `exp-adapter-output-format-affine-vs-direct` (direct arm) | ⚠ calls `train_wan22_i2v_metaworld.py` (wan2.2 script) with a **DynamiCrafter** config — see note below |
| `submit_train_wan22_avid_noshortcut.sh` | `diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml` | superseded by `exp-shortcut-zero-weight-control-run` (stale config: wrong script / 121f / 256px) | legacy |
| `submit_train_wan_shortcut.sh` | `diffusion_wan22_avid_i2v_metaworld.yaml` | — | pre-gate-fix baseline (job-name says avid-shortcut; copy-paste artifact) |
| `submit_overfit_triangle_capshift.sh` | 4 arms: replace-nobase, gatelow-nobase raw, gatelow-nobase cap09, full-data cap+sigmashift | `exp-adapter-replace-nobase-overfit`, `exp-adapter-gatelow-nobase-overfit`, `exp-adapter-gatelow-cap-sigmashift-metaworld-run` | first submit per RUNBOOK |
| `submit_train_adaln_gatelow.sh` | `diffusion_wan22_avid_gatelow_metaworld.yaml` | `exp-adapter-adaln-gatelow-metaworld-run` | NEW — controlled retest, 3 confound fixes in config |
| `submit_train_gatelow_noshortcut.sh` | `diffusion_wan22_avid_gatelow_noshortcut_metaworld.yaml` | `exp-shortcut-zero-weight-control-run` | NEW — shortcut-OFF control for the gatelow baseline |
| `submit_train_dcunet_gatelow.sh` | `diffusion_wan22_dcunet_output_metaworld.yaml` | `exp-adapter-dcunet-gatelow-capacity-run` | NEW — capacity-vs-injection test |
| `submit_train_avid_shortcut_affine.sh` | `diffusion_avid_shortcut_affine_metaworld.yaml` | `exp-adapter-output-format-affine-vs-direct` (affine arm) | NEW — equal budget to the direct arm |
| `submit_train_avid_shortcut_action.sh` | `diffusion_avid_shortcut_action_metaworld.yaml` | `exp-conditioning-add-actions-to-shortcut-adapter` | NEW — verify config-header checklist before launch |
| `submit_train_shortcut_only.sh` | `flow_wan22_shortcut_only_metaworld.yaml` (NEW config) | `exp-shortcut-action-free-isolation` | NEW — action-free pure-D3 test |

Notes:

- The three gatelow-family scripts default to `BATCH_SIZE=2` because their
  configs carry `grad_accum_steps: 4` (effective batch 8, matching the AVID
  reference run `pg3x72uc`). Don't raise it if the comparison matters.
- Wan2.2 configs whose header says so must run via
  `scripts/train_wan22_i2v_metaworld_external.py` (real pretrained weights);
  the new scripts already do.
- ⚠ **Pre-existing mismatch, not fixed here:** `submit_train_avid_shortcut.sh`
  pairs the DynamiCrafter config `diffusion_avid_shortcut_metaworld.yaml` with
  `scripts/train_wan22_i2v_metaworld.py` (which only builds the wan2.2
  provider). The matching entrypoint is `scripts/train_avid_shortcut_metaworld.py`
  (its default config is exactly this file). Fix before submitting the direct
  arm of the affine-vs-direct ablation.
- Default `DS_PATH` per script mirrors its config family
  (wan2.2 external family → `../scratch-shared/metaworld/mw_zoom13.hdf5`;
  DynamiCrafter/AVID family → `ds/metaworld_corner2_large.hdf5`). Override via
  `DS_PATH=... sbatch ...` if a ticket pins different data.

## acwm_phys/

| Script | Config | Vault ticket | Status |
|---|---|---|---|
| `submit_train_acwm_pushblock.sh` | `diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml` | (second-dataset move — `50_Decisions/open/second-dataset-action-informativeness`) | submit after `infra/submit_precompute_acwmphys.sh` finishes |

## infra/

| Script | Purpose |
|---|---|
| `download_acwmphys.sh` | LOGIN-node helper: HF download of ACWM-Phys push_block into `ds/acwm-phys/` |
| `submit_precompute_acwmphys.sh` | VAE-latent precompute for the 3 ACWM splits → `latents.shared` cache |
| `precompute_cache.sh` | MetaWorld latent precompute (`gpu_a100`) |
| `submit_test_wan22_i2v.sh` | Wan2.2 TI2V-5B GPU smoke test (pytest MSE check) |

## Open tickets intentionally NOT scripted here

| Ticket | Why |
|---|---|
| `exp-adapter-param-matched-comparison` | blocked on `feat-adapter-flops-per-step-estimator`; ~39–48-run sweep needs its own harness |
| `exp-adapter-avid-native-reference-run` | separate Poetry/torch-2.1 env (external AVID repo); running locally as `pg3x72uc` — cluster recipe lives in the ticket |
| `exp-adapter-gatelow-cap-sigmashift-metaworld-run` | covered as arm 4 of `submit_overfit_triangle_capshift.sh` |
| `exp-adapter-xattn-gatelow-metaworld-run` | already ran (`bcipghvw`, crashed @624 steps, adapter cloned base); rerun decision pending |
| `exp-shortcut-scale-episodes-longer-train` | config lives in `data/results/20261706/…`; fold in reweighting + endpoint-inversion first |
| `exp-shortcut-vs-image-only-anchor-baseline` | same config as its counterpart, only `shortcut_anchor_prob` differs; gated on the per-stepsize diagnosis |
| `exp-data-single-env-sample-quality` | needs env/camera restriction choice on the current flow-shortcut setup first |
