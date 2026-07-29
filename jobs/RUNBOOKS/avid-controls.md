# AVID controls — does the ORIGINAL recipe follow actions?

**Question:** the AVID recipe (frozen base + a full separate action-UNet) is the
control for our lightweight adapter. Does *it* follow actions — on its own
in-distribution data (RT-1) and on our OOD synthetic ACWM? This separates
"our-adapter problem" from "data problem".

**Metric:** `action_effect_rel` (shuffle/zero) from the AVID-side probe
`external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py`
(reproduces our `eval_action_effect_rel` on `AVIDAdapter.apply_model`), with
`base_null_violation ≈0` as the trust control.

## Prerequisite — the AVID repo is gitignored
`external_repos/` doesn't reach the cluster via `git pull`. rsync it once (and
re-rsync the probe when it changes — it's the only copy with `--config/--ckpt-dir/--data-dir`):
```bash
rsync -av --exclude=outputs --exclude='.venv*' --exclude='*.ckpt' \
  external_repos/avid/  <cluster>:~/generative-flow-adapters/external_repos/avid/
```

## Train the AVID references (the training runs)
- **RT-1 (official, 111 GB):** `jobs/experiments_cluster/avid_official/` — download_rt1.sh →
  setup_avid_env_cluster.sh → submit_train_avid_rt1.sh. See that dir's README.
- **ACWM (robot arm / push cube):** `configs/train/avid/avid_11M_acwm_*.yaml` via
  the AVID repo's `scripts/train.sh` (needs the ACWMVideoDataModule shim, already
  in the AVID tree).

## Probe a checkpoint (the measurement)
RT-1 (sbatch wrapper exists):
```bash
sbatch jobs/experiments_cluster/avid_official/submit_probe_rt1_action.sh
```
ACWM (direct, in the AVID poetry env — set the two paths):
```bash
cd external_repos/avid/latent_diffusion
poetry run python scripts/probe_action_sensitivity.py \
  --config configs/train/avid/avid_11M_acwm_robotarm.yaml \
  --ckpt-dir <the robotarm checkpoints dir> \
  --data-dir ~/scratch-shared/acwm-phys/kinematics/robot_arm/ind_train --num-batches 8
```

## Results / status
- **RT-1 `93qrvr5v`: FOLLOWS actions** — `action_effect_rel 0.0495`, null 0, ~66% of
  the adapter's contribution action-driven (vs our ~5%). ~10× our ACWM blind runs.
  → the blindness is a DATA problem. Note:
  `30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control.md`.
- **ACWM Push Cube: 0.0015 (blind) but CONFOUNDED** — that run was `max_clips: 64`
  (tiny-data memorization). Do not cite as a clean control.
- **ACWM Robot Arm: PENDING** — the full-data run's checkpoint (Snellius) is the
  clean same-recipe control: blind there ⇒ nails "it's the data"; follows there ⇒
  points at our adapter. The most-informative probe left.
