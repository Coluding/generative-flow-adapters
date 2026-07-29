#!/bin/bash
#SBATCH --job-name=avid-acwm-robotarm-action-probe
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/avid-acwm-robotarm/probe-%x-%j.out
#SBATCH --error=logs/avid-acwm-robotarm/probe-%x-%j.err

# Action-sensitivity probe on the OFFICIAL AVID **Robot Arm** checkpoint — the
# clean synthetic-side control. Reproduces our eval_action_effect_rel on
# AVIDAdapter.apply_model, with the frozen base as null control.
#
# THE COMPARISON THIS COMPLETES (same dataset, same probe, recipe varied):
#   ours   Wan      ncztxyyo   0.0056
#   ours   DC       c3pcewxk   0.0034
#   ours   SkyReels 8zjjn7wl   0.0013
#   AVID   RT-1     93qrvr5v   0.0495   <- different data, in-distribution
#   AVID   RobotArm  THIS               <- same data as ours, recipe held fixed
#
# Decision rule (pre-registered before looking):
#   ~0.02 or above, null clean  => AVID follows actions on Robot Arm too
#                                  => the data/OOD hypothesis is WRONG, the gap
#                                     is our implementation or our adapter
#   ~0.005 or below, null clean => AVID is blind here as well
#                                  => recipe exonerated, the substrate is the
#                                     problem; ACWM cannot carry the D2 claim
# Do NOT relax these after seeing the number.
#
# PREREQUISITE: submit_train_avid_acwm_robotarm.sh finished (or has a checkpoint
# far enough in) AND the probe script is on the cluster — external_repos/ is
# gitignored, so rsync it; see the dir README.

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
cd "$REPO"
mkdir -p "$HOME/generative-flow-adapters/logs/avid-acwm-robotarm"

ACWM_DATA_DIR="${ACWM_DATA_DIR:-$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_train}"
DC_CKPT="${DC_CKPT:-$HOME/generative-flow-adapters/ckts/dynami512.ckpt}"
test -f "$DC_CKPT" || { echo "Error: DynamiCrafter checkpoint missing at $DC_CKPT" >&2; exit 1; }

VENV="${AVID_VENV:-$REPO/.venv}"
test -x "$VENV/bin/python" || {
    echo "Error: AVID venv not found at $VENV — run setup_avid_env_cluster.sh on the login node first" >&2
    exit 1; }
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Keep this LOGDIR default in sync with the training script — the config's own
# `logdir:` is the author's local path, so training redirects to scratch and that
# is where the checkpoints actually are, NOT the repo's outputs/.
LOGDIR="${LOGDIR:-/scratch-shared/$USER/avid-acwm-robotarm}"
CKPT_DIR="${AVID_ROBOTARM_CKPT_DIR:-$LOGDIR/avid_acwm_robotarm_11M/checkpoints}"
if ! ls "$CKPT_DIR"/epoch=*-step=*.ckpt >/dev/null 2>&1; then
    echo "No ckpt in $CKPT_DIR. Find it and set AVID_ROBOTARM_CKPT_DIR:" >&2
    find "$LOGDIR" "$REPO" -name "epoch=*-step=*.ckpt" 2>/dev/null | head >&2 || true
    exit 1
fi

# Rebuild the model from the SAME side configs training used (idempotent) rather
# than assuming the training job's generated copies survived.
GEN="$REPO/configs/train/_cluster_generated"
mkdir -p "$GEN"
for cfg in dynamicrafter_512.yaml act_cond_diffusion_11M_acwm_robotarm.yaml; do
    sed -e "s|/home/lukas/projects/generative-flow-adapters/ckts/dynami512.ckpt|$DC_CKPT|g" \
        -e "s|/host_home/wandb/|$LOGDIR/wandb|g" \
        -e "s|/host_home/avid|$LOGDIR|g" \
        "$REPO/configs/train/$cfg" > "$GEN/$cfg"
done

# --data-dir overrides the config's absolute (author-local) ACWM path. Sample
# count matches the RT-1 probe (8 batches x 5 timesteps x 3 paired draws = 120)
# so the two arms are measured identically.
python scripts/probe_action_sensitivity.py \
    --config configs/train/avid/avid_11M_acwm_robotarm.yaml \
    --base-config "$GEN/dynamicrafter_512.yaml" \
    --action-config "$GEN/act_cond_diffusion_11M_acwm_robotarm.yaml" \
    --ckpt-dir "$CKPT_DIR" \
    --data-dir "$ACWM_DATA_DIR" \
    --num-batches 8 --noise-draws 3

# Sanity gate on the output: base_null_violation must be ~0. If the frozen base's
# prediction moves at all across action variants, the harness is leaking actions
# into the base and the effect_rel number is meaningless — that check is what
# made 93qrvr5v and 423pjv8y trustworthy.
