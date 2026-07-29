#!/bin/bash
#SBATCH --job-name=avid-rt1-action-probe
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/avid-rt1/probe-%x-%j.out
#SBATCH --error=logs/avid-rt1/probe-%x-%j.err

# Action-sensitivity probe on the OFFICIAL AVID RT-1 checkpoint — does the
# ORIGINAL AVID recipe follow actions on its OWN full real-world data? The clean
# control for our ACWM action-blindness (the 64-clip smoke was confounded; this
# is a real, fully-trained run). Reproduces our eval_action_effect_rel metric on
# AVIDAdapter.apply_model, with the frozen base as null control.
# Result comparable to our runs: Wan 0.0056 / DC 0.0034 / SkyReels 0.0013 (ACWM).
#
# PREREQUISITE: the RT-1 run finished (submit_train_avid_rt1.sh) AND the updated
# probe is on the cluster (rsync — external_repos is gitignored; see the dir README).

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
cd "$REPO"
mkdir -p "$HOME/generative-flow-adapters/logs/avid-rt1"

RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"
DC_CKPT="${DC_CKPT:-$HOME/generative-flow-adapters/ckts/dynami512.ckpt}"

# Same venv as training — 'poetry run' does not work here (see
# setup_avid_env_cluster.sh for why poetry's env layer is unusable on this cluster).
VENV="${AVID_VENV:-$REPO/.venv}"
test -x "$VENV/bin/python" || {
    echo "Error: AVID venv not found at $VENV — run setup_avid_env_cluster.sh on the login node first" >&2
    exit 1; }
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# avid_11M.yaml has name=avid_rt1_11M + logdir=/host_home/avid (a container path
# that does not exist here), so submit_train_avid_rt1.sh redirects logdir to
# scratch — that is where the checkpoints actually are, NOT the repo's outputs/.
# Keep this default in sync with LOGDIR in the training script.
LOGDIR="${LOGDIR:-/scratch-shared/$USER/avid-rt1}"
CKPT_DIR="${AVID_RT1_CKPT_DIR:-$LOGDIR/avid_rt1_11M/checkpoints}"
if ! ls "$CKPT_DIR"/epoch=*-step=*.ckpt >/dev/null 2>&1; then
    echo "No ckpt in $CKPT_DIR. Find it and set AVID_RT1_CKPT_DIR:" >&2
    find "$LOGDIR" "$REPO" -name "epoch=*-step=*.ckpt" 2>/dev/null | head >&2 || true
    exit 1
fi

# The model must be rebuilt from the SAME side configs training used: the
# vendored ones hardcode the author's DynamiCrafter path, so training generated
# rewritten copies. Regenerate them identically here (idempotent) instead of
# assuming the training job's copies are still around.
GEN="$REPO/configs/train/_cluster_generated"
mkdir -p "$GEN"
test -f "$DC_CKPT" || { echo "Error: DynamiCrafter checkpoint missing at $DC_CKPT" >&2; exit 1; }
for cfg in dynamicrafter_512.yaml act_cond_diffusion_11M.yaml; do
    sed -e "s|/home/lukas/projects/generative-flow-adapters/ckts/dynami512.ckpt|$DC_CKPT|g" \
        -e "s|/host_home/wandb/|$LOGDIR/wandb|g" \
        -e "s|/host_home/avid|$LOGDIR|g" \
        "$REPO/configs/train/$cfg" > "$GEN/$cfg"
done

python scripts/probe_action_sensitivity.py \
    --config configs/train/avid/avid_11M.yaml \
    --base-config "$GEN/dynamicrafter_512.yaml" \
    --action-config "$GEN/act_cond_diffusion_11M.yaml" \
    --ckpt-dir "$CKPT_DIR" \
    --dataset-dir "$RTX_DATA_DIR" \
    --num-batches 8 --noise-draws 3
