#!/bin/bash
#SBATCH --job-name=shortcut-fewstep-videos
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=logs/shortcut_eval/fewstep-%x-%j.out
#SBATCH --error=logs/shortcut_eval/fewstep-%x-%j.err

# Offline few-step video generation from a shortcut checkpoint — the reproducible
# eval_step_grid (gt|base|adapted, rows N∈{1,2,4,8,25,50}) + a step-size
# perturbation strip. Thesis figures + inspect existing runs without restarting.
# Ticket: thesis-vault 20_Tickets/experiments/exp-eval-shortcut-fewstep-videos.md
#
# Defaults to the Wan D3 shortcut arm (wan-shortcut-actionfree-robotarm) on the
# HELD-OUT robot-arm split, so a bare `sbatch <this>` produces the headline
# figure. Override any of CONFIG / CHECKPOINT / DATA_DIR / DATASET to point
# elsewhere — e.g. the DynamiCrafter arm:
#   CONFIG=configs/dc/... \
#   CHECKPOINT=outputs/dc-shortcut-actionfree-robotarm/checkpoints/best.pt \
#   DATA_DIR=$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_test \
#   sbatch jobs/experiments_cluster/shortcut_eval/submit_generate_fewstep.sh
#
# NOTE: env vars used to be REQUIRED (${CONFIG:?...}), so a bare sbatch died at
# line 1 of the job with "CONFIG: set CONFIG=..." (job 25042274). Defaults now
# stand in; the resolved triple is echoed below so the log records what ran.

set -euo pipefail
module purge; module load 2024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/shortcut_eval
source .venv/bin/activate

CONFIG="${CONFIG:-configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/wan-shortcut-actionfree-robotarm/checkpoints/best.pt}"
DATA_DIR="${DATA_DIR:-$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_test}"
DATASET="${DATASET:-acwm_phys}"
OUT_DIR="${OUT_DIR:-outputs/fewstep_videos}"

test -f "$CONFIG"     || { echo "Error: CONFIG $CONFIG not found" >&2; exit 1; }
test -f "$CHECKPOINT" || { echo "Error: CHECKPOINT $CHECKPOINT not found — has the shortcut run written one yet?" >&2; exit 1; }
test -f "$DATA_DIR/metadata.pt" || { echo "Error: $DATA_DIR/metadata.pt missing — DATA_DIR must be a split dir" >&2; exit 1; }
echo ">>> config=$CONFIG checkpoint=$CHECKPOINT data=$DATA_DIR dataset=$DATASET"

python scripts/generate_shortcut_fewstep.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" \
    --ckpt-dir "${CKPT_DIR:-ckpts/Wan2.2-TI2V-5B}" \
    --num-clips "${NUM_CLIPS:-3}" --out-dir "$OUT_DIR" \
    --step-schedule "${STEP_SCHEDULE:-1,2,4,8,25,50}" --stepsize-perturb "$@"

echo "Done -> $OUT_DIR"
