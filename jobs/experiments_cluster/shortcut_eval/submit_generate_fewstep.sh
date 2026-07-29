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
# Set CONFIG / CHECKPOINT / DATA_DIR (+ DATASET). Example:
#   CONFIG=configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml \
#   CHECKPOINT=outputs/wan-shortcut-actionfree-robotarm/checkpoints/best.pt \
#   DATA_DIR=$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_train \
#   sbatch jobs/experiments_cluster/shortcut_eval/submit_generate_fewstep.sh

set -euo pipefail
module purge; module load 2024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/shortcut_eval
source .venv/bin/activate

CONFIG="${CONFIG:?set CONFIG=<shortcut config yaml>}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<outputs/.../checkpoints/best.pt>}"
DATA_DIR="${DATA_DIR:?set DATA_DIR=<split dir with metadata.pt>}"
DATASET="${DATASET:-acwm_phys}"
OUT_DIR="${OUT_DIR:-outputs/fewstep_videos}"

python scripts/generate_shortcut_fewstep.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" \
    --num-clips "${NUM_CLIPS:-3}" --out-dir "$OUT_DIR" \
    --step-schedule "${STEP_SCHEDULE:-1,2,4,8,25,50}" --stepsize-perturb "$@"

echo "Done -> $OUT_DIR"
