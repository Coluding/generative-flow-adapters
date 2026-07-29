#!/bin/bash
#SBATCH --job-name=rt1-wan-action
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/rt1/wan-action-%x-%j.out
#SBATCH --error=logs/rt1/wan-action-%x-%j.err

# THE in-distribution ACTION test — OUR Wan adapter on RT-1 (real robot video).
# Does our lightweight output adapter FOLLOW ACTIONS in-distribution, where it
# went blind on OOD synthetic ACWM? The direct our-adapter counterpart to the
# AVID-RT1 control (93qrvr5v, which followed actions). thesis-vault
# 30_Knowledge/experiments/20260729-avid-rt1-follows-actions-control.md.
# Watch: eval_action_effect_rel (blind ~0.003 on ACWM; should be HIGHER here),
# condition_grad_norm / action_inject_grad_norm, base_null_violation (~0).
#
# PREREQUISITES (in order): convert_rt1.sh -> submit_precompute_rt1_latents.sh
# -> submit_precompute_rt1_captions.sh. --num-windows MUST match the precompute.

set -euo pipefail
module purge; module load 2024
export GFA_PROFILE=0
export BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
source .venv/bin/activate

RT1_DIR="${RT1_OUT:-$HOME/scratch-shared/rt1/train}"
CACHE="$RT1_DIR/latents.shared"
NUM_WINDOWS="${NUM_WINDOWS:-2}"   # must match submit_precompute_rt1_latents.sh
test -f "$RT1_DIR/metadata.pt" || { echo "Error: $RT1_DIR/metadata.pt missing — run convert_rt1.sh" >&2; exit 1; }
test -f "configs/prompts/rt1_captions.contexts.pt" || { echo "Error: per-clip captions missing — run submit_precompute_rt1_captions.sh" >&2; exit 1; }
test -d "$CACHE" || echo "WARNING: no latent cache at $CACHE — run submit_precompute_rt1_latents.sh (else on-the-fly encode, slow)"

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_action_rt1.yaml \
    --dataset rt1 \
    --data-dir "$RT1_DIR" \
    --eval-data-dir "$RT1_DIR" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows $NUM_WINDOWS --max-area 589824 --steps 5000000 \
    --wandb-run-name wan-action-rt1 "$@"
