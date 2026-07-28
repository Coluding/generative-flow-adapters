#!/bin/bash
#SBATCH --job-name=acwm-robotarm-dc-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/shortcut/dc-shortcut-%x-%j.out
#SBATCH --error=logs/shortcut/dc-shortcut-%x-%j.err

# D3 SHORTCUT — DynamiCrafter (DIFFUSION) · ACWM Robot Arm · action-free.
# The diffusion arm of the flow-vs-diffusion test. Diffusion's CURVED denoising
# trajectory is why the thesis argues it's ill-suited for shortcut — so this
# uses the curvature-aware target `endpoint_inversion` (the fair, adjusted
# objective; v_average would be biased by the sagitta). DEFERRED per plan: run
# only AFTER the Wan flow arm shows few-step quality holds. See thesis-vault
# 20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md.
# NOTE: the per-clip-caption path to DC's own CLIP text encoder is untested —
# do a short --steps smoke and confirm the caption reaches DC before trusting it.
#
# PREREQUISITE (login node): download_acwmphys_robotarm.sh (DC encodes latents
# live via its 4-ch SD-VAE — no Wan-style precompute).

set -euo pipefail
module purge
module load 2024

export GFA_PROFILE=0
# Shortcut = 2–3x forwards; DC is 4-ch (lighter than Wan-5B) but still lower the
# batch vs the action run's 24.
export BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/shortcut
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/kinematics/robot_arm"
test -f "$ROOT/ind_train/metadata.pt" || { echo "Error: $ROOT/ind_train/metadata.pt missing — run download_acwmphys_robotarm.sh" >&2; exit 1; }
test -f "ckts/dynami512.ckpt" || { echo "Error: DC checkpoint ckts/dynami512.ckpt missing" >&2; exit 1; }

python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_dc_shortcut_actionfree_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --frame-stride 4 --target-height 384 --target-width 512 \
    --batch-size $BATCH_SIZE --steps 5000000 "$@"
