#!/bin/bash
#SBATCH --job-name=precompute-skyreels-robotarm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/precompute-skyreels-robotarm-%x-%j.out
#SBATCH --error=logs/precompute-skyreels-robotarm-%x-%j.err

# SkyReels z0-latent prewarm for ACWM-Phys Robot Arm (all splits -> shared cache).
# Fills $ROOT/skyreels.latents.shared so submit_train_skyreels_robotarm.sh reads
# 16-ch z0 from disk instead of re-encoding on the first pass.
#
# MUST use the SAME --config and --num-windows as the training job
# (submit_train_skyreels_robotarm.sh: config *_acwm_robotarm.yaml, --num-windows 8)
# or the cache keys won't match and training re-encodes anyway.
# PREREQUISITE: ACWM robot_arm downloaded (jobs/.../download_acwmphys_robotarm.sh).
# SkyReels weights auto-download from HF on first run (no --ckpt-dir).

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="$HOME/scratch-shared/acwm-phys/kinematics/robot_arm"
CONFIG="configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml"
CACHE="$ROOT/skyreels.latents.shared"
NUM_WINDOWS=8   # MUST match training's --num-windows

for split in ind_train ind_test ood_test; do
    d="$ROOT/$split"
    if test ! -f "$d/metadata.pt"; then
        echo "SKIP $split: $d/metadata.pt not found (run the robot_arm download on the login node)" >&2
        continue
    fi
    echo "==============================================================================="
    echo ">>> precompute skyreels robot_arm: $split  ($(date))"
    echo "==============================================================================="
    python scripts/precompute_skyreels_latents.py \
        --config "$CONFIG" \
        --dataset acwm_phys --data-dir "$d" \
        --latent-cache-dir "$CACHE" \
        --num-windows $NUM_WINDOWS \
        --batch-size 4 --num-workers 8
done

echo "All robot_arm splits done ($(date)). Shared SkyReels cache at $CACHE"
