#!/bin/bash
#SBATCH --job-name=precompute-skyreels-robotarm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/preencode/precompute-skyreels-robotarm-%x-%j.out
#SBATCH --error=logs/preencode/precompute-skyreels-robotarm-%x-%j.err

# SkyReels z0-latent prewarm for ACWM-Phys Robot Arm (all splits -> shared cache).
# Fills $ROOT/skyreels.latents.shared so submit_train_skyreels_robotarm.sh reads
# 16-ch z0 from disk instead of re-encoding on the first pass.
#
# MUST use the SAME --config and --num-windows as the training job
# (submit_train_skyreels_robotarm.sh: config *_acwm_robotarm.yaml, --num-windows 8)
# or the cache keys won't match and training re-encodes anyway.
# PREREQUISITE: ACWM robot_arm downloaded (jobs/.../download_acwmphys_robotarm.sh).
# SkyReels weights auto-download from HF on first run (no --ckpt-dir).
#
# Resumable: existing cache entries are counted as hits and skipped. Job 24925783
# ran this on gpu_a100 at 0.3 win/s and hit the 12h wall at 14756/17656 windows;
# gpu_h100 does 0.6 win/s (measured on push_block), so the rest is ~1-2h.
#
# BATCH SIZE: robot_arm is 97f @ 384x512 vs push_block's 65f @ 384x384 — ~2.9x
# the voxels per clip, so it does NOT inherit push_block's --batch-size 24.
# Job 24944891 OOMed there (10.2 GiB single alloc inside the VAE pad, on a 93 GiB
# H100). expandable_segments reclaims the ~26 GiB the allocator was holding
# reserved-but-unallocated after the cache-hit scan phase.

set -euo pipefail

module purge
module load 2024

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/kinematics/robot_arm"
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
        --batch-size 6 --num-workers 8
done

echo "All robot_arm splits done ($(date)). Shared SkyReels cache at $CACHE"
