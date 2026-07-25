#!/bin/bash
#SBATCH --job-name=precompute-acwm-robotarm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/precompute-acwm-%x-%j.out
#SBATCH --error=logs/precompute-acwm-%x-%j.err

# VAE-latent precompute for ACWM-Phys Robot Arm (all three splits).
# PREREQUISITE (login node): bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh
# AND git pull.
#
# Geometry is read from the training config
# (diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml):
# temporal_length 97 (25 latent frames), max_area 589824 -> ~864x640 landscape
# (4:3 source, aspect-preserving, no letterbox). The latent-cache keys bake all
# of these in, so this job and training MUST use the same config + --num-windows.
# 128-frame episodes with 97-frame windows -> 32 valid starts, so --num-windows 8
# gives 8 starts/episode (~16k windows across ~2000 episodes).
# batch-size 4 (not 12): the 97f/~864x640 encode transient is larger than
# Push Cube's 65f/768^2.

set -euo pipefail

#module purge
#module load 2024
#export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
#export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
#export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="$HOME/scratch-shared/acwm-phys/kinematics/robot_arm"
CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml"
NUM_WINDOWS=8   # must match training's --num-windows
CACHE="$ROOT/latents.shared"

for split in ind_train ind_test ood_test; do
    d="$ROOT/$split"
    if test ! -f "$d/metadata.pt"; then
        echo "SKIP $split: $d/metadata.pt not found (run the robot_arm download on the login node)" >&2
        continue
    fi
    echo "==============================================================================="
    echo ">>> precompute robot_arm: $split  ($(date))"
    echo "==============================================================================="
    python scripts/precompute_latents.py \
        --config "$CONFIG" \
        --dataset acwm_phys --data-dir "$d" \
        --latent-cache-dir "$CACHE" \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --num-windows $NUM_WINDOWS --max-area 589824 \
        --batch-size 4 --num-workers 8
done

echo "All robot_arm splits done ($(date)). Shared cache at $CACHE"
