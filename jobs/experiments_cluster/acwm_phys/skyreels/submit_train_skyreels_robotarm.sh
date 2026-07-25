#!/bin/bash
#SBATCH --job-name=acwm-robotarm-skyreels
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/acwm-robotarm-skyreels-%x-%j.out
#SBATCH --error=logs/acwm-robotarm-skyreels-%x-%j.err

# SkyReels-V2-I2V-1.3B x ACWM-Phys Robot Arm (matrix run 3) — the WEAK flow base
# on the VISUALLY-RICH da=7 domain (the clean base-strength arena: the frozen Wan
# base leaves a large residual here, 0.314 at 17f vs Push Cube 0.036, and
# SkyReels' natural-video prior matches realistic 3D). SkyReels weights
# auto-download from HF on first run (Skywork/SkyReels-V2-I2V-1.3B-540P).
# PREREQUISITE: ACWM robot_arm downloaded (jobs/.../download_acwmphys_robotarm.sh).
# NOTE (draft): validate the i2v seam with a short --steps smoke first.

set -euo pipefail

module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
export BATCH_SIZE=6

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="$HOME/scratch-shared/acwm-phys/kinematics/robot_arm"
CACHE="$ROOT/skyreels.latents.shared"

for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run the ACWM robot_arm download first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no SkyReels latent cache at $CACHE — training will encode z0 on the fly (slower)"

python scripts/train_skyreels_acwm.py \
    --config configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name acwm-robotarm-skyreels "$@"
