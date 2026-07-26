#!/bin/bash
#SBATCH --job-name=acwm-pushblock-skyreels
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/skyreels/acwm-pushblock-skyreels-%x-%j.out
#SBATCH --error=logs/skyreels/acwm-pushblock-skyreels-%x-%j.err

# SkyReels-V2-I2V-1.3B x ACWM-Phys Push Cube (matrix run 2) — the WEAK flow base
# for the base-strength axis. SkyReels weights auto-download from HF on first run
# (Skywork/SkyReels-V2-I2V-1.3B-540P); no --ckpt-dir needed.
# PREREQUISITE: ACWM push_block downloaded (jobs/.../download_acwmphys.sh).
# NOTE (draft): SkyReels is I2V-conditioned; the SkyReelsI2VPreprocessor builds
# y/clip_fea/context live. First run should be a short --steps smoke to validate
# the i2v seam before a full launch (see the module header GPU-VALIDATE notes).

set -euo pipefail

module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
export BATCH_SIZE=8

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/rigid_dynamics/push_block"
CACHE="$ROOT/skyreels.latents.shared"

for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run the ACWM push_block download first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no SkyReels latent cache at $CACHE — training will encode z0 on the fly (slower)"

python scripts/train_skyreels_acwm.py \
    --config configs/skyreels/diffusion_skyreels_xattn_acwm_pushblock.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name acwm-pushblock-skyreels "$@"
