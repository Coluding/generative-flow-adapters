#!/bin/bash
#SBATCH --job-name=precompute-acwm-pushblock
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/precompute-acwm-%x-%j.out
#SBATCH --error=logs/precompute-acwm-%x-%j.err

# VAE-latent precompute for ACWM-Phys push_block (all three splits).
# PREREQUISITE (login node): bash jobs/download_acwmphys.sh  AND  git pull
# (needs the ACWMPhysTranslator commit).
#
# Geometry MUST match the training config
# (diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml):
# temporal_length 41, max_area 589824, num-windows 8 — the latent-cache keys
# bake all of these in. ~13.6k windows total; VAE-only, no 5B DiT loaded.

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="/scratch-shared/$USER/acwm-phys/rigid_dynamics/push_block"
CONFIG="configs/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
NUM_WINDOWS=8   # must match training's --num-windows

for split in ind_train ind_test ood_test; do
    d="$ROOT/$split"
    if test ! -f "$d/metadata.pt"; then
        echo "SKIP $split: $d/metadata.pt not found (run jobs/download_acwmphys.sh on the login node)" >&2
        continue
    fi
    echo "==============================================================================="
    echo ">>> precompute: $split  ($(date))"
    echo "==============================================================================="
    python scripts/precompute_latents.py \
        --config "$CONFIG" \
        --dataset acwm_phys --data-dir "$d" \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --num-windows $NUM_WINDOWS --max-area 589824 \
        --batch-size 2 --num-workers 8
done

echo "All splits done ($(date)). Caches at $ROOT/<split>.latents/"
