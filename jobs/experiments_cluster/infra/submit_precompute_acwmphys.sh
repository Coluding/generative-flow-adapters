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
# Geometry is read from the training config
# (diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml):
# temporal_length 65 (near-full episode), max_area 589824 (768^2 square, NO
# letterbox — the frozen base is coherent there; the earlier square=noise was
# a base-corruption bug, since fixed). The latent-cache keys bake all of these
# in, so this job and training MUST use the same config + --num-windows.
# At 65-frame windows on 66-frame episodes there are only 2 valid starts, so
# --num-windows auto-clamps to 2 (~3000 windows total across 1500 episodes).

set -euo pipefail

#module purge
#module load 2024

#export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
#export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
#export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/projects/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="$(pwd)/ds/acwm-phys/rigid_dynamics/push_block"
CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
NUM_WINDOWS=8   # must match training's --num-windows
# ONE shared cache for all splits: cache keys embed the split via env_name
# ("push_block-ind_train" etc.), and training reads train + eval batches
# through a single cache dir — pass this same path as training's
# --latent-cache-dir.
CACHE="$ROOT/latents.shared"

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
        --latent-cache-dir "$CACHE" \
        --ckpt-dir ckpts/Wan2.2-TI2V-5B \
        --num-windows $NUM_WINDOWS --max-area 589824 \
        --batch-size 4 --num-workers 8
done

echo "All splits done ($(date)). Shared cache at $CACHE"
