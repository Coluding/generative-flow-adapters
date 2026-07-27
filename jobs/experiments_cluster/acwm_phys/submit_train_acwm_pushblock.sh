#!/bin/bash
#SBATCH --job-name=acwm-pushblock-gatelow-capshift
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/acwm-pushblock-%x-%j.out
#SBATCH --error=logs/acwm-pushblock-%x-%j.err

# First ACWM-Phys Push Cube training run: gatelow + gate_cap 0.9 +
# sigma_shift 5.0 on the action-informative benchmark (D2 claim (a) vehicle).
# PREREQUISITES: download + precompute jobs completed
# (jobs/download_acwmphys.sh on login node, then
#  sbatch jobs/submit_precompute_acwmphys.sh).
#
# Readout: action-probe the checkpoint (--sigma-sweep --action-probe in
# scripts/generate_wan22_i2v_compare.py) — shuffle-gap > 0 here would be the
# first action-following signal of the project.

set -euo pipefail

module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
export BATCH_SIZE=12

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/rigid_dynamics/push_block"CACHE="$ROOT/latents.shared"

for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run jobs/download_acwmphys.sh first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no shared latent cache at $CACHE — run jobs/submit_precompute_acwmphys.sh first (training will encode on the fly, slow)"

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name acwm-pushblock-gatelow-cap09-shift5 "$@"
