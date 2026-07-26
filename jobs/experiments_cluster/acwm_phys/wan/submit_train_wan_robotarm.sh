#!/bin/bash
#SBATCH --job-name=acwm-robotarm-wan-baseline
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/wan/acwm-robotarm-%x-%j.out
#SBATCH --error=logs/wan/acwm-robotarm-%x-%j.err

# MATRIX RUN 1 — Wan2.2 · ACWM Robot Arm (clean baseline on the rich domain).
# Motivation (measured 2026-07-25): frozen-base masked denoise loss 0.314 at 17f
# on Robot Arm vs ~0.036 on Push Cube (~8.7x more residual) — the base leaves a
# real error for the action adapter to close, so it should stop cloning.
# Ticket: thesis-vault 20_Tickets/experiments/exp-backbone-wan-robotarm-run.md
#
# PREREQUISITES (login node, then this):
#   bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh
#   sbatch jobs/experiments_cluster/infra/submit_precompute_acwmphys_robotarm.sh
# T5 contexts already precomputed (configs/prompts/acwm_robotarm.contexts.pt).

set -euo pipefail

module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
export BATCH_SIZE=6   # robot_arm windows are 25 latent frames (vs pushblock 17); lower if OOM

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

# Robot Arm raw + latent cache live under scratch-shared (see the robot_arm
# download/precompute scripts — different location from push_block's in-repo ds/).
ROOT="../scratch-shared/acwm-phys/rigid_dynamics/push_block"
CACHE="$ROOT/latents.shared"

for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run download_acwmphys_robotarm.sh first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no shared latent cache at $CACHE — run submit_precompute_acwmphys_robotarm.sh first (training will encode on the fly, slow)"

# --num-windows 8 MUST match the precompute job (cache keys bake it in).
python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name acwm-robotarm-wan-cap09-shift5 "$@"
