#!/bin/bash
#SBATCH --job-name=acwm-pushblock-cap50-warmup
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/wan/acwm-pushblock-%x-%j.out
#SBATCH --error=logs/wan/acwm-pushblock-%x-%j.err

# MATRIX RUN 6 — Wan2.2 · ACWM Push Cube · gate_cap 0.5 + AVID warmup 500.
# Intervention test: can a harder cap + adapter warmup rescue the adapter from
# base-parity collapse on the near-zero-residual flat 2D domain (base loss
# ~0.036), WITHOUT moving to Robot Arm? Minimal deltas on the capshift recipe
# (sigma_shift 5.0 kept), so a diff vs the parent run isolates cap+warmup.
# Ticket: thesis-vault 20_Tickets/experiments/exp-adapter-wan-cap50-warmup-pushblock-run.md
#
# PREREQUISITE: the SAME push_block latents as the parent run — already
# precomputed (jobs/experiments_cluster/infra/submit_precompute_acwmphys.sh).
# No new precompute.

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

ROOT="../scratch-shared/acwm-phys/rigid_dynamics/push_block"
CACHE="$ROOT/latents.shared"

for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run jobs/download_acwmphys.sh first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no shared latent cache at $CACHE — run submit_precompute_acwmphys.sh first (training will encode on the fly, slow)"

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_cap50_warmup_acwm_pushblock.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name acwm-pushblock-cap50-warmup500 "$@"
