#!/bin/bash
#SBATCH --job-name=acwm-pushblock-dc
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/dc/acwm-pushblock-dc-%x-%j.out
#SBATCH --error=logs/dc/acwm-pushblock-dc-%x-%j.err

# MATRIX RUN 5 — DynamiCrafter · ACWM Push Cube (weak diffusion base, flat 2D).
# `dynamicrafter_video` provider, live 4-ch SD-VAE encode, no precompute.
# Ticket: thesis-vault 20_Tickets/experiments/exp-backbone-dc-pushblock-run.md
#
# CAVEAT: DC512's real-world prior is likely OOD for flat 2D vector art — run a
# DC base-coherence probe first; expect the flat-visuals ceiling (small base
# residual → cloning pressure), same trap the Wan/SkyReels bases hit on flat art.
# Geometry: --target-height 512 --target-width 512 (square source), fs 4.
#
# PREREQUISITE: push_block raw data in-repo (jobs/download_acwmphys.sh).

set -euo pipefail

module purge
module load 2024

export GFA_PROFILE=0
export BATCH_SIZE=8

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

ROOT="/home/lbierling/scratch-shared/acwm-phys/rigid_dynamics/push_block"
test -f "$ROOT/ind_train/metadata.pt" || { echo "Error: $ROOT/ind_train/metadata.pt missing — run jobs/download_acwmphys.sh" >&2; exit 1; }
test -f "ckts/dynami512.ckpt" || { echo "Error: DC checkpoint ckts/dynami512.ckpt missing" >&2; exit 1; }

python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_dc_acwm_pushblock.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --frame-stride 4 --target-height 512 --target-width 512 \
    --batch-size $BATCH_SIZE --steps 5000000 "$@"
