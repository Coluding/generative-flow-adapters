#!/bin/bash
#SBATCH --job-name=acwm-robotarm-dc
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/acwm-robotarm-%x-%j.out
#SBATCH --error=logs/acwm-robotarm-%x-%j.err

# MATRIX RUN 4 — DynamiCrafter · ACWM Robot Arm (weak diffusion base).
# Uses the BaseVideoModel-conforming `dynamicrafter_video` provider (the old
# `dynamicrafter` UNet wrapper was non-conforming — this is that migration).
# DC encodes latents LIVE via its 4-ch SD-VAE, so NO precompute is needed —
# only the raw mp4 + metadata and the DC .ckpt.
# Ticket: thesis-vault 20_Tickets/experiments/exp-backbone-dc-robotarm-run.md
#
# Geometry (per the config header): --target-height 384 --target-width 512
# (latent 48x64, preserves the 4:3 landscape), --frame-stride 4 (~64 of 128
# frames per 16-frame window). Tune if OOM.
# BEFORE trusting results: DC base-coherence probe on robot_arm (DC512 is a
# real-world I2V prior; arm should sit close to it).
#
# PREREQUISITE (login node): robot_arm raw data present under scratch-shared
#   bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh

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

ROOT="$HOME/scratch-shared/acwm-phys/kinematics/robot_arm"
test -f "$ROOT/ind_train/metadata.pt" || { echo "Error: $ROOT/ind_train/metadata.pt missing — run download_acwmphys_robotarm.sh" >&2; exit 1; }
test -f "ckts/dynami512.ckpt" || { echo "Error: DC checkpoint ckts/dynami512.ckpt missing" >&2; exit 1; }

# DC reads the .ckpt (incl. VAE) via the config's pretrained_model_name_or_path.
python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --frame-stride 4 --target-height 384 --target-width 512 \
    --batch-size $BATCH_SIZE --steps 5000000 "$@"
