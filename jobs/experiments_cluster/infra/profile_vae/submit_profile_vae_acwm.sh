#!/bin/bash
#SBATCH --job-name=profile-vae-acwm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/profile-vae-%x-%j.out
#SBATCH --error=logs/profile-vae-%x-%j.err

# ============================================================================
# Experiment 1/3 — ACWM push_block at the REAL training geometry.
#
# Question: at 768^2 (max_area 589824), 65-frame windows, is ONLINE VAE
# encoding cheap enough to drop the precompute step entirely? This is the
# geometry training actually runs at, so its ms/clip is the number to compare
# against the training step time.
#
# VAE only — no DiT/adapter/optimizer. Latent cache is DISABLED inside the
# script, so every batch genuinely encodes. Peak-alloc here is the encode
# transient in ISOLATION; experiment 3 measures it alongside the resident 5B.
#
# num-workers 0 on purpose: ACWM uses decord, and this profile script builds a
# plain DataLoader without the spawn context, so workers>0 can fork-deadlock
# (see the ACWM decord fork-deadlock fix). Workers only hide the CPU resize
# cost, which the script already reports separately as 'resize+h2d'.
#
# Extra flags forwarded: sbatch submit_profile_vae_acwm.sh --batch-size 1 2 4 8
# ============================================================================

set -euo pipefail

module purge
module load 2024

# uv env (in case .bashrc isn't sourced on compute nodes)
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/projects/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
DATA_DIR="ds/acwm-phys/rigid_dynamics/push_block/ind_train"

python scripts/profile_vae_encode.py \
    --dataset acwm_phys --data-dir "$DATA_DIR" \
    --config "$CONFIG" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --max-area 589824 --temporal-length 65 \
    --batch-size 1 2 4 --num-batches 12 --warmup 2 \
    --num-workers 0 --vae-dtype bf16 \
    "$@"
