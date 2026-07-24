#!/bin/bash
#SBATCH --job-name=profile-vae-metaworld
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/profile-vae-%x-%j.out
#SBATCH --error=logs/profile-vae-%x-%j.err

# ============================================================================
# Experiment 2/3 — MetaWorld, same geometry, for comparison.
#
# Same 768^2 / 65-frame / bs {1,2,4} profile as experiment 1, but on the
# MetaWorld HDF5 source. MetaWorld frames come from HDF5 (no decord), so the
# resize+h2d ('other') cost and the source-decode path differ from ACWM even
# though the VAE encode transient should be near-identical (geometry is what
# drives the VAE cost). Running both isolates dataset-loader overhead from the
# encode itself.
#
# num-workers 0 kept for a like-for-like comparison with experiment 1 (HDF5 is
# fork-safer than decord, but the earlier metaworld.py __getstate__ hazard means
# we don't rely on it in this un-spawned DataLoader).
#
# Extra flags forwarded: sbatch submit_profile_vae_metaworld.sh --batch-size 1 2 4 8
# ============================================================================

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/projects/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
HDF5="ds/metaworld_corner2.hdf5"

python scripts/profile_vae_encode.py \
    --dataset metaworld --hdf5 "$HDF5" \
    --config "$CONFIG" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --max-area 589824 --temporal-length 65 \
    --batch-size 1 2 4 --num-batches 12 --warmup 2 \
    --num-workers 0 --vae-dtype bf16 \
    "$@"
