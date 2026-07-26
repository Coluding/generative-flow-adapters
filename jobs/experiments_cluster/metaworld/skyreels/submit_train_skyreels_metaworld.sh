#!/bin/bash
#SBATCH --job-name=metaworld-skyreels
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/skyreels/metaworld-skyreels-%x-%j.out
#SBATCH --error=logs/skyreels/metaworld-skyreels-%x-%j.err

# SkyReels-V2-I2V-1.3B x MetaWorld (corner2). Companion to the prewarm job
# submit_precompute_skyreels_metaworld.sh — run THAT first to fill the
# .skyreels.latents cache, then this. SkyReels weights auto-download from HF on
# first run (Skywork/SkyReels-V2-I2V-1.3B-540P); no --ckpt-dir needed.
#
# MUST use the SAME --config and --num-windows as the precompute job, or the
# cache keys won't match and z0 re-encodes on the fly.
#
# ⚠️ KNOWN BLOCKER (2026-07-26): SkyReels *training* currently crashes in the i2v
# conditioning path (SkyReelsI2VPreprocessor._build_i2v_conditioning ->
# clip.encode_video: "Input type (cuda.Float) and weight type (CPUBFloat16)") —
# the offloaded CLIP (bf16, CPU) is fed a GPU tensor. The SkyReels configs are
# DRAFT (see the config header GPU-VALIDATE notes). Fix that before a real launch;
# the z0/latent-cache path is unaffected.

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
mkdir -p logs/skyreels
source .venv/bin/activate

HDF5="ds/metaworld_corner2.hdf5"
CONFIG="configs/skyreels/diffusion_skyreels_xattn_metaworld.yaml"
CACHE="${HDF5%.hdf5}.skyreels.latents"   # default cache dir (matches the precompute job)

test -f "$HDF5" || { echo "Error: $HDF5 missing" >&2; exit 1; }
test -d "$CACHE" || echo "WARNING: no SkyReels latent cache at $CACHE — run submit_precompute_skyreels_metaworld.sh first (training will encode z0 on the fly, slower)"

# --num-windows 8 MUST match submit_precompute_skyreels_metaworld.sh. No
# --latent-cache-dir: train_skyreels_acwm.py defaults it to <hdf5 stem>.skyreels.latents,
# identical to the precompute default.
python scripts/train_skyreels_acwm.py \
    --config "$CONFIG" \
    --dataset metaworld --hdf5 "$HDF5" \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name metaworld-skyreels "$@"
