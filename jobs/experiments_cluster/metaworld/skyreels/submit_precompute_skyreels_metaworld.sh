#!/bin/bash
#SBATCH --job-name=precompute-skyreels-metaworld
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=logs/precompute-skyreels-metaworld-%x-%j.out
#SBATCH --error=logs/precompute-skyreels-metaworld-%x-%j.err

# SkyReels z0-latent prewarm for MetaWorld (corner2) -> <hdf5 stem>.skyreels.latents.
# Fills the cache so a SkyReels MetaWorld training run reads 16-ch z0 from disk
# instead of re-encoding on the first pass.
#
# MUST use the SAME --config and --num-windows as the MetaWorld SkyReels training
# run (config diffusion_skyreels_xattn_metaworld.yaml, --num-windows 8) or the
# cache keys won't match and training re-encodes anyway.
# SkyReels weights auto-download from HF on first run (no --ckpt-dir).

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

HDF5="ds/metaworld_corner2.hdf5"
CONFIG="configs/skyreels/diffusion_skyreels_xattn_metaworld.yaml"
NUM_WINDOWS=8   # MUST match the MetaWorld SkyReels training --num-windows

test -f "$HDF5" || { echo "Error: $HDF5 missing" >&2; exit 1; }

python scripts/precompute_skyreels_latents.py \
    --config "$CONFIG" \
    --dataset metaworld --hdf5 "$HDF5" \
    --num-windows $NUM_WINDOWS \
    --batch-size 4 --num-workers 8

echo "MetaWorld SkyReels prewarm done ($(date)). Cache at ${HDF5%.hdf5}.skyreels.latents"
