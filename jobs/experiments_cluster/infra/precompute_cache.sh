#!/bin/bash
#SBATCH --job-name=precompute-latents
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- modules ----------------------------------------------------------------
module purge
module load 2024

export BATCH_SIZE=12

# ---- uv env vars (in case .bashrc isn't sourced on compute nodes) -----------
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

# ---- project ----------------------------------------------------------------
cd "$HOME/generative-flow-adapters"
mkdir -p logs

# uv sync should already have been run on the LOGIN node (no internet on
# compute nodes). This call is a no-op if the venv is up to date.
source .venv/bin/activate


<<<<<<< HEAD
python scripts/precompute_latents.py --hdf5 ../scratch-shared/metaworld/five_task_diverse.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B --num-windows 8 --max-area 589824
=======
# Warm the latent cache for a MetaWorld dataset. Override the dataset with
# DS_PATH, e.g. for the base-parity campaign:
#   DS_PATH=../scratch-shared/metaworld/five_task_diverse.hdf5 sbatch jobs/experiments_cluster/infra/precompute_cache.sh
# Run this BEFORE submitting training arms in parallel — they share this cache
# and would otherwise race writing the same keys (see the cache guard in the
# metaworld/wan/ scripts).
export ds_path="${DS_PATH:-../scratch-shared/metaworld/three_task.hdf5}"
echo "Precomputing latents for: $ds_path"
if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path" >&2
    exit 1
fi

python scripts/precompute_latents.py --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B --num-windows 8 --max-area 589824
>>>>>>> 25c7e5042dbf7c3e06f9dc1e0d145f72bc1bbc97
