#!/bin/bash
#SBATCH --job-name=gfa-train-avid-shortcut-affine
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=28:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- modules ----------------------------------------------------------------
module purge
module load 2024

# Equal budget across the affine and direct arms: keep batch size / steps /
# walltime identical to the direct-arm run.
export BATCH_SIZE=${BATCH_SIZE:-48}

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

# ---- sanity check -----------------------------------------------------------
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda built:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"

export ds_path="${DS_PATH:-ds/metaworld_corner2_large.hdf5}"

echo "Running training with dataset: $ds_path"

if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path"
    exit 1
fi

# ---- run --------------------------------------------------------------------
# AFFINE arm of the output-format ablation (affine per-channel scale+shift vs
# free direct delta). DynamiCrafter base + AVID 11M UNet adapter, shortcut ON.
# The direct arm is configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml -- run both
# with the same budget.
# Vault ticket: 20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct.md
srun uv run python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_avid_shortcut_affine_metaworld.yaml \
    --hdf5 "$ds_path" \
    --batch-size $BATCH_SIZE "$@"
