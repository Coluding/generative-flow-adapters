#!/bin/bash
#SBATCH --job-name=gfa-train-avid-shortcut-action
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
# Shortcut + action combined run: anchor_prob 0.5 with action conditioning ON.
# Differs from diffusion_avid_shortcut_metaworld.yaml in exactly
# shortcut_anchor_prob 1.0 -> 0.5 (plus run naming) for a clean A/B.
# BEFORE launching, work through the "VERIFY BEFORE THE HPC RUN" checklist in
# the config header (real `act` in the batch; a_t fixed across the
# self-consistency micro-step).
# Vault ticket: 20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter.md
srun uv run python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_avid_shortcut_action_metaworld.yaml \
    --hdf5 "$ds_path" \
    --batch-size $BATCH_SIZE "$@"
