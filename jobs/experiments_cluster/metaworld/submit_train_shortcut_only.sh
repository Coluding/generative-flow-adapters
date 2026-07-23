#!/bin/bash
#SBATCH --job-name=gfa-train-shortcut-only
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

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
# Config carries grad_accum_steps: 4 -> effective batch 8 at batch-size 2,
# identical to the gatelow baseline it forks so action-ON vs action-OFF is the
# only variable.
export BATCH_SIZE=${BATCH_SIZE:-2}

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

export ds_path="${DS_PATH:-../scratch-shared/metaworld/mw_zoom13.hdf5}"

echo "Running training with dataset: $ds_path"

if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path"
    exit 1
fi

export latent_path=${ds_path%.hdf5}_latent.hdf5

if test -f "$latent_path"; then
    echo "Found latent dataset file at $latent_path"
else
    echo "Warning: Latent dataset file not found at $latent_path"
    echo "Should run slower"
    echo "==============================================================================="
fi

# ---- run --------------------------------------------------------------------
# Action-free shortcut-only adapter (pure D3 test): step-size-conditioned,
# no action input; whole job is distilling the frozen base into a good
# few-step generator. Headline readout: few-step adapted quality vs the frozen
# base at the same NFE budget (quantified — PSNR/SSIM/LPIPS/FVD per step row).
# Vault ticket: 20_Tickets/experiments/exp-shortcut-action-free-isolation.md
python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/flow_wan22_shortcut_only_metaworld.yaml \
    --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name shortcut-only-actionfree "$@"
