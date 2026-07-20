#!/bin/bash
#SBATCH --job-name=gatelow-metaworld-run
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/gatelow-metaworld-run-%x-%j.out
#SBATCH --error=logs/gatelow-metaworld-run-%x-%j.err

set -euo pipefail

# ---- modules ----------------------------------------------------------------
module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
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



# ---- sanity check -----------------------------------------------------------
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda built:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"


export ds_path="../scratch-shared/metaworld/three_task.hdf5"

echo "Running training with dataset: $ds_path"

if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path"
    exit 1
fi

export latent_path=${ds_path%.hdf5}_latent.hdf5

if test -f $latent_path; then
echo "Found latent dataset file at $latent_path"
else
    echo "Error: Latent dataset file not found at $latent_path"
    echo "Should run slower"
    echo "==============================================================================="
fi


# ---- run --------------------------------------------------------------------
python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_gatelow_metaworld.yaml \
    --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE "$@" --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name gatelow-metaworld-run
