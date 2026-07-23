#!/bin/bash
#SBATCH --job-name=gfa-train-avid-shortcut
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

export BATCH_SIZE=48

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

# ---- gpu usage monitor ------------------------------------------------------
# Prints GPU utilisation / memory every GPU_LOG_INTERVAL seconds in the
# background. The monitor is killed automatically when the script exits.
export GPU_LOG_INTERVAL=${GPU_LOG_INTERVAL:-30}
(
    while true; do
        echo "==== nvidia-smi $(date '+%Y-%m-%d %H:%M:%S') ===="
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader
        sleep "$GPU_LOG_INTERVAL"
    done
) &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

# ---- run --------------------------------------------------------------------
srun uv run python scripts/train_wan22_i2v_metaworld.py \
    --config configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml \
    --hdf5 "../scratch-shared/metaworld/mw_zoom13.hdf5" \
    --batch-size $BATCH_SIZE "$@"
