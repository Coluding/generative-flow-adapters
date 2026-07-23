#!/bin/bash
#SBATCH --job-name=gfa-test-Wan2.2-i2v
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- modules ----------------------------------------------------------------
module purge
module load 2024

# ---- uv env vars (in case .bashrc isn't sourced on compute nodes) -----------
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

# ---- project ----------------------------------------------------------------
cd "$HOME/generative-flow-adapters"
mkdir -p logs outputs

# uv sync should already have been run on the LOGIN node (no internet on
# compute nodes). This call is a no-op if the venv is up to date.
source .venv/bin/activate

# Checkpoint location for the GPU-gated test (override on the command line if
# the TI2V-5B weights live elsewhere).
export WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-ckpts/Wan2.2-TI2V-5B}

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
# -s streams the [Wan2.2-i2v] MSE print; extra args are forwarded to pytest.
srun uv run pytest tests/test_wan22_i2v.py -v -s "$@"
