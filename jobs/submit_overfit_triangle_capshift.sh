#!/bin/bash
#SBATCH --job-name=overfit-triangle-capshift
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/overfit-triangle-%x-%j.out
#SBATCH --error=logs/overfit-triangle-%x-%j.err

# One job, four sequential runs (2026-07-21 base-parity campaign):
#   1. replace-nobase overfit        (no gate, no base input)   — capacity/trap control
#   2. gatelow-nobase overfit, raw   (gate, no base input)      — pure input ablation vs uxrst2k5
#   3. gatelow-nobase overfit, cap09 (capped gate, no base input) — guaranteed-interpretable arm
#   4. full-data gatelow + gate_cap 0.9 + sigma_shift 5.0        — runs until walltime
# Arms 1-3 are bounded (1000 steps each — the uxrst2k5 collapse signal shows
# by step ~150, so 1000 is generous); arm 4 gets the remainder.
# Each arm is failure-guarded: a crash logs and moves on to the next arm.
#
# PREREQUISITE: git pull on the login node first — needs the 2026-07-20/21
# commit (eval action_seq fix, gate_cap, sigma_shift, these configs).

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
source .venv/bin/activate

# ---- sanity check -----------------------------------------------------------
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"

export ds_path="../scratch-shared/metaworld/five_task_diverse.hdf5"
echo "Running with dataset: $ds_path"
if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path"
    exit 1
fi

export latent_path=${ds_path%.hdf5}.latents
if test -d "$latent_path" || test -f "$latent_path"; then
    echo "Found latent cache at $latent_path"
else
    echo "WARNING: no latent cache at $latent_path — first accesses will encode (slower)"
fi

COMMON="--hdf5 $ds_path --ckpt-dir ckpts/Wan2.2-TI2V-5B --max-area 589824"
OVERFIT="--overfit-index 0 --num-windows 1 --steps 1000 --batch-size $BATCH_SIZE"

run_arm () {
    local label="$1"; shift
    echo "==============================================================================="
    echo ">>> ARM: $label  ($(date))"
    echo "==============================================================================="
    if python "$@"; then
        echo ">>> ARM OK: $label ($(date))"
    else
        echo ">>> ARM FAILED (continuing): $label ($(date))"
    fi
}

# ---- 1) replace-nobase overfit ----------------------------------------------
run_arm "replace-nobase-overfit" scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml \
    $COMMON $OVERFIT \
    --wandb-run-name overfit-single-clip-replace-nobase

# ---- 2) gatelow-nobase overfit, raw gate ------------------------------------
run_arm "gatelow-nobase-overfit-raw" scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_gatelow_nobase_overfit_metaworld.yaml \
    $COMMON $OVERFIT \
    --wandb-run-name overfit-single-clip-gatelow-nobase

# ---- 3) gatelow-nobase overfit, gate capped at 0.9 --------------------------
run_arm "gatelow-nobase-overfit-cap09" scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_gatelow_nobase_gatecap_overfit_metaworld.yaml \
    $COMMON $OVERFIT \
    --wandb-run-name overfit-single-clip-gatelow-nobase-cap09

# ---- 4) full-data gatelow + gate_cap + sigma_shift (until walltime) ---------
run_arm "gatelow-cap09-sigmashift5-fulldata" scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_gatelow_cap_sigmashift_metaworld.yaml \
    $COMMON \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name gatelow-cap09-sigmashift5

echo "All arms dispatched. ($(date))"
