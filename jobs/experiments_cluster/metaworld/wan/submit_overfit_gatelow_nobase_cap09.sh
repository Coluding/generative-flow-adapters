#!/bin/bash
#SBATCH --job-name=triangle-gatelow-nobase-cap09
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Base-parity campaign (2026-07-21), ARM 3 of 4 — gatelow-nobase overfit with
# the gate CAPPED at 0.9, no base input. The guaranteed-interpretable arm: the
# cap forces at least 10% of the gradient through the adapter, so the run
# cannot end in the "gate pinned, nothing learned" degenerate state.
#
# Bounded at 1000 steps (the uxrst2k5 collapse signal shows by step ~150, so
# 1000 is generous) — the 12h walltime is only a ceiling.
#
# Readout: adapter_gate_mean / adapter_grad_norm, plus does the prediction
# improve given the guaranteed >=10% gradient.
#   pred improves & gate mixed ⇒ the composition works without the crutch
#
# Vault ticket: 20_Tickets/experiments/exp-adapter-gatelow-nobase-overfit.md
#   (this is the cap09 arm of that ticket's two-arm plan)
#
# PREREQUISITES:
#   - git pull on the login node (needs the 2026-07-20/21 commit: eval
#     action_seq fix, gate_cap, sigma_shift, these configs)
#   - a WARM latent cache — see the cache guard below. The four arms of this
#     campaign are independent jobs now and can run concurrently; on a cold
#     cache they would race writing the same cache keys.

set -euo pipefail

# ---- modules ----------------------------------------------------------------
module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
export BATCH_SIZE=${BATCH_SIZE:-12}

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

export ds_path="${DS_PATH:-../scratch-shared/metaworld/three_task.hdf5}"
echo "Running with dataset: $ds_path"
if test ! -f "$ds_path"; then
    echo "Error: Dataset file not found at $ds_path" >&2
    exit 1
fi

# ---- latent-cache guard (parallel-submission safety) ------------------------
# LatentCache.put() stages through a fixed per-key "<key>.tmp" before an atomic
# rename, so two jobs encoding the SAME key concurrently write the same temp
# file and can publish a corrupt latent. The arms of this campaign overlap on
# window 0, so a cold cache + parallel submission is a real corruption risk.
# Warm it once first:
#   sbatch jobs/experiments_cluster/infra/precompute_cache.sh   (DS_PATH=... to pick the dataset)
# Set ALLOW_COLD_CACHE=1 to override — only safe when this job runs ALONE.
export latent_path=${ds_path%.hdf5}.latents
if test -d "$latent_path" || test -f "$latent_path"; then
    echo "Found latent cache at $latent_path"
elif test "${ALLOW_COLD_CACHE:-0}" = "1"; then
    echo "WARNING: no latent cache at $latent_path — encoding on the fly (ALLOW_COLD_CACHE=1)."
    echo "WARNING: this is only safe if no sibling arm is running concurrently."
else
    echo "Error: no latent cache at $latent_path." >&2
    echo "Precompute it first, or re-submit with ALLOW_COLD_CACHE=1 if running alone." >&2
    exit 1
fi

# ---- run --------------------------------------------------------------------
python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_nobase_gatecap_overfit_metaworld.yaml \
    --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B --max-area 589824 \
    --overfit-index 0 --num-windows 1 --steps 1000 --batch-size $BATCH_SIZE \
    --wandb-run-name overfit-single-clip-gatelow-nobase-cap09 "$@"
