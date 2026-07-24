#!/bin/bash
#SBATCH --job-name=triangle-replace-nobase
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=22:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Base-parity campaign (2026-07-21), ARM 1 of 4 — replace-nobase overfit:
# no gate, no base input. The capacity/trap control: with the identity-copy
# shortcut removed the DiT must denoise on its own, separating an optimization
# trap from a 34M capacity limit.
#
# Bounded at 1000 steps (the uxrst2k5 collapse signal shows by step ~150, so
# 1000 is generous) — the 12h walltime is only a ceiling.
#
# Readout: denoise loss vs the base ~0.05-0.08 floor.
#   well below base ⇒ base-input was the trap
#   at base        ⇒ 34M capacity limit
#
# Vault ticket: 20_Tickets/experiments/exp-adapter-replace-nobase-overfit.md
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

export GFA_PROFILE=1
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
    --config configs/wan22/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml \
    --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B --max-area 589824 \
    --overfit-index 0 --num-windows 1 --steps 1000 --batch-size $BATCH_SIZE \
    --wandb-run-name overfit-single-clip-replace-nobase "$@"
