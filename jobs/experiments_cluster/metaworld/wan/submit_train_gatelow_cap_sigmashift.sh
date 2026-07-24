#!/bin/bash
#SBATCH --job-name=gatelow-cap09-sigmashift5
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Base-parity campaign (2026-07-21), ARM 4 of 4 — FULL-DATA gatelow with both
# countermeasures landed at once: gate_cap 0.9 + sigma_shift 5.0. Unlike arms
# 1-3 (single-clip overfit probes) this is a real training run and uses the
# whole 32h walltime.
#
# Readout: the ACTION PROBE on its checkpoint, NOT the loss delta.
#   shuffle-gap > 0 ⇒ MetaWorld is back in the claim-(a) race
#   shuffle-gap ~ 0 ⇒ the dataset diagnosis is locked in, and the ACWM-Phys
#                     second-dataset move is confirmed
#
# Action probe (local 3090 or cluster, on any checkpoint):
#   python scripts/generate_wan22_i2v_compare.py \
#     --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_cap_sigmashift_metaworld.yaml \
#     --checkpoint <ckpt.pt> --sigma-sweep --action-probe --loss-batches 0
#
# Vault ticket: 20_Tickets/experiments/exp-adapter-gatelow-cap-sigmashift-metaworld-run.md
# Related decision: 50_Decisions/open/second-dataset-action-informativeness.md
#
# PREREQUISITES:
#   - git pull on the login node (needs the 2026-07-20/21 commit: eval
#     action_seq fix, gate_cap, sigma_shift, these configs)
#   - a WARM latent cache — see the cache guard below. This arm reads the full
#     dataset (--num-windows 8), so it overlaps the overfit arms' cache keys.

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
# file and can publish a corrupt latent. This arm walks the full dataset and so
# overlaps every other arm's keys — on a cold cache, parallel submission is a
# real corruption risk. Warm it once first:
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
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_cap_sigmashift_metaworld.yaml \
    --hdf5 "$ds_path" --ckpt-dir ckpts/Wan2.2-TI2V-5B --max-area 589824 \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name gatelow-cap09-sigmashift5 "$@"
