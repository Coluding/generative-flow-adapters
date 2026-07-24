#!/bin/bash
#SBATCH --job-name=profile-train-step
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=logs/profile-train-%x-%j.out
#SBATCH --error=logs/profile-train-%x-%j.err

# ============================================================================
# Closes the "drop precompute?" verdict: the REAL training step time vs the
# online VAE-encode cost, measured in the SAME loop with CUDA-synced timers.
#
# This is NOT a reimplementation — it runs the actual training script with
# GFA_PROFILE=1, which turns on the trainer's per-phase timers (each block is
# torch.cuda.synchronize()-wrapped, so the numbers are real, not async lies).
# Per step it prints:
#     data(load+wait)                 — dataloader (hidden by workers)
#     preprocess (VAE encode + cond)  — the ONLINE encode + resize tax
#       forward / backward / optimizer.step
#     training_step (fwd+bwd+opt) TOTAL
#
# Two variants at the real ACWM geometry (768^2, 65f):
#   A. online encode  (--no-latent-cache)      -> preprocess = full encode cost
#   B. cached latents (--latent-cache-dir ...)  -> preprocess ~ cache-load only
# The per-step wall-time delta (A - B) IS the online-encode tax; compare it to
# 'training_step TOTAL' to decide whether pre-encoding can be dropped.
#
# Variant B is skipped automatically if the shared latent cache isn't warm
# (run submit_precompute_acwmphys.sh first to populate it).
#
# Extra flags forwarded to BOTH variants: sbatch submit_profile_train_step.sh --batch-size 2
# ============================================================================

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/projects/generative-flow-adapters"
mkdir -p logs
source .venv/bin/activate

# Turn on the trainer's CUDA-synced phase timers.
export GFA_PROFILE=1

CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
ROOT="ds/acwm-phys/rigid_dynamics/push_block"
DATA_DIR="$ROOT/ind_train"
CACHE="$ROOT/latents.shared"     # same shared cache submit_precompute_acwmphys.sh writes

# Common args: real geometry, generation-eval OFF, loss-eval pushed past the
# profiled window so nothing but train steps run. --steps 15 gives a couple of
# warmup steps (cuDNN autotune / allocator growth) plus ~12 steady-state steps
# to average by eye. num-workers 8 hides decord decode so 'preprocess' isolates
# the encode+resize (the external training script sets a spawn context, so
# workers are fork-safe here — unlike the plain profile_vae DataLoader).
COMMON=(
    --config "$CONFIG"
    --dataset acwm_phys --data-dir "$DATA_DIR"
    --ckpt-dir ckpts/Wan2.2-TI2V-5B
    --max-area 589824 --temporal-length 65
    --num-windows 8 --num-workers 8
    --batch-size 1
    --steps 15 --log-every 1 --eval-every 100000 --no-eval-gen
)

echo "###############################################################################"
echo ">>> VARIANT A — ONLINE encode (--no-latent-cache)   ($(date))"
echo "###############################################################################"
python scripts/train_wan22_i2v_metaworld_external.py "${COMMON[@]}" --no-latent-cache "$@"

if test -d "$CACHE" && test -n "$(ls -A "$CACHE" 2>/dev/null)"; then
    echo "###############################################################################"
    echo ">>> VARIANT B — CACHED latents (--latent-cache-dir $CACHE)   ($(date))"
    echo "###############################################################################"
    python scripts/train_wan22_i2v_metaworld_external.py "${COMMON[@]}" --latent-cache-dir "$CACHE" "$@"
else
    echo "SKIP variant B: shared latent cache '$CACHE' is empty/missing."
    echo "     Run submit_precompute_acwmphys.sh first to populate it, then re-run."
fi

echo "All variants done ($(date))."
echo "Read: per step compare 'preprocess (VAE encode + cond)' (A) vs 'training_step ... TOTAL'."
echo "      A_preprocess - B_preprocess = the online-encode tax; if TOTAL >> that, drop precompute."
