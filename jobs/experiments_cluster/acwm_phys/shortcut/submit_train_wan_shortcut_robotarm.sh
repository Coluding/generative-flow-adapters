#!/bin/bash
#SBATCH --job-name=acwm-robotarm-wan-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/shortcut/wan-shortcut-%x-%j.out
#SBATCH --error=logs/shortcut/wan-shortcut-%x-%j.err

# D3 SHORTCUT — Wan2.2 (FLOW) · ACWM Robot Arm · action-free, step-size-conditioned.
# The flow arm of the flow-vs-diffusion few-step test (thesis-vault
# 20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md). Flow has
# near-straight trajectories → few-step shortcut SHOULD work; Wan working alone
# is already the headline D3 result.
# Recipe: use_step_level_conditioning, shortcut_direction_weight 1.0,
# multistep_consistency_weight 1.0, shortcut_anchor_prob 0.5 (non-inert — the
# gate passed locally 2026-07-28: both shortcut losses > 0, adapter not cloning).
# Action-free: action_dropout 1.0 + drop_condition 1.0 (adapter sees only step_level).
#
# PREREQUISITES (login node): same as the Wan action run —
#   download_acwmphys_robotarm.sh, submit_precompute_acwmphys_robotarm.sh,
#   submit_precompute_prompts_acwm_robotarm.sh (T5 contexts required at startup).

set -euo pipefail
module purge
module load 2024

export GFA_PROFILE=0
export GFA_DEBUG_CACHE=0
# Shortcut evaluates the frozen 5B at MULTIPLE sub-steps per training step
# (2–3x the forwards of the action runs) → peak memory is much higher, so start
# at a LOWER batch than the action baseline's 12. Raise if VRAM allows.
export BATCH_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/shortcut
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/kinematics/robot_arm"
CACHE="$ROOT/latents.shared"
for d in "$ROOT/ind_train/metadata.pt" "$ROOT/ind_test/metadata.pt"; do
    test -f "$d" || { echo "Error: $d missing — run download_acwmphys_robotarm.sh first" >&2; exit 1; }
done
test -d "$CACHE" || echo "WARNING: no shared latent cache at $CACHE — run submit_precompute_acwmphys_robotarm.sh first"

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --eval-data-dir "$ROOT/ind_test" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 8 --max-area 589824 --steps 5000000 \
    --wandb-run-name wan-shortcut-actionfree-robotarm "$@"
