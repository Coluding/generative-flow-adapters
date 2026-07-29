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
# Throughput, measured 2026-07-29 on the interactive H100 (see thesis-vault
# 30_Knowledge/tech/wan-shortcut-step-throughput.md): average micro-step 22.2 s
# -> 15.4 s (1.44x) from (a) reusing the frozen-base forward the shortcut
# self-consistency prep already took at the same (x_t, t), and (b) reading the
# precomputed latents in the DataLoader workers instead of on the training
# thread (2 143 ms -> 3.5 ms per step). Both are on by default;
# GFA_NO_BASE_REUSE=1 reverts (a) for an A/B.
#
# ⚠️ grad_accum_steps: the trainer used to force a minimum of 2 micro-batches per
# optimizer step regardless of config, so every earlier robot-arm run trained at
# an EFFECTIVE batch of 24, not 12. That is fixed (it now honours the config
# default of 1). To reproduce the old dynamics exactly, set
# `training.grad_accum_steps: 2` in the YAML — at 2x the wall-clock per step.
# Batch sizing, measured 2026-07-29 on H100-94GB (job 25015341).
# bs=12 → 86.5 GiB resident, 90% of the card. Static (frozen weights + optimiser)
# is 13.0 GiB, so the marginal cost is ~5.9 GiB/sample.
#
# WHY so high: the frozen 5B does run under no_grad (adapted_model.py:128), so
# the extra BASE forwards are memory-free — but the shortcut/consistency losses
# invoke the TRAINABLE adapter 2-3x per step and every one of those retains its
# activation graph for backward, at 14 175 tokens/clip. So the original "2-3x
# forwards → start at a lower batch" instinct was right; no_grad on the base
# does not license a bigger batch here.
#
# 12 = parity with the robot-arm action baseline
# (acwm_phys/wan/submit_train_wan_robotarm.sh) so the D3 shortcut arm stays
# comparable to its D2 counterpart. Note parity is in batch size only — the
# action run at 12 uses ~37 GiB, this uses 86.5 GiB.
#
# ⚠️ 90% occupancy is tight: the periodic gen_eval (inference_every_n_steps: 200)
# must fit ~30 GiB of transients inside the already-reserved pool. That relies on
# expandable_segments (set below) reusing freed blocks. If step 200 OOMs, drop to
# 8 (~61 GiB). See thesis-vault 30_Knowledge/tech/adapter-training-vram-headroom.md.
export BATCH_SIZE=12
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
