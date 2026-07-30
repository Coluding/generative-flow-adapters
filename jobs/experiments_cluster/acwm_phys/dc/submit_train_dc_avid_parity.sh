#!/bin/bash
#SBATCH --job-name=dc-avid-parity
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=32:00:00
#SBATCH --output=logs/dc-avid-parity/%x-%j.out
#SBATCH --error=logs/dc-avid-parity/%x-%j.err

# ===========================================================================
# AVID-PARITY REPRODUCTION — our DC adapter vs the AVID reference, same data
#
# Decision: thesis-vault 50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan.md
# Ticket:   thesis-vault 20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm.md
#
# WHY: the UNMODIFIED AVID recipe reaches action_effect_rel 0.029475 (null 0,
# ~42% of adapter contribution action-driven) on ACWM Robot Arm — wandb
# rqp4s3gp. OUR adapter on the SAME data, SAME frozen DC weights, SAME probe
# reaches 0.004238 (effect/adapter 0.014588) — wandb c3pcewxk. Same composition
# form (avid_mask_mix, gate_bias 0.0, condition_on_base_outputs, velocity,
# action_dim 7, AVID's own 11M UNet config), so the gap is our recipe.
#
# Two named divergences (thesis-vault 30_Knowledge/tech/avid-vs-ours-action-conditioning.md):
#   1. action_time_combine: we ADD a full-width 128 action emb onto the time emb;
#      AVID CONCATs orthogonal 64+64 halves. A large time signal can swamp the
#      action.  (adapters/output/dynamicrafter.py:84 -> add_act_time_emb)
#   2. frame_stride: our DC config ships 4, and data/translators/acwm_phys.py:215
#      does actions.reshape(length, stride, -1).sum(dim=1) -> we train on SUMS of
#      4 consecutive actions. AVID's config uses frame_stride 1, raw per-frame.
#
# Both push the same way, so a single flip could produce a misleading null ->
# three single-variable arms, plus a full-parity arm.
#
# TARGET: eval_action_effect_rel >= 0.02 with eval_action_base_null_violation ~ 0.
# The probe runs inside the eval cycle (eval_every_n_steps: 500,
# action_sensitivity_probe: true, action_sensitivity_keys: [act]) so the number
# lands in wandb automatically — no separate probe job.
#
# SUBMIT FROM THE REPO ROOT:
#   cd ~/generative-flow-adapters
#   mkdir -p logs/dc-avid-parity     # Slurm will NOT create --output's dir
#   for a in 0 0S A B C D; do ARM=$a sbatch -J dc-parity-$a \
#       jobs/experiments_cluster/acwm_phys/dc/submit_train_dc_avid_parity.sh; done
#
# All six are independent — launch together. Arms 0 and 0S are CONTROLS and are
# not optional: 0 re-measures the untreated baseline under the same 3x4 probe as
# the treatments (c3pcewxk's 0.004238 used a 2x2 probe on a crashed run), and
# 0S repeats it at a different seed to give the run-to-run noise floor. Without
# them a treatment arm reading e.g. 0.012 cannot be called movement or noise.
#
# PREREQUISITE (login node): bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh
# NOTE: DC encodes latents LIVE via its own SD-VAE — changing frame_stride or
# geometry needs NO precompute. Verified 2026-07-30.
# ===========================================================================

set -euo pipefail
module purge
module load 2024

export GFA_PROFILE=0
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/dc-avid-parity
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/kinematics/robot_arm"
test -f "$ROOT/ind_train/metadata.pt" || { echo "Error: $ROOT/ind_train/metadata.pt missing — run download_acwmphys_robotarm.sh" >&2; exit 1; }
test -f "ckts/dynami512.ckpt" || { echo "Error: DC checkpoint ckts/dynami512.ckpt missing" >&2; exit 1; }

CFG_DIR=configs/dynamicrafter
ARM="${ARM:?set ARM=0|0S|A|B|C|D — see the header}"

# Each arm's frame_stride/geometry is set BOTH here and in the arm's config
# `data:` block, to the same value. The CLI flag wins
# (train_avid_shortcut_metaworld.py:374); the config keeps its copy so it reads
# truthfully on its own. Keep the two in sync when editing.
# SEED defaults to 0 but arms 0/0S override it below — their seed IS the treatment.
SEED="${SEED:-0}"
case "$ARM" in
  # --- controls: untreated baseline, re-measured under the SAME probe as the
  # treatment arms. 0 and 0S differ ONLY by seed -> their gap is the noise floor
  # for eval_action_effect_rel. Judge every treatment arm against THAT, not
  # against c3pcewxk's 2x2-probe number from a crashed run.
  0) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_arm0_baseline.yaml
     STRIDE=4; HEIGHT=384; SEED=0; OUT=outputs/dc_acwm_robotarm_arm0_baseline_s0
     DESC="CONTROL: untreated baseline (add + stride 4), seed 0" ;;
  0S) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_arm0_baseline.yaml
     STRIDE=4; HEIGHT=384; SEED=1; OUT=outputs/dc_acwm_robotarm_arm0_baseline_s1
     DESC="CONTROL: untreated baseline, seed 1 (noise floor vs arm 0)" ;;
  A) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_armA_concat.yaml
     STRIDE=4; HEIGHT=384; OUT=outputs/dc_acwm_robotarm_armA_concat
     DESC="concat only (isolates the time+action subspace)" ;;
  B) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_armB_stride1.yaml
     STRIDE=1; HEIGHT=384; OUT=outputs/dc_acwm_robotarm_armB_stride1
     DESC="frame_stride 1 only (isolates per-frame action detail)" ;;
  C) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_armC_concat_stride1.yaml
     STRIDE=1; HEIGHT=384; OUT=outputs/dc_acwm_robotarm_armC_concat_stride1
     DESC="concat + frame_stride 1 (interaction)" ;;
  # D reuses arm C's config and differs ONLY by geometry, so it must be told
  # apart by --output-dir / -J, not by the wandb run name.
  D) CONFIG=$CFG_DIR/diffusion_dc_acwm_robotarm_armC_concat_stride1.yaml
     STRIDE=1; HEIGHT=320; OUT=outputs/dc_acwm_robotarm_armD_fullparity
     DESC="full AVID parity: concat + stride 1 + DC-native 320x512" ;;
  *) echo "Error: ARM must be 0, 0S, A, B, C or D (got '$ARM')" >&2; exit 1 ;;
esac

# 384 preserves the source 4:3 aspect (latent 48x64) and is what c3pcewxk ran, so
# arms A-C hold it fixed for single-variable comparability. 320 is DC512's native
# latent 40x64 and what the AVID reference rqp4s3gp ran — arm D only.
BATCH_SIZE="${BATCH_SIZE:-24}"

echo ">>> ARM $ARM — $DESC"
echo ">>> config=$CONFIG stride=$STRIDE height=$HEIGHT seed=$SEED batch=$BATCH_SIZE out=$OUT"

python scripts/train_avid_shortcut_metaworld.py \
    --config "$CONFIG" \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --frame-stride "$STRIDE" --target-height "$HEIGHT" --target-width 512 \
    --output-dir "$OUT" --seed "$SEED" \
    --batch-size "$BATCH_SIZE" --steps 5000000 "$@"

# READING THE RESULT — compare at a step comparable to c3pcewxk (~801) AND further
# out; c3pcewxk crept only 0.0034 -> 0.004238 over 11.7 h, so a late bloom is
# unlikely but worth confirming.
#   eval_action_effect_rel        >= 0.02  -> this arm is the cause
#   eval_action_base_null_violation ~ 0    -> REQUIRED, else the number is void
#   eval_action_effect_vs_adapter          -> the real discriminator
#                                             (ours 0.0146 vs AVID 0.422)
