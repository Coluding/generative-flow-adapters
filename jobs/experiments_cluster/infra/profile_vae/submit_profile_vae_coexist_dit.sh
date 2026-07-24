#!/bin/bash
#SBATCH --job-name=profile-vae-coexist-dit
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=logs/profile-vae-%x-%j.out
#SBATCH --error=logs/profile-vae-%x-%j.err

# ============================================================================
# Experiment 3/3 — THE DECISIVE ONE: encode ALONGSIDE the resident 5B.
#
# --with-dit loads the full Wan2.2-TI2V-5B DiT + adapter onto the card FIRST
# (provider forced to wan2.2_external, the real pretrained weights), then runs
# the same online-encode profile. Peak-alloc now reflects the encode transient
# ON TOP OF the resident training model — this is the number that actually
# decides whether online encoding can coexist with the 5B during training, or
# whether pre-encoding must stay. Isolated encode transient (experiments 1-2)
# fitting the card means nothing if it OOMs next to the resident model.
#
# Reads the resident-model footprint too ("resident model ... GB allocated"),
# so peak_alloc - resident ~ the added encode transient.
#
# Longer walltime + fewer batch sizes: loading the 5B costs a few minutes and
# resident memory shrinks the headroom, so keep the sweep light.
#
# Extra flags forwarded: sbatch submit_profile_vae_coexist_dit.sh --batch-size 1 2
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

CONFIG="configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml"
DATA_DIR="ds/acwm-phys/rigid_dynamics/push_block/ind_train"

python scripts/profile_vae_encode.py \
    --with-dit \
    --dataset acwm_phys --data-dir "$DATA_DIR" \
    --config "$CONFIG" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --max-area 589824 --temporal-length 65 \
    --batch-size 1 2 --num-batches 12 --warmup 2 \
    --num-workers 0 --vae-dtype bf16 \
    "$@"
