#!/bin/bash
#SBATCH --job-name=acwm-robotarm-skyreels-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/shortcut/skyreels-shortcut-%x-%j.out
#SBATCH --error=logs/shortcut/skyreels-shortcut-%x-%j.err

# D3 SHORTCUT — SkyReels-V2-1.3B (WEAK FLOW) · ACWM Robot Arm · action-free.
# Second flow datapoint alongside Wan: does base STRENGTH matter for few-step
# shortcut fidelity? Flow → v_average consistency target (straight trajectory).
# See thesis-vault 20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md.
# NOTE: SkyReels denoise must thread step_level to the adapter at runtime
# (config-valid, untested) — confirm on a short --steps smoke first.
#
# PREREQUISITES (login node): download_acwmphys_robotarm.sh +
# submit_precompute_acwmphys_robotarm.sh (shares the robot_arm latent cache).

set -euo pipefail
module purge
module load 2024

export GFA_PROFILE=0
export BATCH_SIZE=2   # shortcut 2–3x forwards; SkyReels action run used 6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/shortcut
source .venv/bin/activate

ROOT="../scratch-shared/acwm-phys/kinematics/robot_arm"
CACHE="$ROOT/latents.shared"
test -f "$ROOT/ind_train/metadata.pt" || { echo "Error: $ROOT/ind_train/metadata.pt missing — run download_acwmphys_robotarm.sh" >&2; exit 1; }

python scripts/train_skyreels_acwm.py \
    --config configs/skyreels/diffusion_skyreels_shortcut_actionfree_robotarm.yaml \
    --dataset acwm_phys \
    --data-dir "$ROOT/ind_train" \
    --latent-cache-dir "$CACHE" \
    --batch-size $BATCH_SIZE --num-windows 8 --steps 5000000 \
    --wandb-run-name skyreels-shortcut-actionfree-robotarm "$@"
