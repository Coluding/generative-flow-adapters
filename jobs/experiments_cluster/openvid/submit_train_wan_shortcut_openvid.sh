#!/bin/bash
#SBATCH --job-name=openvid-wan-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/openvid/wan-shortcut-%x-%j.out
#SBATCH --error=logs/openvid/wan-shortcut-%x-%j.err

# D3 SHORTCUT — Wan2.2 (FLOW) · OpenVid (in-distribution captioned real video).
# The clean flow arm of the flow-vs-diffusion few-step test on data the base
# generates coherently (so few-step fidelity is well-defined, no OOD drift).
# Wan working here is the headline D3 result. TI2V (frame-0 anchor + per-clip
# caption). See thesis-vault exp-shortcut-flow-vs-diffusion-openvid.md.
#
# PREREQUISITES (in order):
#   bash   jobs/experiments_cluster/openvid/download_openvid.sh                 (login node)
#   sbatch jobs/experiments_cluster/openvid/submit_precompute_openvid_captions.sh
# (the second writes configs/prompts/openvid_train.contexts.pt, required at startup).

set -euo pipefail
module purge; module load 2024
export GFA_PROFILE=0
export BATCH_SIZE=4   # shortcut = 2–3x base forwards
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/openvid
source .venv/bin/activate

OPENVID_DIR="${OPENVID_DIR:-$HOME/scratch-shared/openvid/train}"
test -f "$OPENVID_DIR/metadata.pt" || { echo "Error: $OPENVID_DIR/metadata.pt missing — run download_openvid.sh" >&2; exit 1; }
test -f "configs/prompts/openvid_train.contexts.pt" || { echo "Error: per-clip caption table missing — run submit_precompute_openvid_captions.sh" >&2; exit 1; }

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_shortcut_openvid.yaml \
    --dataset openvid \
    --data-dir "$OPENVID_DIR" \
    --eval-data-dir "$OPENVID_DIR" \
    --latent-cache-dir "$OPENVID_DIR/latents.shared" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --batch-size $BATCH_SIZE --num-windows 1 --max-area 589824 --steps 5000000 \
    --wandb-run-name wan-shortcut-openvid "$@"
