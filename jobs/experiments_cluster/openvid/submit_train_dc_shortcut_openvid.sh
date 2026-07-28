#!/bin/bash
#SBATCH --job-name=openvid-dc-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/openvid/dc-shortcut-%x-%j.out
#SBATCH --error=logs/openvid/dc-shortcut-%x-%j.err

# D3 SHORTCUT — DynamiCrafter (DIFFUSION) · OpenVid. The diffusion arm of the
# flow-vs-diffusion few-step test on in-distribution captioned video, with the
# curvature-aware `endpoint_inversion` target. DEFERRED: run only after the Wan
# flow arm shows few-step quality holds. See exp-shortcut-flow-vs-diffusion-openvid.md.
# CAVEAT: DC conditions text via its OWN CLIP encoder (not the Wan positive
# table) — confirm the per-clip caption reaches DC on a short --steps smoke first.
#
# PREREQUISITES: download_openvid.sh (+ captions precompute if DC ends up using
# the same positive table; otherwise DC reads the caption from metadata via CLIP).

set -euo pipefail
module purge; module load 2024
export GFA_PROFILE=0
export BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/openvid
source .venv/bin/activate

OPENVID_DIR="${OPENVID_DIR:-$HOME/scratch-shared/openvid/train}"
test -f "$OPENVID_DIR/metadata.pt" || { echo "Error: $OPENVID_DIR/metadata.pt missing — run download_openvid.sh" >&2; exit 1; }
test -f "ckts/dynami512.ckpt" || { echo "Error: DC checkpoint ckts/dynami512.ckpt missing" >&2; exit 1; }

# OpenVid clips downloaded at 320x512 → no resize needed.
python scripts/train_avid_shortcut_metaworld.py \
    --config configs/dynamicrafter/diffusion_dc_shortcut_openvid.yaml \
    --dataset openvid \
    --data-dir "$OPENVID_DIR" \
    --frame-stride 1 --target-height 320 --target-width 512 \
    --batch-size $BATCH_SIZE --steps 5000000 "$@"
