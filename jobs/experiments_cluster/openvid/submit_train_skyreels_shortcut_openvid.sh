#!/bin/bash
#SBATCH --job-name=openvid-skyreels-shortcut
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=32:00:00
#SBATCH --output=logs/openvid/skyreels-shortcut-%x-%j.out
#SBATCH --error=logs/openvid/skyreels-shortcut-%x-%j.err

# D3 SHORTCUT — SkyReels-V2-1.3B (WEAK FLOW) · OpenVid. Second flow datapoint:
# does base strength matter for few-step fidelity on in-distribution captioned
# video? Flow → v_average target. See exp-shortcut-flow-vs-diffusion-openvid.md.
# CAVEAT: SkyReels denoise must thread step_level to the adapter (untested) —
# short --steps smoke first.
#
# PREREQUISITES: download_openvid.sh + submit_precompute_openvid_captions.sh
# (SkyReels uses its own T5, but the per-clip caption still flows via the
# translator's per-clip caption — confirm on the smoke).

set -euo pipefail
module purge; module load 2024
export GFA_PROFILE=0
export BATCH_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/openvid
source .venv/bin/activate

OPENVID_DIR="${OPENVID_DIR:-$HOME/scratch-shared/openvid/train}"
test -f "$OPENVID_DIR/metadata.pt" || { echo "Error: $OPENVID_DIR/metadata.pt missing — run download_openvid.sh" >&2; exit 1; }

python scripts/train_skyreels_acwm.py \
    --config configs/skyreels/diffusion_skyreels_shortcut_openvid.yaml \
    --dataset openvid \
    --data-dir "$OPENVID_DIR" \
    --latent-cache-dir "$OPENVID_DIR/latents.shared" \
    --batch-size $BATCH_SIZE --num-windows 1 --steps 5000000 \
    --wandb-run-name skyreels-shortcut-openvid "$@"
