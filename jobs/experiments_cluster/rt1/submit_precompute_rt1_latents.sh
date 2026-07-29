#!/bin/bash
#SBATCH --job-name=rt1-precompute-latents
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/rt1/precompute-%x-%j.out
#SBATCH --error=logs/rt1/precompute-%x-%j.err

# STAGE 2/3 — pre-encode Wan2.2 VAE latents for the converted RT-1 (the
# "pre-encode the rt1 data" step). Uses --dataset acwm_phys because RT-1's
# mp4+metadata is schema-identical and latents are text-free (the per-clip
# caption is irrelevant to the VAE). Geometry MUST match the training config
# (diffusion_wan22_action_rt1.yaml: temporal_length 17, max_area 589824) and
# --num-windows must match the train job's, or the cache keys won't hit.
#
# PREREQUISITE: bash jobs/experiments_cluster/rt1/convert_rt1.sh

set -euo pipefail
module purge; module load 2024
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
source .venv/bin/activate

RT1_DIR="${RT1_OUT:-$HOME/scratch-shared/rt1/train}"
CACHE="$RT1_DIR/latents.shared"
NUM_WINDOWS="${NUM_WINDOWS:-2}"   # RT-1 episodes are short; must match the train job
test -f "$RT1_DIR/metadata.pt" || { echo "Error: $RT1_DIR/metadata.pt missing — run convert_rt1.sh" >&2; exit 1; }

python scripts/precompute_latents.py \
    --config configs/wan22/diffusion_wan22_action_rt1.yaml \
    --dataset acwm_phys --data-dir "$RT1_DIR" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --num-windows $NUM_WINDOWS --max-area 589824 \
    --batch-size 4 --num-workers 8

echo "Done. Latent cache -> $CACHE (num-windows $NUM_WINDOWS)"
echo "Next: sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_captions.sh"
