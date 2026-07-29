#!/bin/bash
#SBATCH --job-name=rt1-latents-shard
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/rt1/latents-shard-%a-%A.out
#SBATCH --error=logs/rt1/latents-shard-%a-%A.err

# STAGE 2/3 (FULL DATASET) — pre-encode Wan2.2 VAE latents for all 87 212 RT-1
# episodes, fanned out over N GPUs.
#
# WHY AN ARRAY: measured 9 690 windows in 2 h 19 m on one H100 (job 25047953)
# => ~70 windows/min => ~40 h for the full ~169k-window pool. Slurm bills per
# GPU-hour, so 16 cards x 2.5 h costs exactly the same as 1 card x 40 h — the
# serial version buys nothing and costs two days.
#
# WHY IT IS SAFE TO RUN CONCURRENTLY: the cache key is content-derived
# (env|episode_idx|start_idx|frame_stride|txhxw -> sha1, data/latent_cache.py:27),
# so disjoint --shard-index values target disjoint files; LatentCache.put writes
# atomically and skips files that already exist. Shards may overlap or be re-run
# without corrupting anything — worst case they re-encode a window.
#
# GEOMETRY MUST MATCH the training config (diffusion_wan22_action_rt1.yaml:
# temporal_length 17, max_area 589824) and --num-windows must match the train
# job's, or every cache key differs and training silently re-encodes.
#
# PREREQUISITE: the sharded conversion AND the merge —
#   sbatch jobs/experiments_cluster/rt1/submit_convert_rt1_shards.sh
#   sbatch --dependency=afterok:<array> jobs/experiments_cluster/rt1/submit_merge_rt1_shards.sh

set -euo pipefail
module purge; module load 2024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
source .venv/bin/activate

RT1_DIR="${RT1_OUT:-$HOME/scratch-shared/rt1/full}"
CACHE="$RT1_DIR/latents.shared"
NUM_WINDOWS="${NUM_WINDOWS:-2}"      # must match the train job
NSHARDS="${LATENT_NSHARDS:-16}"      # must equal the --array width above
K="${SLURM_ARRAY_TASK_ID:-0}"

test -f "$RT1_DIR/metadata.pt" || { echo "Error: $RT1_DIR/metadata.pt missing — run the convert array + merge first" >&2; exit 1; }
test -f "$RT1_DIR/frame_counts.json" || echo "WARNING: no frame_counts.json — this task will re-probe every mp4" >&2

echo ">>> latent shard $K/$NSHARDS -> $CACHE (num-windows $NUM_WINDOWS)"
python scripts/precompute_latents.py \
    --config configs/wan22/diffusion_wan22_action_rt1.yaml \
    --dataset acwm_phys --data-dir "$RT1_DIR" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --num-windows $NUM_WINDOWS --max-area 589824 \
    --num-shards $NSHARDS --shard-index $K \
    --batch-size 4 --num-workers 8

echo "Done shard $K. Latent cache -> $CACHE"
echo "When ALL shards are COMPLETED: RT1_OUT=$RT1_DIR sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_captions.sh"
