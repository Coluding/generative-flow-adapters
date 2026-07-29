#!/bin/bash
#SBATCH --job-name=rt1-merge-shards
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/rt1/merge-%x-%j.out
#SBATCH --error=logs/rt1/merge-%x-%j.err

# STAGE 1b/3 (FULL DATASET) — the reduce step of the sharded conversion.
# Concatenates the 18 shard manifests into ONE metadata.pt, applies the per-dim
# action std-normalization GLOBALLY (the shards ran --no-normalize precisely so
# it could be done once here), rebases video_path onto the merged root, and
# reassigns collision-free clip_ids. See merge_rt1_shards.py for why each of
# those is load-bearing.
#
# Submit with a dependency so it only runs if EVERY shard succeeded:
#   sbatch --dependency=afterok:<array-jobid> jobs/experiments_cluster/rt1/submit_merge_rt1_shards.sh
# afterok is the guard: a dead array task means no merge, rather than a quietly
# short dataset.

set -euo pipefail
module purge; module load 2024
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
source .venv/bin/activate

ROOT="${RT1_FULL_OUT:-$HOME/scratch-shared/rt1/full}"
NSHARDS="${RT1_NSHARDS:-18}"
TOTAL="${RT1_TOTAL:-87212}"

python jobs/experiments_cluster/rt1/merge_rt1_shards.py \
    --root "$ROOT" \
    --expect-shards "$NSHARDS" \
    --expect-episodes "$TOTAL"

test -f "$ROOT/metadata.pt" || { echo "FAILED: no $ROOT/metadata.pt" >&2; exit 1; }
echo "Done. Merged manifest -> $ROOT/metadata.pt"

# Build the frame_counts.json sidecar ONCE, here, while we are still serial.
# The translator probes every mp4 with decord on first construction and caches
# the result next to metadata.pt (translators/acwm_phys.py:_probe_frame_counts).
# The sharded latent array would otherwise have all N tasks probe 87k videos
# simultaneously and race on a NON-atomic write_text of the same file — the
# corrupt-read path degrades to a silent re-probe, so it would cost wall-clock
# on every task rather than failing loudly.
echo ">>> pre-building frame_counts.json (once, serially)"
python - <<PY
from generative_flow_adapters.config import load_config
from generative_flow_adapters.data import build_rt1_clip_dataset
cfg = load_config("configs/wan22/diffusion_wan22_action_rt1.yaml")
tl = int(cfg.model.extra.get("temporal_length", 17))
_, ds = build_rt1_clip_dataset(cfg.data, default_window_width=tl, data_dir="$ROOT",
                               frame_stride=1, sampling="random", num_windows=2)
print(f"probed: {len(ds._episodes)} episodes usable, {len(ds.fixed_window_enumeration())} windows to encode")
PY
test -f "$ROOT/frame_counts.json" || echo "WARNING: frame_counts.json not written — the array will re-probe per task" >&2

echo "Next: RT1_OUT=$ROOT sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents_array.sh"
