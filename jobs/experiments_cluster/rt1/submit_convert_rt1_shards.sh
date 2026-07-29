#!/bin/bash
#SBATCH --job-name=rt1-convert-shard
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/rt1/convert-shard-%a-%A.out
#SBATCH --error=logs/rt1/convert-shard-%a-%A.err

# STAGE 1/3 (FULL DATASET) — convert ALL 87 212 RT-1 episodes, sharded.
#
# convert_rt1.sh converts a slice in ONE process; at the full split that is ~21 h
# with NO resume (metadata.pt is written only at the end), so a single kill loses
# everything. This splits the TFDS split into 18 disjoint index ranges run as a
# job array — ~1.2 h wall-clock, and a dead shard costs one shard.
#
# ⚠️ Each shard runs with --no-normalize. The per-dim std-normalization is
# GLOBAL over every episode (octo convention), so it CANNOT be computed
# per-shard — normalizing each shard by its own stats would give 18 different
# action scales and silently confound the action test. merge_rt1_shards.py does
# the normalization once, over the concatenated set. Raw deltas on disk until
# then; the merged metadata.pt is the only normalized artifact.
#
# PREREQUISITE: RT-1 downloaded — jobs/experiments_cluster/avid_official/download_rt1.sh
# AFTER all shards succeed:
#   python jobs/experiments_cluster/rt1/merge_rt1_shards.py --root $HOME/scratch-shared/rt1/full

set -euo pipefail
PYBIN="${AVID_PY:-$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion/.venv/bin/python}"
test -x "$PYBIN" || { echo "Error: AVID python not at $PYBIN — run avid_official/setup_avid_env_cluster.sh (or set AVID_PY)" >&2; exit 1; }

cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"
ROOT="${RT1_FULL_OUT:-$HOME/scratch-shared/rt1/full}"   # NOT rt1/train — keeps the 5000-episode slice intact
TOTAL="${RT1_TOTAL:-87212}"                             # fractal20220817_data train split (dataset_info.json)
NSHARDS="${RT1_NSHARDS:-18}"                            # must equal the --array width above

test -d "$RTX_DATA_DIR/fractal20220817_data" || { echo "Error: RT-1 not at $RTX_DATA_DIR — run avid_official/download_rt1.sh" >&2; exit 1; }

K="${SLURM_ARRAY_TASK_ID:-0}"
PER=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
BEG=$(( K * PER ))
END=$(( BEG + PER )); [ "$END" -gt "$TOTAL" ] && END=$TOTAL
[ "$BEG" -ge "$TOTAL" ] && { echo "shard $K starts past $TOTAL — nothing to do"; exit 0; }
OUT="$ROOT/shard_$K"

echo ">>> shard $K/$NSHARDS: train[$BEG:$END] (RAW actions; merge normalizes) -> $OUT"
"$PYBIN" -u jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py \
    --data-dir "$RTX_DATA_DIR" \
    --split "train[$BEG:$END]" \
    --out-dir "$OUT" \
    --no-normalize

test -f "$OUT/metadata.pt" || { echo "FAILED: no $OUT/metadata.pt" >&2; exit 1; }
echo "OK shard $K: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l) clips"
echo "When ALL shards are COMPLETED: python jobs/experiments_cluster/rt1/merge_rt1_shards.py --root $ROOT"
