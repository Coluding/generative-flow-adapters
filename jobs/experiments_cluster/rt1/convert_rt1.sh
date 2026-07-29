#!/bin/bash
# STAGE 1/3 — convert RT-1 (fractal20220817_data, OXE RLDS) to our mp4+metadata
# schema so OUR Wan/DC/SkyReels adapters can train on it (the in-distribution
# ACTION test — does our adapter follow actions on real data where ACWM failed?).
# Reads the LOCAL RT-1 already downloaded for AVID ($RTX_DATA_DIR), so no GCS/
# internet is needed. Uses the AVID poetry env (tfds). Per-dim action
# std-normalization is ON by default (octo convention; stored in metadata).
#
# PREREQUISITE: RT-1 downloaded — jobs/experiments_cluster/avid_official/download_rt1.sh
# RUN ON THE LOGIN NODE (or an interactive node); a large slice takes a while.

set -euo pipefail
PYBIN="${AVID_PY:-$HOME/.cache/pypoetry/virtualenvs/ldwma-MkWu4YH_-py3.10/bin/python}"

cd "$HOME/generative-flow-adapters"
RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"        # local RT-1 RLDS (AVID download)
OUT="${RT1_OUT:-$HOME/scratch-shared/rt1/train}"
SPLIT="${RT1_SPLIT:-train[:5000]}"   # ~5k episodes: plenty to avoid the 64-clip memorization confound

test -d "$RTX_DATA_DIR/fractal20220817_data" || { echo "Error: RT-1 not at $RTX_DATA_DIR — run avid_official/download_rt1.sh" >&2; exit 1; }

echo ">>> converting RT-1 $SPLIT (local, std-normalized actions) -> $OUT"
"$PYBIN" jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py \
    --data-dir "$RTX_DATA_DIR" \
    --split "$SPLIT" \
    --out-dir "$OUT"

test -f "$OUT/metadata.pt" && echo "OK: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l) clips + metadata.pt" || { echo "FAILED" >&2; exit 1; }
echo "Next: sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents.sh"
