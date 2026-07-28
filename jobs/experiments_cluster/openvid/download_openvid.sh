#!/bin/bash
# Download an OpenVid-1M SUBSET (real-world captioned video) for the D3 shortcut
# test — the in-distribution, non-robotic dataset (thesis-vault
# 20_Tickets/experiments/exp-shortcut-flow-vs-diffusion-openvid.md).
# RUN ON THE LOGIN NODE (HF download needs internet). ~a few GB for the subset.
#
# Produces ds/openvid/train/ in the ACWM mp4+metadata schema, with each clip's
# REAL caption carried per-clip (metadata.pt). Next: precompute per-clip T5.

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
source .venv/bin/activate

OUT="${OPENVID_DIR:-$HOME/scratch-shared/openvid/train}"
NUM_CLIPS="${NUM_CLIPS:-2000}"   # too many to memorize; raise for a fuller run

echo ">>> downloading OpenVid-1M subset ($NUM_CLIPS clips) -> $OUT"
python scripts/download_openvid.py \
    --out-dir "$OUT" --num-clips "$NUM_CLIPS" --part 0 \
    --max-frames 32 --height 320 --width 512

echo "---- verifying ----"
test -f "$OUT/metadata.pt" && echo "  metadata.pt OK  ($(ls "$OUT"/*.mp4 2>/dev/null | wc -l) clips)" || { echo "  MISSING metadata.pt" >&2; exit 1; }
echo "Next: sbatch jobs/experiments_cluster/openvid/submit_precompute_openvid_captions.sh"
