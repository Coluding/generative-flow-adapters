#!/bin/bash
# R1 — action-sensitivity probe on a DynamiCrafter/AVID adapter checkpoint.
#
# Answers: does perturbing the action change the prediction at all?
# This gates the storyline's step 1 ("AVID works") and step 2 (planning):
# a model that converges cleanly but ignores actions cannot support a planner,
# because every candidate action sequence produces the same rollout.
#
# Eval only — no training, no wandb. Runs against an existing checkpoint.
#
# Usage:
#   jobs/experiments/eval_action_sensitivity_dc_metaworld.sh <checkpoint.pt> [extra args...]

set -euo pipefail

cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <checkpoint.pt> [extra args...]}"
shift || true

CONFIG="${CONFIG:-configs/dynamicrafter/diffusion_avid_shortcut_metaworld.yaml}"
HDF5="${HDF5:-ds/metaworld_corner2.hdf5}"
VAE="${VAE:-ckts/dynami512.ckpt}"
OUT_DIR="${OUT_DIR:-outputs/eval-action-sensitivity/dc-metaworld}"

for path in "$CKPT" "$CONFIG" "$HDF5" "$VAE"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: not found: $path" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo "=== R1 action-sensitivity probe (DynamiCrafter / AVID) ==="
echo "  config     : $CONFIG"
echo "  checkpoint : $CKPT"
echo "  dataset    : $HDF5"
echo "  out        : $OUT_DIR"
echo

# --num-batches 8 x --num-draws 4 = 32 paired samples, enough for a usable
# bootstrap CI while staying a few minutes on one GPU. Widen both if the
# verdict comes back INCONCLUSIVE.
python scripts/eval_action_sensitivity.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --dataset metaworld \
    --hdf5 "$HDF5" \
    --vae-checkpoint "$VAE" \
    --num-batches 8 \
    --batch-size 2 \
    --num-draws 4 \
    --out-dir "$OUT_DIR" \
    "$@" | tee "$OUT_DIR/report.txt"

echo
echo "report : $OUT_DIR/report.txt"
echo "json   : $OUT_DIR/action_sensitivity.json"
echo
echo "MetaWorld is the action-REDUNDANT dataset (ablation-axes Axis 1)."
echo "A flat result here is expected-ish and does NOT by itself prove the"
echo "adapter is broken — re-run on ACWM Push Cube before concluding:"
echo "  jobs/experiments/eval_action_sensitivity_dc_acwm.sh <ckpt>"
