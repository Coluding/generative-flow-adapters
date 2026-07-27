#!/bin/bash
# R1/R8 — action-sensitivity probe on a Wan2.2 adapter checkpoint, ACWM-Phys.
#
# This is the backbone the ACWM dataset-axis runs are on, so it is the probe
# that reads out the D2 headline: does the adapter use actions on an
# action-INFORMATIVE dataset, or does it base-clone there too?
#
# --max-area MUST match training, or the latent cache misses and a full VAE
# encode runs next to the resident 5B (slow at best, OOM at worst).
#
# Usage:
#   jobs/experiments/eval_action_sensitivity_wan_acwm.sh <checkpoint.pt> [extra args...]
#
#   ENV=push_block (default) | robot_arm | reacher
#   SPLIT=ind_train (default)
#   CONFIG=... MAX_AREA=589824 WAN_CKPT=ckpts/Wan2.2-TI2V-5B

set -euo pipefail

cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <checkpoint.pt> [extra args...]}"
shift || true

ENV_NAME="${ENV:-push_block}"
SPLIT="${SPLIT:-ind_train}"

case "$ENV_NAME" in
    push_block) FAMILY="rigid_body" ;;
    robot_arm|reacher) FAMILY="kinematics" ;;
    *) echo "ERROR: unknown ENV=$ENV_NAME (push_block|robot_arm|reacher)" >&2; exit 1 ;;
esac

CONFIG="${CONFIG:-configs/wan22/diffusion_wan22_avid_xattn_gatelow_metaworld.yaml}"
DATA_DIR="${DATA_DIR:-ds/acwm-phys/$FAMILY/$ENV_NAME/$SPLIT}"
WAN_CKPT="${WAN_CKPT:-ckpts/Wan2.2-TI2V-5B}"
MAX_AREA="${MAX_AREA:-589824}"
OUT_DIR="${OUT_DIR:-outputs/eval-action-sensitivity/wan-acwm-$ENV_NAME}"

for path in "$CKPT" "$CONFIG" "$DATA_DIR" "$WAN_CKPT"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: not found: $path" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo "=== R1 action-sensitivity probe (Wan2.2, ACWM $ENV_NAME) ==="
echo "  config     : $CONFIG"
echo "  checkpoint : $CKPT"
echo "  data       : $DATA_DIR"
echo "  max_area   : $MAX_AREA  (must match training)"
echo "  out        : $OUT_DIR"
echo

python scripts/eval_action_sensitivity.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --dataset acwm_phys \
    --data-dir "$DATA_DIR" \
    --wan-ckpt-dir "$WAN_CKPT" \
    --max-area "$MAX_AREA" \
    --num-batches 8 \
    --batch-size 1 \
    --num-draws 4 \
    --num-windows 16 \
    --num-workers 0 \
    --out-dir "$OUT_DIR" \
    "$@" | tee "$OUT_DIR/report.txt"

echo
echo "report : $OUT_DIR/report.txt"
echo "json   : $OUT_DIR/action_sensitivity.json"
echo
echo "NOTE: batch-size 1 means 'shuffle' borrows its donor from the NEXT batch."
echo "Keep --num-batches >= 2 or the shuffle variant degenerates to the identity."
