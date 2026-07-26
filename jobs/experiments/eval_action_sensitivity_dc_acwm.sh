#!/bin/bash
# R1 (ACWM arm) — action-sensitivity probe on an action-INFORMATIVE dataset.
#
# The MetaWorld companion runs on the action-redundant anchor, where a flat
# result is confounded with the data. ACWM-Phys is action-informative by
# construction (the commanded action determines the future), so this is the
# run that can actually distinguish "the adapter ignores actions" from
# "the dataset does not reward using them".
#
# Usage:
#   jobs/experiments/eval_action_sensitivity_dc_acwm.sh <checkpoint.pt> [extra args...]
#
#   ENV=push_block  (default) | robot_arm | reacher
#   SPLIT=ind_train (default)

set -euo pipefail

cd "$(dirname "$0")/../.."

CKPT="${1:?usage: $0 <checkpoint.pt> [extra args...]}"
shift || true

ENV_NAME="${ENV:-push_block}"
SPLIT="${SPLIT:-ind_train}"

case "$ENV_NAME" in
    push_block) FAMILY="rigid_body"; CONFIG_DEFAULT="configs/dynamicrafter/diffusion_dc_acwm_pushblock.yaml" ;;
    robot_arm)  FAMILY="kinematics"; CONFIG_DEFAULT="configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml" ;;
    reacher)    FAMILY="kinematics"; CONFIG_DEFAULT="configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml" ;;
    *) echo "ERROR: unknown ENV=$ENV_NAME (push_block|robot_arm|reacher)" >&2; exit 1 ;;
esac

CONFIG="${CONFIG:-$CONFIG_DEFAULT}"
DATA_DIR="${DATA_DIR:-ds/acwm-phys/$FAMILY/$ENV_NAME/$SPLIT}"
VAE="${VAE:-ckts/dynami512.ckpt}"
OUT_DIR="${OUT_DIR:-outputs/eval-action-sensitivity/dc-acwm-$ENV_NAME}"

for path in "$CKPT" "$CONFIG" "$DATA_DIR" "$VAE"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: not found: $path" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo "=== R1 action-sensitivity probe (DynamiCrafter / AVID, ACWM $ENV_NAME) ==="
echo "  config     : $CONFIG"
echo "  checkpoint : $CKPT"
echo "  data       : $DATA_DIR"
echo "  out        : $OUT_DIR"
echo

python scripts/eval_action_sensitivity.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --dataset acwm_phys \
    --data-dir "$DATA_DIR" \
    --vae-checkpoint "$VAE" \
    --num-batches 8 \
    --batch-size 2 \
    --num-draws 4 \
    --num-workers 0 \
    --out-dir "$OUT_DIR" \
    "$@" | tee "$OUT_DIR/report.txt"

echo
echo "report : $OUT_DIR/report.txt"
echo "json   : $OUT_DIR/action_sensitivity.json"
