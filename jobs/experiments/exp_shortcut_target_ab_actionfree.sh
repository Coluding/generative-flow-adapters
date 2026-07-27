#!/bin/bash
# R7 — the curvature theory's falsifiable prediction, as an A/B.
#
# THE PREDICTION (thesis-vault theory/shortcut-v-averaging-bias.md):
#   Under `v_average` (Frans eq.4) the COARSE (few-step) rungs plateau and never
#   settle, because the true velocity field is not a fixed point of the averaging
#   rule — more training cannot fix it. Under `endpoint_inversion` the same
#   coarse rungs should CONVERGE.
#
# Read it off `shortcut_direction_loss/N{steps:03d}` in wandb: compare the
# coarse rungs (N001, N002) across the two arms. The fine rungs (N064, N128)
# should look the same in both — they are near the d->0 limit where the bias
# vanishes, so a difference THERE would indicate something other than the target
# rule changed.
#
# TWO DESIGN CHOICES, both deliberate:
#
#  1. BOTH ARMS RUN AT THE SAME COMMIT. The historical v_average curves predate
#     the endpoint-inversion commit (279cdb7, 2026-06-24) by ~2 months of other
#     changes; comparing against them would confound the target rule with
#     everything else. Only `--shortcut-consistency-target` differs here.
#
#  2. ACTION-FREE. The action-conditioned runs confound "does the shortcut work"
#     with "does action conditioning work" — and the action side is the one
#     currently failing. Stripping actions isolates the D3 question, which is
#     also why D3 does NOT have to wait for D2 to succeed.
#
# Usage:
#   jobs/experiments/exp_shortcut_target_ab_actionfree.sh [extra train args...]
#
#   STEPS=20000 BATCH_SIZE=6 HDF5=ds/metaworld_corner2.hdf5 ARMS="v_average endpoint_inversion"

set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/dynamicrafter/diffusion_avid_shortcut_actionfree_metaworld.yaml}"
HDF5="${HDF5:-ds/metaworld_corner2.hdf5}"
VAE="${VAE:-ckts/dynami512.ckpt}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-6}"
ARMS="${ARMS:-v_average endpoint_inversion}"
OUT_ROOT="${OUT_ROOT:-outputs/exp-shortcut-target-ab}"

for path in "$CONFIG" "$HDF5" "$VAE"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: not found: $path" >&2
        exit 1
    fi
done

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git diff --quiet 2>/dev/null; then
    echo "WARNING: working tree is dirty at $COMMIT."
    echo "         Both arms must run at the SAME code state — if you edit between"
    echo "         arms the comparison is confounded and the result is unusable."
    echo
fi

echo "=== R7 shortcut-target A/B (action-free) ==="
echo "  config : $CONFIG"
echo "  commit : $COMMIT"
echo "  arms   : $ARMS"
echo "  steps  : $STEPS   batch: $BATCH_SIZE"
echo

for ARM in $ARMS; do
    OUT_DIR="$OUT_ROOT/$ARM"
    mkdir -p "$OUT_DIR"
    echo "--- arm: $ARM -> $OUT_DIR ---"

    # Record provenance next to the run: hard rule 8 needs commit + config +
    # ckpt alongside any number that reaches the thesis, and the previous
    # v_average data is unusable precisely because this was not captured.
    cat > "$OUT_DIR/provenance.txt" <<EOF
arm                         : $ARM
commit                      : $COMMIT
config                      : $CONFIG
shortcut_consistency_target : $ARM
dataset                     : $HDF5
steps                       : $STEPS
batch_size                  : $BATCH_SIZE
EOF

    python scripts/train_avid_shortcut_metaworld.py \
        --config "$CONFIG" \
        --hdf5 "$HDF5" \
        --vae-checkpoint "$VAE" \
        --shortcut-consistency-target "$ARM" \
        --steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUT_DIR" \
        "$@"

    echo "--- arm $ARM done ---"
    echo
done

cat <<EOF

=== how to read the result ===
Compare per-rung curves across arms:  shortcut_direction_loss/N001, /N002 (coarse)
                                      shortcut_direction_loss/N064, /N128 (fine)

  PREDICTION CONFIRMED  coarse rungs plateau under v_average and converge under
                        endpoint_inversion; fine rungs match across arms.
  PREDICTION REFUTED    coarse rungs behave the same in both arms -> the plateau
                        has another cause. Still a real finding: it would mean
                        the derived bias is not the binding constraint on
                        few-step quality, and the D3 story needs rewriting.
  CONFOUNDED            fine rungs differ between arms -> something other than
                        the target rule changed. Do not report.

Then the quality half: few-step rollout PSNR/SSIM/LPIPS/FVD at s in {1/2, 1/4},
adapted vs frozen base at the same NFE budget. Quantify, do not eyeball.

Run dirs: $OUT_ROOT/<arm>/  (provenance.txt, metrics.jsonl, checkpoints/)
EOF
