#!/bin/bash
# LOCAL 3090 — ACWM-Phys Push Cube single-episode overfit: end-to-end
# validation of the ACWM training path + first capacity reading on the new
# domain (can the capped gatelow adapter overfit one episode? watch
# denoise_adapter_delta go positive and adapter_gate_mean stay off the 0.9
# pin). Runs the intended full-run settings (gate_cap 0.9 + sigma_shift 5.0,
# NATIVE letterboxed 1280x704 geometry), so NOT sigma-comparable to the
# MetaWorld triangle. Single-episode overfit says nothing about action USAGE
# (memorizable without actions) — that is the full ind_train run's job.
#
# Prerequisites (once):
#   1. bash jobs/experiments_cluster/infra/download_acwmphys.sh                      (~120 MB into ds/acwm-phys/)
#   2. bash jobs/experiments/local_compare_acwm_native.sh  (CPU-encodes the
#      17f native window into the shared native17 cache — training encodes
#      on GPU with the 5B resident and would OOM on a cache miss at this
#      resolution, so the window MUST be cached first)
#
# --temporal-length 41 (not the config's 41): at native 1280x704 the 3090
# fits neither the 41f encode nor the 41f DiT forward — 17f is the local
# regime; the cluster runs the full 41f. Trends transfer, numbers don't.
# Thesis-vault: 50_Decisions/open/second-dataset-action-informativeness.md

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

DATA_DIR="ds/acwm-phys/rigid_dynamics/push_block/ind_train"
CACHE="ds/acwm-phys/rigid_dynamics/push_block/square768.latents"
test -f "$DATA_DIR/metadata.pt" || { echo "Error: $DATA_DIR/metadata.pt missing — run: bash jobs/experiments_cluster/infra/download_acwmphys.sh" >&2; exit 1; }
test -d "$CACHE" || { echo "Error: $CACHE missing — run: bash jobs/experiments/local_compare_acwm_native.sh once first (CPU-encodes the window)" >&2; exit 1; }

PYTORCH_ALLOC_CONF=expandable_segments:True python -u scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
    --dataset acwm_phys --data-dir "$DATA_DIR" \
    --latent-cache-dir "$CACHE" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --temporal-length 41 \
    --overfit-index 0 --num-windows 1 \
    --steps 800 --batch-size 2 --no-eval-gen \
    --wandb-run-name local-overfit-acwm-pushblock-768 "$@"
