#!/bin/bash
# LOCAL 3090 — ACWM-Phys Push Cube single-episode overfit: end-to-end
# validation of the ACWM training path + first capacity reading on the new
# domain (can the capped gatelow adapter overfit one episode? watch
# denoise_adapter_delta go positive and adapter_gate_mean stay off the 0.9
# pin). Runs the intended full-run settings (gate_cap 0.9 + sigma_shift 5.0),
# so NOT sigma-comparable to the MetaWorld triangle. Single-episode overfit
# says nothing about action USAGE (memorizable without actions) — that is the
# full ind_train run's job.
#
# Prerequisite (once): bash jobs/download_acwmphys.sh  (~120 MB into
# ds/acwm-phys/). First step VAE-encodes the clip on cache miss (~4 s), then
# it's cached. The ACWM config is natively 41-frame (66-frame episodes) — no
# --temporal-length override needed.
# Thesis-vault: 50_Decisions/open/second-dataset-action-informativeness.md

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

DATA_DIR="ds/acwm-phys/rigid_dynamics/push_block/ind_train"
test -f "$DATA_DIR/metadata.pt" || { echo "Error: $DATA_DIR/metadata.pt missing — run: bash jobs/download_acwmphys.sh" >&2; exit 1; }

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
    --dataset acwm_phys --data-dir "$DATA_DIR" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --overfit-index 0 --num-windows 1 \
    --steps 800 --batch-size 2 --no-eval-gen \
    --wandb-run-name local-overfit-acwm-pushblock "$@"
