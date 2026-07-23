#!/bin/bash
# LOCAL 3090 — overfit triangle ARM 3: mask_mix gatelow with gate_cap 0.9,
# NO base-output input, single MetaWorld clip. The gate is clamped so pred
# keeps >= 10% of the gradient no matter what — the guaranteed-interpretable
# arm: whatever pred does (learns / stalls at base) is a clean answer to the
# input-conditioning question even if the saturation pull exists.
# Thesis-vault: 20_Tickets/experiments/exp-adapter-gatelow-nobase-overfit.md

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_nobase_gatecap_overfit_metaworld.yaml \
    --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --overfit-index 0 --num-windows 1 --temporal-length 41 \
    --steps 800 --batch-size 2 --no-eval-gen \
    --wandb-run-name local-overfit-gatelow-nobase-cap09-41f "$@"
