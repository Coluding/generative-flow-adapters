#!/bin/bash
# LOCAL 3090 — overfit triangle ARM 2: mask_mix gatelow (gate_bias 0.0,
# uncapped), NO base-output input, single MetaWorld clip. The pure
# one-variable ablation vs uxrst2k5 (only the base input removed). Watch
# adapter_gate_mean + adapter_grad_norm from step 1: gate pinning ~0.99 with
# grad-norm collapse = the gate trap alone suffices (then this arm cannot
# speak to the input question — that's what the cap09 sibling is for).
# Thesis-vault: 20_Tickets/experiments/exp-adapter-gatelow-nobase-overfit.md

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_nobase_overfit_metaworld.yaml \
    --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --overfit-index 0 --num-windows 1 --temporal-length 41 \
    --steps 800 --batch-size 2 --no-eval-gen \
    --wandb-run-name local-overfit-gatelow-nobase-41f "$@"
