#!/bin/bash
# LOCAL 3090 — overfit triangle ARM 1: replace composition, NO base-output
# input, single MetaWorld clip. No gate, no copy path: the 34M adapter must
# denoise from (x_t, t, actions) alone. Readout: loss well below the base's
# floor => the base-input concat was the trap; parks at base level => 34M
# capacity limit. Thesis-vault: 20_Tickets/experiments/exp-adapter-replace-nobase-overfit.md
#
# 41-frame windows (hits the local latent cache), generation eval off (the
# 768^2 VAE decode OOMs 24 GB). Trends transfer to the cluster's 97f runs;
# absolute numbers don't.

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

python scripts/train_wan22_i2v_metaworld_external.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_replace_nobase_overfit_metaworld.yaml \
    --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --overfit-index 0 --num-windows 1 --temporal-length 41 \
    --steps 800 --batch-size 2 --no-eval-gen \
    --wandb-run-name local-overfit-replace-nobase-41f "$@"
