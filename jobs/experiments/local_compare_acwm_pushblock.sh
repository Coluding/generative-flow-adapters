#!/bin/bash
# LOCAL 3090 — σ-sweep + action-sensitivity probe on an ACWM Push Cube
# checkpoint (ind_test clips). THE readout for the ACWM runs: a nonzero
# shuffled/zeroed action gap would be the first action-following signal of
# the project (MetaWorld baseline: gaps < 1e-4 at every σ).
#
# Checkpoint: defaults to the latest in the config's output_dir
# (outputs/acwm-pushblock-gatelow-capshift-run/checkpoints/); override with
#   bash jobs/experiments/local_compare_acwm_pushblock.sh --checkpoint path/to/step_XXXX.pt
# Any extra args pass through (e.g. --sweep-sigmas "0.1,0.5,0.9").
#
# Cache: points at the shared training cache; an empty/absent cache is fine —
# the run encodes its ~6 probe windows on miss (~4 s each) and caches them.
# Prereq: the T5 prompt contexts for the config's text_prompts_file must
# exist (one-off: external_repos/Wan2.2/precompute_prompt_contexts.py).
# Outputs: table on stdout + sigma_sweep.{csv,png} in outputs/replace_debug/.

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

python -u scripts/generate_wan22_i2v_compare.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
    --dataset acwm_phys \
    --data-dir ds/acwm-phys/rigid_dynamics/push_block/ind_test \
    --latent-cache-dir ds/acwm-phys/rigid_dynamics/push_block/latents.shared \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --sigma-sweep --action-probe --loss-batches 0 --num-windows 8 \
    "$@"
