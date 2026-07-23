#!/bin/bash
# LOCAL 3090 — ACWM Push Cube base-vs-adapted rollout at the FIXED native
# geometry (letterboxed 1280x704, max_area 901120). This is the presentation
# that made the frozen 5B produce coherent in-domain video via the upstream
# pipeline (2026-07-23 finding: square 768x768 -> pure noise; native
# letterboxed -> works). The run validates that OUR pipeline reproduces it.
#
# Everything happens on the fly in one process: on a latent-cache miss the
# 17-frame window is encoded on CPU (~5-15 min, once — the VAE shim moves
# encode+decode to CPU because at native res neither fits next to the 5B on
# 24 GB), cached, and the GPU solver runs the two 50-step rollouts. Repeat
# runs hit the cache and go straight to the rollouts.
#
# Default: --random-init (no trained ACWM checkpoint yet — the BASE panel is
# the subject). Once a checkpoint exists:
#   bash jobs/experiments/local_compare_acwm_native.sh --checkpoint outputs/acwm-pushblock-gatelow-capshift-run/checkpoints/step_XXXX.pt
# Other args pass through. Success signature: x_std DECAYING through the
# seam table (0.96 -> ~0.7, unlike the square-768 failure's flat ~1.0) and a
# base panel showing the white Push Cube scene, not static.
# Outputs: outputs/replace_debug/clip0_s50_g5.0_{gt_base_adapted.mp4,strip.png}

set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

DATA_DIR="ds/acwm-phys/rigid_dynamics/push_block/ind_train"
CACHE="ds/acwm-phys/rigid_dynamics/push_block/square768.latents"
test -f "$DATA_DIR/metadata.pt" || { echo "Error: $DATA_DIR/metadata.pt missing — run: bash jobs/experiments_cluster/infra/download_acwmphys.sh" >&2; exit 1; }

EXTRA="--random-init"
case " $* " in *" --checkpoint "*) EXTRA="";; esac

PYTORCH_ALLOC_CONF=expandable_segments:True python -u scripts/generate_wan22_i2v_compare.py \
    --config configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_pushblock.yaml \
    $EXTRA \
    --dataset acwm_phys --data-dir "$DATA_DIR" \
    --latent-cache-dir "$CACHE" \
    --max-area 589824 --num-windows 1 \
    --loss-batches 1 --num-steps 50 \
    "$@"
