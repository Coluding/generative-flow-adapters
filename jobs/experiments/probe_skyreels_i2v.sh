#!/bin/bash
# Capability probe: does SkyReels-V2-I2V-1.3B produce COHERENT i2v video on our
# target domain? (Same question we answered for Wan2.2 before integrating it.)
# This is a pure base test — no adapter, no our-code — using SkyReels' own
# generate_video.py, to decide if the weak flow base is worth wiring in.
#
# Prereq: bash jobs/experiments/setup_skyreels.sh
#
# Overridable via env:
#   IMG_SRC : mp4 (frame 0 taken) or image file  [default: ACWM Push Cube ep0]
#   PROMPT  : text prompt                          [default: the validated Push Cube prompt]
#   FRAMES  : num_frames (4n+1)                    [default: 97]
#   OUT     : output dir                           [default: outputs/skyreels_probe]
#
# Readout: watch the saved mp4. Coherent scene + motion = viable weak base ->
# scope the wrapper. Noise = try a natural in-distribution frame first (set
# IMG_SRC to a real photo) to tell "model/install broken" from "domain OOD".

set -euo pipefail
cd "$(dirname "$0")/../.."

IMG_SRC="${IMG_SRC:-ds/acwm-phys/rigid_dynamics/push_block/ind_train/episode_0.mp4}"
PROMPT="${PROMPT:-Flat 2D animation on a plain white background: colored squares and a gray circle move on a white surface, minimal vector illustration style, crisp edges, smooth motion, static camera.}"
FRAMES="${FRAMES:-97}"
OUT="${OUT:-outputs/skyreels_probe}"
FRAME="$(pwd)/$OUT/cond_frame.png"
mkdir -p "$OUT"

# 1) Pull the conditioning frame with the MAIN venv (has decord + PIL).
.venv/bin/python - "$IMG_SRC" "$FRAME" <<'PY'

import sys
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
if src.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    from decord import VideoReader
    Image.fromarray(VideoReader(src)[0].asnumpy()).convert("RGB").save(out)
else:
    Image.open(src).convert("RGB").save(out)
print("conditioning frame ->", out)
PY
# 2) Generate with SkyReels' own pipeline in its isolated venv.
# shellcheck disable=SC1091
source .venv-skyreels/bin/activate
echo ">>> SkyReels-V2-I2V-1.3B: ${FRAMES} frames, 540P, shift 3.0, guide 5.0"
cd external_repos/SkyReels-V2
PYTORCH_ALLOC_CONF=expandable_segments:True python3 generate_video.py \
    --model_id Skywork/SkyReels-V2-I2V-1.3B-540P \
    --resolution 540P \
    --num_frames "$FRAMES" \
    --guidance_scale 5.0 \
    --shift 3.0 \
    --fps 24 \
    --image "$FRAME" \
    --prompt "$PROMPT" \
    --offload

echo ">>> done. SkyReels writes its mp4 under external_repos/SkyReels-V2 (see its stdout for the path);"
echo "    conditioning frame: $FRAME"
