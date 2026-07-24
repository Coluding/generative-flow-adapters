#!/bin/bash
# One-time setup for the SkyReels-V2-I2V-1.3B capability probe.
#
# SkyReels-V2 is a Wan(2.1)-lineage DiT: flow matching (velocity) + diffusion
# forcing + Wan VAE (AutoencoderKLWan). We evaluate it as the WEAK flow-matching
# base for the base-strength axis (vs Wan2.2-5B strong, DynamiCrafter diffusion).
# See thesis-vault 30_Knowledge/writing/ablation-axes.md (Axis 5).
#
# Uses a DEDICATED venv (.venv-skyreels) so its dep pins (diffusers etc.) do NOT
# clobber the working Wan .venv. The 1.3B checkpoint auto-downloads from HF
# (Skywork/SkyReels-V2-I2V-1.3B-540P, public) on first generation.

set -euo pipefail
cd "$(dirname "$0")/../.."

REPO="external_repos/SkyReels-V2"
if [ ! -d "$REPO" ]; then
    echo ">>> cloning SkyReels-V2 -> $REPO"
    git clone https://github.com/SkyworkAI/SkyReels-V2 "$REPO"
else
    echo ">>> $REPO already present"
fi

# Python 3.10 (SkyReels' tested env): its pins torch==2.5.1 + torchvision==0.20.1
# have NO wheels for 3.13/3.12, so use uv's cpython-3.10 for this venv.
if [ ! -x ".venv-skyreels/bin/python" ] || ! .venv-skyreels/bin/python --version 2>&1 | grep -q "3.10"; then
    echo ">>> creating .venv-skyreels with Python 3.10 (via uv)"
    rm -rf .venv-skyreels
    uv venv --python 3.10 .venv-skyreels
fi
# shellcheck disable=SC1091
source .venv-skyreels/bin/activate
uv pip install --upgrade pip
echo ">>> installing SkyReels-V2 requirements (isolated 3.10 venv)"
uv pip install -r "$REPO/requirements.txt"

# flash-attn: commented out in their requirements.txt, but SkyReels' code
# imports it at runtime. Building from source is slow/fragile; use a prebuilt
# wheel (mjun0812/flash-attention-prebuild-wheels) matching this venv's
# python(3.10) + torch(2.5) + CUDA. Auto-detect CUDA from the installed torch.
echo ">>> installing prebuilt flash-attn (matched to torch/CUDA)"
CUDA_TAG="$(python -c 'import torch; print("cu"+torch.version.cuda.replace(".",""))')"
echo "    detected torch $(python -c 'import torch;print(torch.__version__)'), CUDA tag ${CUDA_TAG}"
FA_WHEEL="flash_attn-2.7.4%2B${CUDA_TAG}torch2.5-cp310-cp310-linux_x86_64.whl"
FA_URL="$(curl -s 'https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100' \
    | grep 'browser_download_url' | grep -oE 'https://[^\"]+' \
    | grep "${FA_WHEEL}" | head -1)"
if [ -n "$FA_URL" ]; then
    echo "    -> $FA_URL"
    uv pip install "$FA_URL"
else
    echo "    WARN: no prebuilt flash-attn wheel found for ${CUDA_TAG}/torch2.5/cp310."
    echo "    Browse: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases"
    echo "    (SkyReels may still run on SDPA — try the probe; only install flash-attn if it ImportErrors.)"
fi
# (frame extraction is done by the MAIN .venv in the probe script, so no
# decord needed here.)

echo ">>> done. Next: bash jobs/experiments/probe_skyreels_i2v.sh"
echo "    (first run auto-downloads the 1.3B checkpoint; ~14.7 GB VRAM at 540P with --offload)"
