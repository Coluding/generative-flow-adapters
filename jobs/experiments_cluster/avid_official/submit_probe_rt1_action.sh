#!/bin/bash
#SBATCH --job-name=avid-rt1-action-probe
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/avid-rt1/probe-%x-%j.out
#SBATCH --error=logs/avid-rt1/probe-%x-%j.err

# Action-sensitivity probe on the OFFICIAL AVID RT-1 checkpoint — does the
# ORIGINAL AVID recipe follow actions on its OWN full real-world data? The clean
# control for our ACWM action-blindness (the 64-clip smoke was confounded; this
# is a real, fully-trained run). Reproduces our eval_action_effect_rel metric on
# AVIDAdapter.apply_model, with the frozen base as null control.
# Result comparable to our runs: Wan 0.0056 / DC 0.0034 / SkyReels 0.0013 (ACWM).
#
# PREREQUISITE: the RT-1 run finished (checkpoints under
# outputs/avid_rt1_official_11M/checkpoints) AND the updated probe is on the
# cluster (rsync — external_repos is gitignored; see the dir README).

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
mkdir -p "$HOME/generative-flow-adapters/logs/avid-rt1"

RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"

# avid_11M.yaml has name=avid_rt1_11M + logdir=/host_home/avid (a container path).
# Checkpoints land at <resolved logdir>/avid_rt1_11M/checkpoints — usually the
# repo's outputs/ fallback. Override with AVID_RT1_CKPT_DIR if it's elsewhere.
CKPT_DIR="${AVID_RT1_CKPT_DIR:-outputs/avid_rt1_11M/checkpoints}"
if ! ls "$CKPT_DIR"/epoch=*-step=*.ckpt >/dev/null 2>&1; then
    echo "No ckpt in $CKPT_DIR. Find it and set AVID_RT1_CKPT_DIR:" >&2
    find "$HOME/generative-flow-adapters/external_repos/avid" /host_home/avid -name "epoch=*-step=*.ckpt" 2>/dev/null | grep -i rt1 | head >&2 || true
    exit 1
fi

poetry run python scripts/probe_action_sensitivity.py \
    --config configs/train/avid/avid_11M.yaml \
    --ckpt-dir "$CKPT_DIR" \
    --dataset-dir "$RTX_DATA_DIR" \
    --num-batches 8 --noise-draws 3
