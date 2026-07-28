#!/bin/bash
#SBATCH --job-name=openvid-caption-precompute
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/openvid/precompute-%x-%j.out
#SBATCH --error=logs/openvid/precompute-%x-%j.err

# Per-clip caption T5 precompute for OpenVid: umT5-encode every clip's caption
# into a positive table keyed by clip_id (PromptContextProvider positive mode),
# so the frozen Wan/SkyReels base is conditioned on THIS clip's caption at train
# time with no T5 in the loop. Writes configs/prompts/openvid_train.contexts.pt.
# GPU because the umT5 encoder is heavy.
#
# PREREQUISITE (login node): bash jobs/experiments_cluster/openvid/download_openvid.sh

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/openvid
source .venv/bin/activate

OPENVID_DIR="${OPENVID_DIR:-$HOME/scratch-shared/openvid/train}"
test -f "$OPENVID_DIR/metadata.pt" || { echo "Error: $OPENVID_DIR/metadata.pt missing — run download_openvid.sh first" >&2; exit 1; }

python scripts/precompute_clip_captions.py \
    --data-dir "$OPENVID_DIR" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --out configs/prompts/openvid_train.contexts.pt

echo "Done. Per-clip caption table -> configs/prompts/openvid_train.contexts.pt"
echo "Next: sbatch jobs/experiments_cluster/openvid/submit_train_wan_shortcut_openvid.sh"
