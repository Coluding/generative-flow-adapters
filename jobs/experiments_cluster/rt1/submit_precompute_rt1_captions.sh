#!/bin/bash
#SBATCH --job-name=rt1-precompute-captions
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rt1/captions-%x-%j.out
#SBATCH --error=logs/rt1/captions-%x-%j.err

# STAGE 3/3 — per-clip caption T5 table for RT-1. RT-1 ships a
# natural_language_instruction per episode ("pick up the coke can"); the
# converter stored it as the per-clip caption/clip_id. umT5-encode each into a
# positive table keyed by clip_id (PromptContextProvider positive mode), so the
# frozen Wan base is conditioned on THIS clip's instruction — the real text
# signal, not a generic prompt. Writes configs/prompts/rt1_captions.contexts.pt.
#
# PREREQUISITE: bash jobs/experiments_cluster/rt1/convert_rt1.sh

set -euo pipefail
module purge; module load 2024
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
source .venv/bin/activate

RT1_DIR="${RT1_OUT:-$HOME/scratch-shared/rt1/train}"
test -f "$RT1_DIR/metadata.pt" || { echo "Error: $RT1_DIR/metadata.pt missing — run convert_rt1.sh" >&2; exit 1; }

python scripts/precompute_clip_captions.py \
    --data-dir "$RT1_DIR" \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B \
    --out configs/prompts/rt1_captions.contexts.pt

echo "Done. Per-clip caption table -> configs/prompts/rt1_captions.contexts.pt"
echo "Next: sbatch jobs/experiments_cluster/rt1/submit_train_wan_rt1_action.sh"
