#!/bin/bash
#SBATCH --job-name=precompute-prompts-acwm-robotarm
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/preencode/precompute-prompts-acwm-robotarm-%x-%j.out
#SBATCH --error=logs/preencode/precompute-prompts-acwm-robotarm-%x-%j.err

# T5 prompt-context prewarm for the Wan2.2 Robot Arm run.
# Writes configs/prompts/acwm_robotarm.contexts.pt (the <stem>.contexts.pt
# sibling of the prompts YAML — that path is derived, not configurable here).
#
# WHY THIS EXISTS: submit_train_wan_robotarm.sh claims in its header that the
# contexts are "already precomputed", but only acwm_pushblock.contexts.pt was
# ever generated. Without this, the Wan robotarm job dies at startup —
# scripts/train_wan22_i2v_metaworld_external.py raises when
# text_prompts_file is set and its .contexts.pt sibling is missing.
#
# The prompts themselves are already validated (SkyReels coherence probe
# 2026-07-25 — brand words removed; see the header of the prompts YAML).
# Only the T5 encode step was outstanding.

set -euo pipefail

module purge
module load 2024

export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/generative-flow-adapters"
mkdir -p logs/preencode
source .venv/bin/activate

PROMPTS="configs/prompts/acwm_robotarm.yaml"
test -f "$PROMPTS" || { echo "Error: $PROMPTS missing" >&2; exit 1; }
test -d "ckpts/Wan2.2-TI2V-5B" || { echo "Error: ckpts/Wan2.2-TI2V-5B missing" >&2; exit 1; }

python external_repos/Wan2.2/precompute_prompt_contexts.py \
    --ckpt_dir ckpts/Wan2.2-TI2V-5B \
    --prompts_file "$PROMPTS"

echo "Done ($(date)). Contexts at ${PROMPTS%.yaml}.contexts.pt"
