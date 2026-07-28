#!/bin/bash
#SBATCH --job-name=avid-rt1-official
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/avid-rt1/%x-%j.out
#SBATCH --error=logs/avid-rt1/%x-%j.err

# OFFICIAL AVID reproduction on RT-1 (fractal20220817_data) — the UNMODIFIED
# avid_11M.yaml (RTXDataModule + act_cond_diffusion_11M.yaml, action_dims 7),
# only localizing the data path + shrinking batch to a single GPU + online wandb.
# Goal: does the action-blindness we saw on ACWM replicate on AVID's OWN data?
# (thesis-vault 30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind.md)
#
# PREREQUISITES (login node):
#   bash jobs/experiments_cluster/avid_official/download_rt1.sh          # ~111 GB
#   bash jobs/experiments_cluster/avid_official/setup_avid_env_cluster.sh # poetry env
#
# After training: run the action-sensitivity probe on the checkpoint (see the
# dir README — the probe needs the RTX datamodule instead of ACWM; small swap).

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

# tcmalloc cuts the RT-1 tfds dataloader's CPU-memory blow-up (AVID README).
# Point LD_PRELOAD at libtcmalloc if a jemalloc/gperftools module is available:
#   export LD_PRELOAD=/path/to/libtcmalloc.so
REPO="$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
cd "$REPO"

RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"
test -d "$RTX_DATA_DIR/fractal20220817_data" || {
    echo "Error: RT-1 data missing at $RTX_DATA_DIR/fractal20220817_data — run download_rt1.sh on the login node first" >&2
    exit 1; }
mkdir -p logs/avid-rt1

BATCH=${BATCH:-8}   # avid_11M.yaml ships batch 16 for 4xA100; 8 is a single-H100 starting point — raise if VRAM allows

# UNMODIFIED avid_11M.yaml except: dataset_dir localized, batch for 1 GPU,
# online wandb. This IS the paper recipe on the paper data.
poetry run python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=12500 --node_rank=0 \
    scripts/train_avid.py --base configs/train/avid/avid_11M.yaml --train --devices 1 \
    data.params.dataset_dir="$RTX_DATA_DIR" \
    data.params.batch_size=$BATCH \
    lightning.trainer.num_nodes=1 \
    lightning.trainer.max_steps=20000 \
    lightning.trainer.log_every_n_steps=20 \
    lightning.trainer.logger.params.offline=False \
    lightning.trainer.logger.params.project=avid-rt1-official
