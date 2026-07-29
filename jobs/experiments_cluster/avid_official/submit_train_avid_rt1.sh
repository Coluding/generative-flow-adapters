#!/bin/bash
#SBATCH --job-name=avid-rt1-official
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/avid-rt1/%x-%j.out
#SBATCH --error=logs/avid-rt1/%x-%j.err

# OFFICIAL AVID reproduction on RT-1 (fractal20220817_data) — the UNMODIFIED
# avid_11M.yaml (RTXDataModule + act_cond_diffusion_11M.yaml, action_dims 7),
# only localizing paths + shrinking batch to a single GPU + online wandb.
# Goal: does the action-blindness we saw on ACWM replicate on AVID's OWN data?
# (thesis-vault 30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind.md)
#
# SUBMIT FROM THE REPO ROOT:
#   cd ~/generative-flow-adapters
#   mkdir -p logs/avid-rt1     # Slurm will NOT create --output's dir; without
#                              # it the job dies instantly with no log at all
#   sbatch jobs/experiments_cluster/avid_official/submit_train_avid_rt1.sh
#
# PREREQUISITES (login node, once):
#   bash jobs/experiments_cluster/avid_official/download_rt1.sh           # ~111 GB (DONE)
#   bash jobs/experiments_cluster/avid_official/setup_avid_env_cluster.sh # .venv (py3.10)
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

# Base DynamiCrafter weights. The vendored configs point at the AUTHOR's laptop
# path (/home/lukas/projects/...), which does not exist here.
DC_CKPT="${DC_CKPT:-$HOME/generative-flow-adapters/ckts/dynami512.ckpt}"
test -f "$DC_CKPT" || {
    echo "Error: DynamiCrafter checkpoint missing at $DC_CKPT" >&2; exit 1; }

# Run outputs (workdir/checkpoints/wandb) go to scratch, not $HOME — the vendored
# configs use /host_home/*, a path from the authors' Docker mount that does not
# exist on this cluster, and checkpoints are large.
LOGDIR="${LOGDIR:-/scratch-shared/$USER/avid-rt1}"
mkdir -p "$LOGDIR"

# Built by setup_avid_env_cluster.sh (uv venv on python 3.10). Sourced directly
# rather than via 'poetry run' — see that script for why poetry's env layer is
# unusable here.
VENV="${AVID_VENV:-$REPO/.venv}"
test -x "$VENV/bin/python" || {
    echo "Error: AVID venv not found at $VENV — run setup_avid_env_cluster.sh on the login node first" >&2
    exit 1; }
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The checkpoint path lives in the two SIDE configs (base_config_file /
# action_config_file), which train_avid.py loads with a separate OmegaConf.load
# AFTER the CLI dotlist is merged — so 'model.pretrained_checkpoint=...' on the
# command line silently does nothing. They have to be rewritten on disk. These
# generated copies keep the fix inside this tracked script; external_repos/ is
# gitignored, so editing the vendored yamls in place would be invisible.
GEN="$REPO/configs/train/_cluster_generated"
mkdir -p "$GEN"
for cfg in dynamicrafter_512.yaml act_cond_diffusion_11M.yaml; do
    sed -e "s|/home/lukas/projects/generative-flow-adapters/ckts/dynami512.ckpt|$DC_CKPT|g" \
        -e "s|/host_home/wandb/|$LOGDIR/wandb|g" \
        -e "s|/host_home/avid|$LOGDIR|g" \
        "$REPO/configs/train/$cfg" > "$GEN/$cfg"
done

BATCH=${BATCH:-8}   # avid_11M.yaml ships batch 16 for 4xA100; 8 is a single-H100 starting point — raise if VRAM allows

# UNMODIFIED avid_11M.yaml except the localizations below. This IS the paper
# recipe on the paper data.
#   entity=null       — the config hardcodes 'causica' (the authors' wandb org,
#                       which we cannot write to); null = our default entity.
#   log_model=False   — upstream 'all' + save_top_k -1 would upload EVERY
#                       checkpoint to wandb.
# (Compute nodes DO have outbound internet — 250 online wandb runs from gcn*
# nodes in ~/scratch-shared/wandb — so offline=False is fine.)
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=12500 --node_rank=0 \
    scripts/train_avid.py --base configs/train/avid/avid_11M.yaml --train --devices 1 \
    base_config_file="$GEN/dynamicrafter_512.yaml" \
    action_config_file="$GEN/act_cond_diffusion_11M.yaml" \
    logdir="$LOGDIR" \
    data.params.dataset_dir="$RTX_DATA_DIR" \
    data.params.batch_size=$BATCH \
    lightning.trainer.num_nodes=1 \
    lightning.trainer.max_steps=20000 \
    lightning.trainer.log_every_n_steps=20 \
    lightning.trainer.logger.params.save_dir="$LOGDIR/wandb" \
    lightning.trainer.logger.params.offline=False \
    lightning.trainer.logger.params.entity=null \
    lightning.trainer.logger.params.log_model=False \
    lightning.trainer.logger.params.project=avid-rt1-official
