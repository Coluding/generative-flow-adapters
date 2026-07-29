#!/bin/bash
#SBATCH --job-name=avid-acwm-robotarm-official
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/avid-acwm-robotarm/%x-%j.out
#SBATCH --error=logs/avid-acwm-robotarm/%x-%j.err

# OFFICIAL AVID reference on ACWM Robot Arm, FULL DATA — the synthetic-side twin
# of submit_train_avid_rt1.sh. This is the missing cell in the action-blindness
# grid: the UNMODIFIED AVID recipe on the SAME dataset our three adapters were
# probed on (Wan ncztxyyo / DC c3pcewxk / SkyReels 8zjjn7wl, all effect_rel ~0).
#
#   AVID x RT-1        (real, in-dist)  -> effect_rel 0.0495  [93qrvr5v, clean]
#   AVID x Push Cube   (synth, OOD)     -> effect_rel 0.0015  [423pjv8y, CONFOUNDED:
#                                          max_clips=64, 187 epochs = memorization]
#   AVID x Robot Arm   (synth, OOD)     -> THIS RUN, the clean synthetic control
#
# Holds the recipe fixed and varies ONLY the data, so it tests the data/OOD
# hypothesis directly:
#   blind here     => the ACWM blindness is the data, recipe exonerated
#   follows here   => data hypothesis dead; look at our implementation instead
#
# HISTORY — a previous attempt FAILED: wandb `iybbufly` (2026-07-28 10:45,
# lukas-station, local), 0 logged steps. Its log
# (outputs/avid_acwm_robotarm_11M/loginfo/log_0:2026-07-28T10-45-03.txt) ends
# cleanly after "@Training [51] Paramters for Image_proj_model." with NO
# traceback, because stdout/stderr were never redirected to a file. Cause still
# unknown. Ruled out by inspection: action_dims is correctly 7 (robot_arm actions
# verified [128, 7]); the short-video guard IS present (acwm.py:127-132, the fix
# from bug-data-acwm-robotarm-short-videos); num_workers defaults to 0 so the
# decord fork deadlock does not apply. Under sbatch the --error file WILL capture
# the traceback, so run SMOKE=1 first and read it if this dies again.
#
# SUBMIT FROM THE REPO ROOT:
#   cd ~/generative-flow-adapters
#   mkdir -p logs/avid-acwm-robotarm   # Slurm will NOT create --output's dir;
#                                      # without it the job dies with no log at all
#   SMOKE=1 sbatch jobs/experiments_cluster/avid_official/submit_train_avid_acwm_robotarm.sh
#   # then, once the smoke job logs steps and writes a checkpoint:
#   sbatch jobs/experiments_cluster/avid_official/submit_train_avid_acwm_robotarm.sh
#
# PREREQUISITES (login node, once):
#   bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh   # -> $HOME/scratch-shared/acwm-phys
#   bash jobs/experiments_cluster/avid_official/setup_avid_env_cluster.sh
#   # external_repos/ is gitignored -> rsync the avid repo to the cluster yourself.

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
cd "$REPO"

ACWM_DATA_DIR="${ACWM_DATA_DIR:-$HOME/scratch-shared/acwm-phys/kinematics/robot_arm/ind_train}"
test -d "$ACWM_DATA_DIR" || {
    echo "Error: ACWM robot_arm data missing at $ACWM_DATA_DIR — run download_acwmphys_robotarm.sh on the login node first" >&2
    exit 1; }
test -f "$ACWM_DATA_DIR/metadata.pt" || {
    echo "Error: no metadata.pt in $ACWM_DATA_DIR — the datamodule needs it for actions + episode lengths" >&2
    exit 1; }

# Base DynamiCrafter weights. The vendored configs point at the author's laptop
# path (/home/lukas/projects/...), which does not exist here.
DC_CKPT="${DC_CKPT:-$HOME/generative-flow-adapters/ckts/dynami512.ckpt}"
test -f "$DC_CKPT" || { echo "Error: DynamiCrafter checkpoint missing at $DC_CKPT" >&2; exit 1; }

# Run outputs go to scratch, not $HOME — avid_11M_acwm_robotarm.yaml hardcodes a
# `logdir:` under the author's local checkout, and checkpoints are large.
LOGDIR="${LOGDIR:-/scratch-shared/$USER/avid-acwm-robotarm}"
mkdir -p "$LOGDIR"

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
# command line silently does nothing. They must be rewritten on disk. These
# generated copies keep the fix inside this tracked script; external_repos/ is
# gitignored, so editing the vendored yamls in place would be invisible.
# NOTE the action config differs from RT-1's: _acwm_robotarm keeps action_dims 7
# (robot_arm's 7-dim action — same as RT-1's default, verified against
# metadata.pt: actions [128, 7]).
GEN="$REPO/configs/train/_cluster_generated"
mkdir -p "$GEN"
for cfg in dynamicrafter_512.yaml act_cond_diffusion_11M_acwm_robotarm.yaml; do
    sed -e "s|/home/lukas/projects/generative-flow-adapters/ckts/dynami512.ckpt|$DC_CKPT|g" \
        -e "s|/host_home/wandb/|$LOGDIR/wandb|g" \
        -e "s|/host_home/avid|$LOGDIR|g" \
        "$REPO/configs/train/$cfg" > "$GEN/$cfg"
done

BATCH=${BATCH:-8}          # config ships 2 (single-GPU smoke); 8 is a single-H100 start
MAX_STEPS=${MAX_STEPS:-15000}   # matches the RT-1 reference ckpt (epoch=14-step=15000)
                                # so the two arms are compared at equal training
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    # Validate the pipeline end-to-end cheaply BEFORE committing 48 h — this run
    # has already died once at exactly this stage with no traceback.
    echo ">>> SMOKE MODE: 64 clips, 60 steps. Confirm steps log + a ckpt lands, then rerun without SMOKE=1."
    MAX_STEPS=60
    EXTRA+=( data.params.max_clips=64 )
fi

# GEOMETRY FIX — the config ships target_height: 384, but the action config
# declares image_size: [40, 64] = 320x512 / 8, and the DynamiCrafter base is
# natively 320x512. Push Cube (the arm that DID train, 423pjv8y) uses 320.
# 384 is used only by ddpm3d.py:1119,1148 for sampling shape, so it does not
# break the training forward — it breaks the ImageLogger at batch_frequency 1000
# and puts the frozen base off-distribution. Forced to 320 for base parity.
HEIGHT=${HEIGHT:-320}

# UNMODIFIED avid_11M_acwm_robotarm.yaml except the localizations below — the
# real AVIDAdapter composition (base*mask + adapter*(1-mask), learnt_mask,
# init_mask_bias 0.0 balanced), full data (max_clips: null in the config).
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=12501 --node_rank=0 \
    scripts/train_avid.py --base configs/train/avid/avid_11M_acwm_robotarm.yaml --train --devices 1 \
    base_config_file="$GEN/dynamicrafter_512.yaml" \
    action_config_file="$GEN/act_cond_diffusion_11M_acwm_robotarm.yaml" \
    logdir="$LOGDIR" \
    data.params.data_dir="$ACWM_DATA_DIR" \
    data.params.batch_size=$BATCH \
    data.params.target_height=$HEIGHT \
    lightning.trainer.num_nodes=1 \
    lightning.trainer.max_steps=$MAX_STEPS \
    lightning.trainer.log_every_n_steps=20 \
    lightning.trainer.num_sanity_val_steps=0 \
    lightning.trainer.limit_val_batches=5 \
    lightning.trainer.logger.params.save_dir="$LOGDIR/wandb" \
    lightning.trainer.logger.params.offline=False \
    lightning.trainer.logger.params.entity=null \
    lightning.trainer.logger.params.log_model=False \
    lightning.trainer.logger.params.project=avid-acwm-robotarm-official \
    "${EXTRA[@]}"

# Index build note: with max_clips: null the datamodule probes the true frame
# count of ALL 2002 robot_arm episodes in setup() (acwm.py:127, sequential,
# num_workers=0) and caches them to a sidecar next to metadata.pt. First launch
# therefore sits silent for a while before step 0 — that is the probe, not a
# hang. Subsequent launches read the cache.
