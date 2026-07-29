#!/bin/bash
# Download the OFFICIAL AVID latent-diffusion training data — RT-1
# (fractal20220817_data, Open X-Embodiment RLDS) from the PUBLIC bucket
# gs://gresearch/robotics — to cluster scratch.
#
# RUN ON THE LOGIN NODE (compute nodes have no internet). ~111 GB, so this
# takes a while and needs the disk quota. The bucket is public (Google
# Research), so anonymous read works.
#
# WHY: reproduce the AVID paper's own training data to test whether the
# action-blindness we measured on ACWM (thesis-vault
# 30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind.md)
# also appears on AVID's official data — the control that tells us whether it's
# our data/adaptation or the recipe.

set -euo pipefail

# Absolute, so the destination does not depend on the caller's cwd (a relative
# path silently lands somewhere else when this is run from anywhere but the repo
# root). Override with DEST=... to stage elsewhere.
DEST="${DEST:-$HOME/scratch-shared/oxe}"
mkdir -p "$DEST"

# Compute nodes DO have outbound internet: ~/scratch-shared/wandb holds 250
# ONLINE wandb runs (0 offline) started on gcn* nodes. An earlier version of this
# script hard-failed under Slurm claiming otherwise; that was wrong — job
# 24986795 died on 'gsutil not found' (PATH, fixed below), never reaching a
# network call, and job 24998843 then died on the bogus guard itself.
# Still prefer the login node: this is a ~111 GB single-stream transfer that
# would burn GPU-hours doing nothing but I/O.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "NOTE: running under Slurm (job $SLURM_JOB_ID on $(hostname)). This works," >&2
    echo "but a 111 GB download does not need an allocation — prefer the login node:" >&2
    echo "  nohup bash $0 > logs/download_rt1.log 2>&1 &" >&2
fi

# gsutil is the robust parallel path for ~111 GB. Installed per-user with
# 'pip install --user gsutil' (gsutil 5.37), which lands in ~/.local/bin — add
# it to PATH here so the script works under nohup/cron where .bashrc is not
# sourced.
[ -d "$HOME/.local/bin" ] && case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
esac
export PATH

# Public bucket → anonymous read needs NO credentials and NO ~/.boto at all
# (verified: 'gsutil ls gs://gresearch/robotics/fractal20220817_data/' works
# clean). Do not bother with 'gsutil config -n' — in gsutil 5.x it still
# prompts for a project id and dies with EOFError when non-interactive.
if ! command -v gsutil >/dev/null 2>&1; then
    echo "ERROR: gsutil not found. Install it (pip install --user gsutil) or load a google-cloud-sdk module, then re-run." >&2
    echo "Alternative: 'gcloud storage cp -r gs://gresearch/robotics/fractal20220817_data $DEST/'" >&2
    exit 1
fi

echo ">>> downloading RT-1 (fractal20220817_data, ~111 GB) -> $DEST/  ($(date))"
# -m parallel, -n skip-existing (so this is resumable if interrupted).
gsutil -m cp -n -r gs://gresearch/robotics/fractal20220817_data "$DEST/"

echo "---- verifying ----"
if [ -d "$DEST/fractal20220817_data" ]; then
    du -sh "$DEST/fractal20220817_data" 2>/dev/null || true
    # '|| true': head exits early and SIGPIPEs ls, which under 'set -o pipefail'
    # would abort the script right here — after a fully successful transfer.
    ls "$DEST/fractal20220817_data"/*/ 2>/dev/null | head || true
    echo "OK. Set the training job's dataset_dir to: $DEST"
    echo "   (the RTX loader reads \$dataset_dir/fractal20220817_data/<version>/)"
else
    echo "FAILED: $DEST/fractal20220817_data missing" >&2; exit 1
fi
echo "Done ($(date))."
