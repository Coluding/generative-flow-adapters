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

DEST="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"
mkdir -p "$DEST"

# gsutil is the robust parallel path for ~111 GB. If absent:
#   pip install --user gsutil   (or: module load a google-cloud-sdk module)
# Public bucket → anonymous. If gsutil prompts for auth, create an anonymous
# boto config once with:  gsutil config -n   (writes ~/.boto, no credentials)
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
    ls "$DEST/fractal20220817_data"/*/ 2>/dev/null | head
    echo "OK. Set the training job's dataset_dir to: $DEST"
    echo "   (the RTX loader reads \$dataset_dir/fractal20220817_data/<version>/)"
else
    echo "FAILED: $DEST/fractal20220817_data missing" >&2; exit 1
fi
echo "Done ($(date))."
