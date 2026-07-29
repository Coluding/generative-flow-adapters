#!/bin/bash
#SBATCH --job-name=rt1-convert
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/rt1/convert-%x-%j.out
#SBATCH --error=logs/rt1/convert-%x-%j.err

# STAGE 1/3 — convert RT-1 (fractal20220817_data, OXE RLDS) to our mp4+metadata
# schema so OUR Wan/DC/SkyReels adapters can train on it (the in-distribution
# ACTION test — does our adapter follow actions on real data where ACWM failed?).
# Reads the LOCAL RT-1 already downloaded for AVID ($RTX_DATA_DIR), so no GCS/
# internet is needed. Uses the AVID venv (tfds). Per-dim action
# std-normalization is ON by default (octo convention; stored in metadata).
#
# PREREQUISITE: RT-1 downloaded — jobs/experiments_cluster/avid_official/download_rt1.sh
#
# ⚠️ SUBMIT IT — `sbatch jobs/experiments_cluster/rt1/convert_rt1.sh`. This is
# CPU-only (tfds decode + mp4 encode, no GPU), so it goes to `genoa`, NOT to a
# GPU partition. It used to say "RUN ON THE LOGIN NODE": at the default 5000
# episodes the login-node watchdog SIGTERMs it (2026-07-29, killed at 3421/5000
# after ~63 min — exit 143, log just says "Terminated"). `bash <this>` is still
# fine for a small --split smoke slice. There is NO resume: metadata.pt is
# written only at the end, and the std-normalization needs every episode, so a
# kill at 99% loses the whole run.

set -euo pipefail
# The AVID env lives in the repo, NOT under poetry — poetry's env layer is
# unusable on this cluster (see avid_official/setup_avid_env_cluster.sh), so the
# old $HOME/.cache/pypoetry/... default never existed here. Same venv as
# avid_official/submit_probe_rt1_action.sh; keep the two in sync.
PYBIN="${AVID_PY:-$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion/.venv/bin/python}"
test -x "$PYBIN" || { echo "Error: AVID python not at $PYBIN — run avid_official/setup_avid_env_cluster.sh on the login node (or set AVID_PY)" >&2; exit 1; }

cd "$HOME/generative-flow-adapters"
mkdir -p logs/rt1
RTX_DATA_DIR="${RTX_DATA_DIR:-$HOME/scratch-shared/oxe}"        # local RT-1 RLDS (AVID download)
OUT="${RT1_OUT:-$HOME/scratch-shared/rt1/train}"
SPLIT="${RT1_SPLIT:-train[:5000]}"   # ~5k episodes: plenty to avoid the 64-clip memorization confound

test -d "$RTX_DATA_DIR/fractal20220817_data" || { echo "Error: RT-1 not at $RTX_DATA_DIR — run avid_official/download_rt1.sh" >&2; exit 1; }

echo ">>> converting RT-1 $SPLIT (local, std-normalized actions) -> $OUT"
# -u: the per-episode progress print is block-buffered when stdout is a file,
# so without it the log stays empty and a stalled run is indistinguishable from
# a fast one.
"$PYBIN" -u jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py \
    --data-dir "$RTX_DATA_DIR" \
    --split "$SPLIT" \
    --out-dir "$OUT"

test -f "$OUT/metadata.pt" && echo "OK: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l) clips + metadata.pt" || { echo "FAILED" >&2; exit 1; }
echo "Next: sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents.sh"
