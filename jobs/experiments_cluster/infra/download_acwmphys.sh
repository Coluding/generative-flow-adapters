#!/bin/bash
# Download the ACWM-Phys push_block environment (the paper's Push Cube,
# da=2) from HF to scratch. RUN ON THE LOGIN NODE — compute nodes have no
# internet. Small: ~70 KB/episode mp4, ~120 MB total for the env.
#
#   bash jobs/download_acwmphys.sh
#
# Then submit the latent precompute:  sbatch jobs/submit_precompute_acwmphys.sh

set -euo pipefail

DEST="$HOME/projects/generative-flow-adapters/ds/acwm-phys"
mkdir -p "$DEST"

cd "$HOME/projects/generative-flow-adapters"
source .venv/bin/activate

huggingface-cli download t1an/ACWM-Phys \
    --repo-type dataset \
    --include "rigid_dynamics/push_block/*" \
    --local-dir "$DEST"

echo "---- verifying ----"
for split in ind_train ind_test ood_test; do
    d="$DEST/rigid_dynamics/push_block/$split"
    n=$(ls "$d" 2>/dev/null | grep -c "episode_" || true)
    if test -f "$d/metadata.pt"; then
        echo "  $split: $n episode files + metadata.pt  OK"
    else
        echo "  $split: MISSING metadata.pt" >&2
    fi
done
echo "Done. Next: sbatch jobs/submit_precompute_acwmphys.sh"
