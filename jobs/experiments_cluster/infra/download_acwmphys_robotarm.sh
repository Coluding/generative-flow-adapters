#!/bin/bash
# Download the ACWM-Phys Robot Arm environment (kinematics, da=7) from HF.
# RUN ON THE LOGIN NODE (compute nodes have no internet). Larger than Push
# Cube: ~400 KB/episode mp4 (realistic 3D Isaac Sim renders vs Push Cube's
# flat vector graphics), ~1000 train + 100+100 test episodes -> roughly 1-2 GB.
#
# WHY Robot Arm: Push Cube's flat white-background visuals make the frozen base
# near-perfect (denoise loss ~0.036 vs MetaWorld ~5.4) — standard MSE is
# dominated by the trivial background and the action-dependent motion is a
# rounding error, so the adapter clones the base. Robot Arm is visually rich
# and its 7-DoF articulated motion fills the frame, so the base has a real
# residual to leave for the adapter. See thesis-vault
# 30_Knowledge/writing/ablation-axes.md (dataset axis).
#
#   bash jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh
#
# Then base-validate (does the frozen Wan base produce coherent video here?)
# before precompute/training — same check performed for Push Cube.

set -euo pipefail

DEST="..scratch-shared/acwm-phys"
mkdir -p "$DEST"

source .venv/bin/activate

huggingface-cli download t1an/ACWM-Phys \
    --repo-type dataset \
    --include "kinematics/robot_arm/*" \
    --local-dir 

echo "---- verifying ----"
for split in ind_train ind_test ood_test; do
    d="$DEST/kinematics/robot_arm/$split"
    n=$(ls "$d" 2>/dev/null | grep -c "episode_" || true)
    if test -f "$d/metadata.pt"; then
        echo "  $split: $n episode files + metadata.pt  OK"
    else
        echo "  $split: MISSING metadata.pt" >&2
    fi
done
echo "Done. data-dir for the pipeline: ds/acwm-phys/kinematics/robot_arm/<split>"
echo "Next: base-coherence probe, e.g."
echo "  IMG_SRC=ds/acwm-phys/kinematics/robot_arm/ind_train/episode_0.mp4 \\"
echo "  bash jobs/experiments/local_compare_acwm_native.sh   # (point it at the robot_arm data-dir)"
