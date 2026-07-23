#!/bin/bash
# BISECTION PROBE: our WanTI2VVideoModel wrapper vs upstream, identical
# inputs (letterboxed ACWM frame, table context, native 1280x704, 17f).
# See the .py header for the decision rule.
# Output: outputs/replace_debug/acwm_wrapper_probe.mp4 + _strip.png
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
PYTORCH_ALLOC_CONF=expandable_segments:True python -u jobs/experiments/local_probe_acwm_wrapper.py "$@"
