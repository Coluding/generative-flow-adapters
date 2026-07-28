#!/bin/bash
# Build the AVID latent-diffusion python env on the CLUSTER, reproducing the
# exact pins in external_repos/avid/latent_diffusion/{pyproject.toml,poetry.lock}
# via poetry: python 3.10, torch 2.1.0+cu118, pytorch-lightning 1.9.3,
# tensorflow/tensorflow-datasets, open_clip 2.22.0, transformers 4.25.1, and the
# editable local libs octo / lvdm(dynamicrafter) / avid_utils.
#
# RUN ON THE LOGIN NODE (needs internet to resolve/download wheels).
#
# UNTESTED on this cluster — the torch cu118 wheel must be compatible with the
# node's CUDA driver (cu118 is widely forward-compatible, but confirm). If the
# install or a later import fails, iterate here; this is the one piece I could
# not validate remotely. poetry.lock makes the resolution deterministic, so a
# failure is almost certainly a wheel/CUDA/native-lib mismatch, not versions.

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true

cd "$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"

if ! command -v poetry >/dev/null 2>&1; then
    echo ">>> installing poetry (user site)"
    pip install --user poetry
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">>> poetry install (this reproduces the locked env; ~torch 2.1+cu118 download is large)"
poetry env use 3.10 || poetry env use python3.10
poetry install --no-interaction

echo ">>> sanity import"
poetry run python -c "import torch, pytorch_lightning as pl, tensorflow_datasets as tfds, omegaconf, transformers; \
print('torch', torch.__version__, '| pl', pl.__version__, '| tfds', tfds.__version__, '| cuda', torch.cuda.is_available())"

echo ">>> env path (record this):"; poetry env info --path
echo "Done. Train with: sbatch jobs/experiments_cluster/avid_official/submit_train_avid_rt1.sh"
