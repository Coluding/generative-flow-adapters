#!/bin/bash
# Build the AVID latent-diffusion python env on the CLUSTER, reproducing the
# exact pins in external_repos/avid/latent_diffusion/{pyproject.toml,poetry.lock}:
# python 3.10, torch 2.1.0+cu118, pytorch-lightning 1.9.3, tensorflow 2.15 /
# tensorflow-datasets 4.9.2, open_clip 2.22.0, transformers 4.25.1, and the
# editable local libs octo / lvdm(dynamicrafter) / avid_utils.
#
# RUN ON THE LOGIN NODE (needs internet to download wheels).
#
# WHY poetry-export + uv rather than 'poetry install':
#   - poetry.lock (277 packages) is the only real lock; the repo's uv.lock is an
#     EMPTY STUB (0 packages), so 'uv sync' installs nothing.
#   - but poetry's own env creation is broken on this machine: every 'poetry env
#     use' dies with "module '...cpython.mac_os' has no attribute
#     'CPython2macOsArmFramework'" — a stale virtualenv entry-point resolving
#     through the python3.9 user site, which survived pinning both poetry (1.8.2)
#     and virtualenv (20.26.6) and clearing ~/.local/share/virtualenv.
#   So: let poetry do the part that works (resolve the lock -> requirements.txt,
#   which needs no venv at all) and let uv create and populate the venv. The
#   pins still come from poetry.lock, so the env is the locked one.
#
# UNTESTED-AT-RUNTIME on this cluster: the torch cu118 wheel must work against
# the node's CUDA driver (cu118 is widely forward-compatible). The sanity import
# at the end reports torch.cuda.is_available(), but that is only meaningful on a
# GPU node — on the login node it prints False and that is expected.

set -euo pipefail
module purge 2>/dev/null || true
module load 2024 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

REPO="$HOME/generative-flow-adapters/external_repos/avid/latent_diffusion"
cd "$REPO"

# NOT the shared /scratch-shared/$USER/uv-cache the other job scripts use: that
# one has corrupt entries (cd-fvd 0.1.1 extracted with no setup.py; mdurl 0.1.2
# archive missing its METADATA), which fail the install even though resolution
# succeeds. A dedicated cache avoids wiping a cache other runs depend on.
# NB: the shared cache is probably worth a 'uv cache clean' at some point.
export UV_CACHE_DIR=/scratch-shared/$USER/uv-cache-avid
export UV_PYTHON_INSTALL_DIR=/scratch-shared/$USER/uv-python
# venv (home) and cache (scratch) are different filesystems, so hardlinks fail.
export UV_LINK_MODE=copy
# uv's 30s default times out extracting the big wheels straight to GPFS
# (tensorflow 2.15 is ~475 MB, torch cu118 ~2.4 GB).
export UV_HTTP_TIMEOUT=600

# There is no python3.10 on this cluster: system python is 3.9 and the newest
# module is Python/3.12.3, while pyproject pins python = "~3.10". uv's managed
# interpreters already have 3.10.19.
PY310="$UV_PYTHON_INSTALL_DIR/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10"
if [ ! -x "$PY310" ]; then
    echo ">>> installing cpython 3.10 via uv"
    uv python install 3.10
    PY310=$(uv python find 3.10)
fi

if ! command -v poetry >/dev/null 2>&1; then
    # 1.8.2 = the version that generated poetry.lock (see its header line), and
    # the last line that still ships poetry-plugin-export by default.
    echo ">>> installing poetry==1.8.2 (user site; only used to export the lock)"
    pip install --user "poetry==1.8.2"
fi

REQ="$REPO/requirements.cluster.txt"
echo ">>> exporting poetry.lock -> $REQ"
poetry export -f requirements.txt --without-hashes -o "$REQ"

echo ">>> creating .venv (python 3.10)"
uv venv --python "$PY310" "$REPO/.venv"

# EVERYTHING is installed with --no-build-isolation, which is not the usual
# advice but is required here. uv's isolated build envs inherit this
# interpreter's distutils-precedence.pth / _distutils_hack (it ships setuptools
# 80.9.0), and inside the throwaway env that hack resolves to a setuptools whose
# _distutils/dist.py does 'from packaging.utils import ...' with no packaging on
# the path. Every source build then dies with:
#   ModuleNotFoundError: No module named 'packaging.utils'
# — first on the git dep dlimp, then on the three editable local libs.
# Seeding the venv with the lock's own build deps and turning isolation off makes
# builds use packaging 24.1 (which HAS .utils) and setuptools 75.1.0.
echo ">>> seeding build deps (needed because build isolation is off, see above)"
uv pip install --python "$REPO/.venv/bin/python" \
    "setuptools==75.1.0" "wheel==0.44.0" "packaging==24.1"

# The three path/develop deps (octo, lvdm, avid_utils) are already in the export
# as '-e file://...' lines, so this single pass covers them too.
echo ">>> installing locked deps (torch 2.1+cu118 download is large)"
uv pip install --python "$REPO/.venv/bin/python" --no-build-isolation -r "$REQ"

# The ROOT project itself ('ldwma', src/ldwma — the AVID training code) is NOT in
# the export: 'poetry export' emits dependencies only, never the root package.
# So the three path deps above land fine while ldwma is silently absent, and the
# job dies ~90 s in on the GPU with 'ModuleNotFoundError: No module named ldwma'
# (job 25009171). poetry-core is the build backend and must be present in the
# venv because isolation is off.
echo ">>> installing the root package 'ldwma' (editable; NOT covered by the export)"
uv pip install --python "$REPO/.venv/bin/python" "poetry-core==2.4.1"
uv pip install --python "$REPO/.venv/bin/python" --no-deps --no-build-isolation -e "$REPO"

# PyPI's xformers 0.0.22.post7 is built against cu121; the lock's torch is cu118,
# so its C++/CUDA extensions refuse to load. That is NOT harmless here: lvdm's
# attention.py sets XFORMERS_IS_AVAILBLE from a bare 'import xformers.ops', which
# still succeeds, so the UNet dispatches to memory_efficient_attention and only
# then fails on the GPU. Pull the cu118 build from the pytorch index instead.
echo ">>> re-pinning xformers to the cu118 build (PyPI ships cu121)"
uv pip install --python "$REPO/.venv/bin/python" --no-deps --reinstall \
    --index-url https://download.pytorch.org/whl/cu118 "xformers==0.0.22.post7"

echo ">>> sanity import"
# ldwma + the config targets are imported here on purpose: a missing root package
# or a broken lvdm import must fail on the LOGIN node, not 90 s into a GPU job.
"$REPO/.venv/bin/python" -c "import torch, pytorch_lightning as pl, tensorflow_datasets as tfds, omegaconf, transformers; \
import lvdm, octo, avid_utils, ldwma, xformers; \
import ldwma.lightning.data_modules.rtx, lvdm.models.ddpm3d, lvdm.modules.networks.openaimodel3d; \
print('torch', torch.__version__, '| pl', pl.__version__, '| tfds', tfds.__version__, \
'| xformers', xformers.__version__, '| cuda', torch.cuda.is_available())"

echo ">>> venv: $REPO/.venv"
echo "Done. Train with:"
echo "  cd ~/generative-flow-adapters && mkdir -p logs/avid-rt1"
echo "  sbatch jobs/experiments_cluster/avid_official/submit_train_avid_rt1.sh"
