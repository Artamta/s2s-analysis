#!/usr/bin/env bash
# setup_fcn3_env.sh
# -----------------
# Creates a clean conda env "fcn3run" with earth2studio + FCN-3 dependencies.
# Run once on gpu1:
#   bash setup_fcn3_env.sh
#
# Total time: ~10-15 min (downloads + torch_harmonics source build)

set -euo pipefail

ENV_NAME="fcn3run"
PYTHON_VER="3.11"

echo "========================================================"
echo "  FCN-3 Environment Setup  →  conda env: ${ENV_NAME}"
echo "========================================================"

# ── 0. Load CUDA module ───────────────────────────────────────────────────
module load cuda-12.9
echo "[1/7] CUDA module loaded"

# ── 1. Create fresh env ───────────────────────────────────────────────────
conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
echo "[2/7] conda env '${ENV_NAME}' created"

# ── 2. Install PyTorch 2.4.1 with CUDA 12.4 (works with CUDA 12.9 driver) ─
conda run -n "${ENV_NAME}" pip install \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124
echo "[3/7] PyTorch 2.4.1+cu124 installed"

# Sanity-check GPU is visible before continuing
conda run -n "${ENV_NAME}" python -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available — check driver/module'
print(f'    GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}')
"

# ── 3. Install torch-harmonics + remove broken CUDA .so (use Python fallback)
conda run -n "${ENV_NAME}" pip install "torch-harmonics==0.9.1" -q
# Delete compiled extensions that cause ABI mismatches; torch_harmonics
# detects their absence and falls back to pure-Python implementations.
TH_SITE="$(conda run -n "${ENV_NAME}" python -c \
    'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
find "${TH_SITE}/torch_harmonics" -name "_C*.so" -delete 2>/dev/null && \
    echo "[4/7] torch-harmonics installed (optimized kernels disabled → Python fallback)"

# ── 4. Install earth2studio core + zarr 3.1.x ────────────────────────────
conda run -n "${ENV_NAME}" pip install \
    "earth2studio>=0.15.0" \
    "zarr>=3.1.0,<4" \
    gcsfs s3fs fsspec[http]
echo "[5/7] earth2studio + zarr 3.1.x installed"

# ── 5. Install makani (FCN-3 model package) ───────────────────────────────
conda run -n "${ENV_NAME}" pip install \
    "makani @ git+https://github.com/NVIDIA/modulus-makani.git"
echo "[6/7] makani installed"

# ── 6. Patch zarr AsyncGroup import for 3.1.x ────────────────────────────
ASYNC_ZARR="$(conda run -n "${ENV_NAME}" python -c \
    'import earth2studio.io.async_zarr as m; print(m.__file__)' 2>/dev/null)"
if grep -q "from zarr import AsyncGroup" "${ASYNC_ZARR}" 2>/dev/null; then
    sed -i 's/from zarr import AsyncGroup/try:\n    from zarr import AsyncGroup\nexcept ImportError:\n    from zarr.core.group import AsyncGroup/' "${ASYNC_ZARR}"
    # clear pyc
    find "$(dirname "${ASYNC_ZARR}")/__pycache__" -name "async_zarr*" -delete 2>/dev/null || true
    echo "[6b] async_zarr.py patched for zarr 3.1.x"
fi

# ── 7. Full import check ─────────────────────────────────────────────────
echo "[7/7] Running import checks …"
conda run -n "${ENV_NAME}" python -c "
import torch
from earth2studio.models.px import FCN3
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as e2s_run
import torch_harmonics
print()
print('  ALL CHECKS PASSED')
print(f'  torch           {torch.__version__}')
print(f'  torch_harmonics {torch_harmonics.__version__}')
print(f'  CUDA            {torch.cuda.get_device_name(0)}')
" 2>/dev/null

echo ""
echo "========================================================"
echo "  Setup complete.  Run forecasts with:"
echo "    bash launch_fcn3_weekly.sh"
echo "========================================================"
