#!/usr/bin/env bash
# launch_fcn3_weekly.sh
# ---------------------
# Runs FCN-3 weekly JFM 2026 hindcasts using the fcn3run conda env.
# First-time setup: run setup_fcn3_env.sh once, then use this.
#   bash launch_fcn3_weekly.sh

set -euo pipefail

ENV_NAME="fcn3run"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_fcn3_weekly_jfm2026.py"

echo "============================================"
echo " FCN-3 Weekly JFM2026 Hindcast Launcher"
echo "============================================"

# ── Load CUDA module ──────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module load cuda-12.9 2>/dev/null && echo "[ENV] cuda-12.9 loaded" \
        || echo "[ENV] WARNING: could not load cuda-12.9"
fi

# ── Prepend env libs so CUDA runtime is found ────────────────────────────
CONDA_ENV_DIR="$(conda run -n ${ENV_NAME} python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
[[ -n "${CONDA_ENV_DIR}" ]] && export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

# ── Sanity check ─────────────────────────────────────────────────────────
conda run -n ${ENV_NAME} python -c "
import torch
print(f'[CHECK] torch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
" 2>/dev/null || echo "[CHECK] WARNING: GPU check failed"

echo "============================================"

# ── Run ───────────────────────────────────────────────────────────────────
conda run -n ${ENV_NAME} python "${PYTHON_SCRIPT}" "$@"
