#!/usr/bin/env bash
#SBATCH --job-name=fcn3_env_build
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/env_build_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/env_build_%j.err

set -euo pipefail

RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
SOURCE_ROOT="${RUN_ROOT}/_src"
BASE_ENV="/home/raj.ayush/.conda/envs/earth2"
CUDA_HOME="/usr/local/cuda-12.4"

export CUDA_HOME
export CC="${BASE_ENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${BASE_ENV}/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="${CXX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${BASE_ENV}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export PIP_CACHE_DIR="${RUN_ROOT}/_shared_cache/pip"
export TMPDIR="/tmp/fcn3_env_build_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR}/xdg"
export MPLCONFIGDIR="${TMPDIR}/mpl"

mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
cleanup() { rm -rf -- "${TMPDIR:?}"; }
trap cleanup EXIT

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version
"${CXX}" --version | head -1

"${PYTHON}" -m pip install --no-deps --no-build-isolation --force-reinstall \
  "${SOURCE_ROOT}/torch-harmonics-a632ca7"
"${PYTHON}" -m pip install \
  "hatchling==1.31.0" "moviepy==1.0.3" "numba==0.66.0"
"${PYTHON}" -m pip install --no-deps --no-build-isolation --force-reinstall \
  "${SOURCE_ROOT}/makani-b38fcb2"
"${PYTHON}" -m pip install --no-deps --no-build-isolation --force-reinstall \
  "${SOURCE_ROOT}/earth2studio-0.16.0"

"${PYTHON}" - <<'PY'
import torch
import torch_harmonics
import numba
from makani.models.model_package import load_model_package
from earth2studio.models.px import FCN3
from torch_harmonics.attention import _C as attention_cuda
from torch_harmonics.disco import _C as disco_cuda

assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_harmonics", torch_harmonics.__version__, torch_harmonics.__file__)
print("numba", numba.__version__)
print("attention_cuda", attention_cuda.__file__)
print("disco_cuda", disco_cuda.__file__)
print("FCN3", FCN3)
print("gpu", torch.cuda.get_device_name(0))
PY

"${PYTHON}" -m pip freeze --all > "${RUN_ROOT}/_envs/fcn3-prod/environment.freeze.txt"
sha256sum "${RUN_ROOT}/_envs/fcn3-prod/environment.freeze.txt"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
