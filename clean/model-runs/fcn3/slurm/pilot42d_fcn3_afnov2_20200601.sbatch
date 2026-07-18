#!/usr/bin/env bash
#SBATCH --job-name=fcn3_42d_test
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/pilot42d_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/pilot42d_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/fcn3_2020_2024.json"
SCRIPTS="${REPO}/model-runs/fcn3/scripts"
PRODUCT_ROOT="${RUN_ROOT}/fcn3/fcn3_v1_afnov2_tp_t2m_00z_2020_2024_ens10"
LOCAL_ROOT="/tmp/fcn3_pilot42d_${SLURM_JOB_ID}"

mkdir -p "${LOCAL_ROOT}/tmp" "${LOCAL_ROOT}/xdg" "${LOCAL_ROOT}/mpl" "${LOCAL_ROOT}/cuda"
cleanup() { rm -rf -- "${LOCAL_ROOT:?}"; }
trap cleanup EXIT

export TMPDIR="${LOCAL_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg"
export MPLCONFIGDIR="${LOCAL_ROOT}/mpl"
export CUDA_CACHE_PATH="${LOCAL_ROOT}/cuda"
export EARTH2STUDIO_MODEL_CACHE="${RUN_ROOT}/_shared_cache/earth2studio"
export EARTH2STUDIO_DATA_CACHE="${RUN_ROOT}/_shared_cache/earth2studio"
export EARTH2STUDIO_PACKAGE_TIMEOUT=3600
export EARTH2STUDIO_DISABLE_MSC=1
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/earth2/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" - <<'PY'
import numba
import torch
import torch_harmonics
from makani.models.model_package import load_model_package
from torch_harmonics.attention import _C as attention_cuda
from torch_harmonics.disco import _C as disco_cuda

assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_harmonics", torch_harmonics.__version__)
print("numba", numba.__version__)
print("attention_cuda", attention_cuda.__file__)
print("disco_cuda", disco_cuda.__file__)
print("gpu", torch.cuda.get_device_name(0))
PY

"${PYTHON}" -u "${SCRIPTS}/stage_fcn3_ic.py" \
  --config "${CONFIG}" --init-date 2020-06-01
"${PYTHON}" -u "${SCRIPTS}/run_fcn3_afnov2.py" \
  --config "${CONFIG}" --init-date 2020-06-01 --steps 168 --members 2 \
  --test-output "${PRODUCT_ROOT}/pilots/20200601_42d_ens2.nc"

echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
