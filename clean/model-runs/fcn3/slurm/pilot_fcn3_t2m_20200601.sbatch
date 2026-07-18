#!/usr/bin/env bash
#SBATCH --job-name=fcn3_t2m_test
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/t2m_pilot_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/t2m_pilot_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/fcn3_t2m_2020_2024.json"
SCRIPTS="${REPO}/model-runs/fcn3/scripts"
PRODUCT_ROOT="${RUN_ROOT}/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3"
LOCAL_ROOT="/tmp/fcn3_t2m_pilot_${SLURM_JOB_ID}"

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
import torch
from earth2studio.models.px import FCN3

assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("model", FCN3.__name__)
PY

"${PYTHON}" -u "${SCRIPTS}/prefetch_fcn3.py" --config "${CONFIG}"
"${PYTHON}" -u "${SCRIPTS}/stage_fcn3_ic.py" \
  --config "${CONFIG}" --init-date 2020-06-01
"${PYTHON}" -u "${SCRIPTS}/run_fcn3_t2m.py" \
  --config "${CONFIG}" --init-date 2020-06-01 --steps 8 --members 2 \
  --test-output "${PRODUCT_ROOT}/pilots/20200601_2d_ens2.nc"

mkdir -p "${PRODUCT_ROOT}/provenance"
"${PYTHON}" -m pip freeze --all > "${PRODUCT_ROOT}/provenance/environment.freeze.txt"
sha256sum "${PRODUCT_ROOT}/provenance/environment.freeze.txt"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
