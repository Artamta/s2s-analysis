#!/usr/bin/env bash
#SBATCH --job-name=afnov2_era5_test
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/afnov2_era5_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/afnov2_era5_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/fcn3_2020_2024.json"
PRODUCT_ROOT="${RUN_ROOT}/fcn3/fcn3_v1_afnov2_tp_t2m_00z_2020_2024_ens10"
INIT_DATE="${AFNOV2_TEST_DATE:-2020-06-01}"
DATE_STAMP="${INIT_DATE//-/}"
LOCAL_ROOT="/tmp/afnov2_era5_${SLURM_JOB_ID}"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -u "${REPO}/model-runs/fcn3/scripts/validate_afnov2_on_era5.py" \
  --config "${CONFIG}" --init-date "${INIT_DATE}" \
  --output "${PRODUCT_ROOT}/pilots/diagnostics/afnov2_on_era5_${DATE_STAMP}_day1.nc"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
