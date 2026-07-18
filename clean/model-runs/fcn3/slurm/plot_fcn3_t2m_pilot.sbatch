#!/usr/bin/env bash
#SBATCH --job-name=fcn3_t2m_plot
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/t2m_plot_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fcn3/logs/t2m_plot_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
PRODUCT_ROOT="${RUN_ROOT}/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3"
PILOT_NAME="${FCN3_PILOT_NAME:-20200601_42d_ens3}"
FORECAST="${FCN3_FORECAST:-${PRODUCT_ROOT}/pilots/${PILOT_NAME}.nc}"
LOCAL_ROOT="/tmp/fcn3_t2m_plot_${SLURM_JOB_ID}"

mkdir -p "${LOCAL_ROOT}/tmp" "${LOCAL_ROOT}/xdg" "${LOCAL_ROOT}/mpl"
cleanup() { rm -rf -- "${LOCAL_ROOT:?}"; }
trap cleanup EXIT

export TMPDIR="${LOCAL_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg"
export MPLCONFIGDIR="${LOCAL_ROOT}/mpl"
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/earth2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds) pilot=${PILOT_NAME}"
"${PYTHON}" -u "${REPO}/model-runs/fcn3/scripts/plot_fcn3_vs_era5.py" \
  --forecast "${FORECAST}" \
  --output-dir "${PRODUCT_ROOT}/plots/${PILOT_NAME}"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
