#!/usr/bin/env bash
#SBATCH --job-name=dlesym_v1_test
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/dlesym/logs/v1_test_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/dlesym/logs/v1_test_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
PYTHON="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_envs/dlesym-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/dlesym_2020_2024.json"
SCRIPTS="${REPO}/model-runs/dlesym/scripts"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens4"
LOCAL_ROOT="/tmp/dlesym_v1_test_${SLURM_JOB_ID}"

mkdir -p "${LOCAL_ROOT}/tmp" "${LOCAL_ROOT}/xdg" "${LOCAL_ROOT}/mpl" "${LOCAL_ROOT}/cuda"
cleanup() { rm -rf -- "${LOCAL_ROOT:?}"; }
trap cleanup EXIT

export TMPDIR="${LOCAL_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg"
export MPLCONFIGDIR="${LOCAL_ROOT}/mpl"
export CUDA_CACHE_PATH="${LOCAL_ROOT}/cuda"
export EARTH2STUDIO_MODEL_CACHE="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_shared_cache/earth2studio"
export EARTH2STUDIO_DATA_CACHE="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_shared_cache/earth2studio"
export EARTH2STUDIO_PACKAGE_TIMEOUT=1800
export EARTH2STUDIO_DISABLE_MSC=1
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/earth2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds) product=v1_t2m"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -u "${SCRIPTS}/prefetch_dlesym.py" --config "${CONFIG}" --product v1_t2m
"${PYTHON}" -u "${SCRIPTS}/stage_dlesym_ic.py" --config "${CONFIG}" --product v1_t2m --init-date 2020-06-01
"${PYTHON}" -u "${SCRIPTS}/run_dlesym_42d.py" --config "${CONFIG}" --product v1_t2m --init-date 2020-06-01
"${PYTHON}" -u "${SCRIPTS}/plot_dlesym_vs_era5.py" \
  --forecast "${RUN_ROOT}/forecasts/2020/20200601.nc" \
  --output-dir "${RUN_ROOT}/plots/20200601"
"${PYTHON}" -m pip freeze --all > "${RUN_ROOT}/provenance/environment.freeze.txt"
sha256sum "${RUN_ROOT}/provenance/environment.freeze.txt"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
