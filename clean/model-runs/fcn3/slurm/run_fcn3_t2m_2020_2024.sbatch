#!/usr/bin/env bash
#SBATCH --job-name=fcn3_t2m_5y
#SBATCH --partition=GPU-AI
#SBATCH --array=0-516%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fcn3/logs/t2m_production_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fcn3/logs/t2m_production_%A_%a.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs"
PYTHON="${RUN_ROOT}/_envs/fcn3-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/fcn3_t2m_2020_2024.json"
SCRIPTS="${REPO}/model-runs/fcn3/scripts"
LOCAL_ROOT="/tmp/fcn3_t2m_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -u "${SCRIPTS}/stage_fcn3_ic.py" \
  --config "${CONFIG}" --index "${SLURM_ARRAY_TASK_ID}"
"${PYTHON}" -u "${SCRIPTS}/run_fcn3_t2m.py" \
  --config "${CONFIG}" --index "${SLURM_ARRAY_TASK_ID}"
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} end=$(date --iso-8601=seconds) status=passed"
