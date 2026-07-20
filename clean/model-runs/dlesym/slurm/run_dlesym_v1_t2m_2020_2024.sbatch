#!/usr/bin/env bash
#SBATCH --job-name=dlesym_v1_5y
#SBATCH --partition=GPU-AI
#SBATCH --array=0-516%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/dlesym/logs/v1_production_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/dlesym/logs/v1_production_%A_%a.err

set -euo pipefail
REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
PYTHON="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_envs/dlesym-prod/bin/python"
CONFIG="${REPO}/model-runs/configs/dlesym_2020_2024.json"
SCRIPTS="${REPO}/model-runs/dlesym/scripts"
LOCAL_ROOT="/tmp/dlesym_v1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LOCAL_ROOT}"/{tmp,xdg,mpl,cuda}
cleanup() { rm -rf -- "${LOCAL_ROOT:?}"; }
trap cleanup EXIT
export TMPDIR="${LOCAL_ROOT}/tmp" XDG_CACHE_HOME="${LOCAL_ROOT}/xdg" MPLCONFIGDIR="${LOCAL_ROOT}/mpl" CUDA_CACHE_PATH="${LOCAL_ROOT}/cuda"
export EARTH2STUDIO_MODEL_CACHE="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_shared_cache/earth2studio"
export EARTH2STUDIO_DATA_CACHE="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_shared_cache/earth2studio"
export EARTH2STUDIO_PACKAGE_TIMEOUT=1800 EARTH2STUDIO_DISABLE_MSC=1
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/earth2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -u "${SCRIPTS}/stage_dlesym_ic.py" --config "${CONFIG}" --product v1_t2m --index "${SLURM_ARRAY_TASK_ID}"
"${PYTHON}" -u "${SCRIPTS}/run_dlesym_42d.py" --config "${CONFIG}" --product v1_t2m --index "${SLURM_ARRAY_TASK_ID}"
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} end=$(date --iso-8601=seconds) status=passed"
