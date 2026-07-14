#!/bin/bash
#SBATCH --job-name=fuxi_strict
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-620%2
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/logs/slurm_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/logs/slurm_%A_%a.err

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
PYTHON="${FUXI_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}"
SCRIPT="${REPO}/clean/model-runs/fuxi/scripts/run_fuxi_forecast.py"
CONFIG="${REPO}/clean/config/fuxi_strict00z_2020_2025.json"
LOCAL_CACHE="/tmp/fuxi_strict_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${LOCAL_CACHE}/xdg" "${LOCAL_CACHE}/mpl"
cleanup() {
  rm -rf -- "${LOCAL_CACHE:?}"
}
trap cleanup EXIT

export XDG_CACHE_HOME="${LOCAL_CACHE}/xdg"
export MPLCONFIGDIR="${LOCAL_CACHE}/mpl"
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled

echo "strict00z job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" "${SCRIPT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --config "${CONFIG}"
echo "strict00z job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} end=$(date --iso-8601=seconds)"
