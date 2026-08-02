#!/bin/bash
#SBATCH --job-name=fuxi_native_rf
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-699%8
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021/logs/slurm_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021/logs/slurm_%A_%a.err

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
PYTHON="${FUXI_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021"
MANIFEST="${REPO}/clean/config/fuxi_native_reforecast_jjas_2002_2021.csv"
SCRIPT="${REPO}/clean/model-runs/fuxi/scripts/standardize_native_fuxi_reforecast.py"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"

mkdir -p "${RUN_ROOT}/shards" "${RUN_ROOT}/shards/qc" "${RUN_ROOT}/logs"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "native-reforecast task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" "${SCRIPT}" \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${RUN_ROOT}/shards" \
  --scratch-root "${SCRATCH_ROOT}"
echo "native-reforecast task=${SLURM_ARRAY_TASK_ID} end=$(date --iso-8601=seconds)"
