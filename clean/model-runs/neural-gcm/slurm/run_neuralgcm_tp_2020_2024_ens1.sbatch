#!/usr/bin/env bash
#SBATCH --job-name=ngcm_tp5y
#SBATCH --partition=GPU-AI_prio
#SBATCH --array=0-516%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/neural-gcm/logs/production_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/neural-gcm/logs/production_%A_%a.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
PYTHON="/home/raj.ayush/.conda/envs/neuralgcm/bin/python"
TEMPLATE="${REPO}/model-runs/configs/neuralgcm_production_tp_2020_2024_ens1.json"
MATERIALIZE="${REPO}/model-runs/neural-gcm/scripts/materialize_neuralgcm_case.py"
CONTRACT="${REPO}/model-runs/common/neuralgcm_run_contract.py"
STAGE="${REPO}/model-runs/neural-gcm/scripts/stage_neuralgcm_run_ic.py"
RUNNER="${REPO}/model-runs/neural-gcm/scripts/run_neuralgcm_42d.py"
LOCAL_ROOT="/tmp/neuralgcm_tp5y_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
CASE_CONFIG="${LOCAL_ROOT}/case.json"

mkdir -p \
  "${LOCAL_ROOT}/tmp" \
  "${LOCAL_ROOT}/xdg" \
  "${LOCAL_ROOT}/cuda" \
  "${LOCAL_ROOT}/jax" \
  "${LOCAL_ROOT}/mpl"
cleanup() {
  rm -rf -- "${LOCAL_ROOT:?}"
}
trap cleanup EXIT

export TMPDIR="${LOCAL_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg"
export CUDA_CACHE_PATH="${LOCAL_ROOT}/cuda"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_ROOT}/jax"
export MPLCONFIGDIR="${LOCAL_ROOT}/mpl"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset LD_LIBRARY_PATH || true

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" -u "${MATERIALIZE}" \
  --template "${TEMPLATE}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --output "${CASE_CONFIG}"
"${PYTHON}" -u "${CONTRACT}" --config "${CASE_CONFIG}"
"${PYTHON}" -u "${STAGE}" --config "${CASE_CONFIG}"
"${PYTHON}" -u "${RUNNER}" --config "${CASE_CONFIG}"
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} end=$(date --iso-8601=seconds) status=passed"
