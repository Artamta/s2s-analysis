#!/bin/bash
#SBATCH --job-name=ngcm_smoke
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/neural-gcm/logs/slurm_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/neural-gcm/logs/slurm_%j.err

set -euo pipefail

REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
PYTHON="/home/raj.ayush/.conda/envs/neuralgcm/bin/python"
CONFIG="${REPO}/model-runs/configs/neuralgcm_smoke_20200601.json"
CONTRACT="${REPO}/model-runs/common/pilot_contract.py"
STAGE="${REPO}/model-runs/neural-gcm/scripts/stage_neuralgcm_ic.py"
SMOKE="${REPO}/model-runs/neural-gcm/scripts/smoke_neuralgcm.py"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_smoke_20200601_ens1"
LOCAL_ROOT="/tmp/neuralgcm_smoke_${SLURM_JOB_ID}"

mkdir -p "${LOCAL_ROOT}/tmp" "${LOCAL_ROOT}/xdg" "${LOCAL_ROOT}/cuda" "${LOCAL_ROOT}/jax"
cleanup() {
  rm -rf -- "${LOCAL_ROOT:?}"
}
trap cleanup EXIT

export TMPDIR="${LOCAL_ROOT}/tmp"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg"
export CUDA_CACHE_PATH="${LOCAL_ROOT}/cuda"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_ROOT}/jax"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset LD_LIBRARY_PATH || true

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON}" --version
"${PYTHON}" -u "${CONTRACT}" --config "${CONFIG}"
"${PYTHON}" -u "${STAGE}" --config "${CONFIG}"
"${PYTHON}" -u "${SMOKE}" --config "${CONFIG}"
echo "job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds) status=passed"
