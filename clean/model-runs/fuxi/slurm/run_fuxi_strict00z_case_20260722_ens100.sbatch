#!/bin/bash
#SBATCH --job-name=fuxi_0722_e100
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/fuxi_0722_e100_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/fuxi_0722_e100_%j.err

set -euo pipefail

REPO=/home/raj.ayush/s2s/s2s_anlysis
PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
CONFIG=${REPO}/clean/config/fuxi_strict00z_case_20260722_ens100.json
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_strict00z_case_20260722_ens100
SOURCE_INPUT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_strict00z_case_20260722_ens1_trial/inputs/annual2026/20260722/input.nc
TARGET_INPUT=${RUN_ROOT}/inputs/annual2026/20260722/input.nc
LOCAL_CACHE=/tmp/fuxi_0722_e100_${SLURM_JOB_ID}

if [[ ! -f "${SOURCE_INPUT}" ]]; then
  echo "validated deterministic input is missing: ${SOURCE_INPUT}" >&2
  exit 1
fi
mkdir -p "$(dirname "${TARGET_INPUT}")" "${LOCAL_CACHE}/xdg" "${LOCAL_CACHE}/mpl"
if [[ ! -e "${TARGET_INPUT}" ]]; then
  ln -s "${SOURCE_INPUT}" "${TARGET_INPUT}"
fi

cleanup() { rm -rf -- "${LOCAL_CACHE:?}"; }
trap cleanup EXIT

export XDG_CACHE_HOME=${LOCAL_CACHE}/xdg
export MPLCONFIGDIR=${LOCAL_CACHE}/mpl
export LD_LIBRARY_PATH=/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled

echo "host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" "${REPO}/clean/model-runs/fuxi/scripts/run_fuxi_forecast.py" \
  --date 20260722 --config "${CONFIG}" --verify-large-checkpoint
echo "end=$(date --iso-8601=seconds)"
