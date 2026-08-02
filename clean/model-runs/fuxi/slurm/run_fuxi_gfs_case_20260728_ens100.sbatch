#!/bin/bash
#SBATCH --job-name=fuxi_gfs_0728_e100
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/fuxi_gfs_0728_e100_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/fuxi_gfs_0728_e100_%j.err

set -euo pipefail

REPO=/home/raj.ayush/s2s/s2s_anlysis
PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
CONFIG=${REPO}/clean/config/fuxi_gfs_case_20260728_ens100.json
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_gfs_proxy_case_20260728_ens100
SOURCE_INPUT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_gfs_proxy_case_20260728_ens24/inputs/annual2026/20260728/input.nc
TARGET_INPUT=${RUN_ROOT}/inputs/annual2026/20260728/input.nc
LOCAL_CACHE=/tmp/fuxi_gfs_0728_e100_${SLURM_JOB_ID}

test -f "${SOURCE_INPUT}"
mkdir -p "${LOCAL_CACHE}/xdg" "${LOCAL_CACHE}/mpl" "$(dirname "${TARGET_INPUT}")"
cleanup() { rm -rf -- "${LOCAL_CACHE:?}"; }
trap cleanup EXIT
ln -sfn "${SOURCE_INPUT}" "${TARGET_INPUT}"

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
  --date 20260728 --config "${CONFIG}" --verify-large-checkpoint
echo "end=$(date --iso-8601=seconds)"
