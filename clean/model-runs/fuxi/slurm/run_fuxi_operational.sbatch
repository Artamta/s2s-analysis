#!/bin/bash
# Reusable official 100-member FuXi GPU inference job.
# Required exported variables: FUXI_CONFIG, FUXI_DATE, FUXI_SOURCE.

#SBATCH --job-name=fuxi-e100
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_inference_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_inference_%j.err

set -euo pipefail

: "${FUXI_CONFIG:?FUXI_CONFIG must be exported by the submitter}"
: "${FUXI_DATE:?FUXI_DATE must be exported by the submitter}"
: "${FUXI_SOURCE:?FUXI_SOURCE must be exported by the submitter}"

REPO=${S2S_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}
PYTHON=${S2S_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}
RUNNER=${REPO}/clean/model-runs/fuxi/scripts/run_fuxi_forecast.py
TASK_CACHE=/tmp/fuxi_infer_${FUXI_SOURCE}_${FUXI_DATE}_${SLURM_JOB_ID}

mkdir -p "${TASK_CACHE}/xdg" "${TASK_CACHE}/mpl"
cleanup() { rm -rf -- "${TASK_CACHE:?}"; }
trap cleanup EXIT
export XDG_CACHE_HOME=${TASK_CACHE}/xdg
export MPLCONFIGDIR=${TASK_CACHE}/mpl
export LD_LIBRARY_PATH=/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled

echo "source=${FUXI_SOURCE} date=${FUXI_DATE} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" -u "${RUNNER}" \
  --date "${FUXI_DATE}" \
  --config "${FUXI_CONFIG}" \
  --verify-large-checkpoint
echo "source=${FUXI_SOURCE} date=${FUXI_DATE} end=$(date --iso-8601=seconds)"
