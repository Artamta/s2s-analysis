#!/bin/bash
# Reusable CPU staging job for one operational FuXi source/date.
# Required exported variables: FUXI_CONFIG, FUXI_DATE, FUXI_SOURCE.

#SBATCH --job-name=fuxi-stage
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_stage_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_stage_%j.err

set -euo pipefail

: "${FUXI_CONFIG:?FUXI_CONFIG must be exported by the submitter}"
: "${FUXI_DATE:?FUXI_DATE must be exported by the submitter}"
: "${FUXI_SOURCE:?FUXI_SOURCE must be exported by the submitter}"

REPO=${S2S_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}
PYTHON=${S2S_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}
SCRIPT=${REPO}/clean/model-runs/fuxi/scripts/fuxi_operational.py
TASK_CACHE=/tmp/fuxi_stage_${FUXI_SOURCE}_${FUXI_DATE}_${SLURM_JOB_ID}

mkdir -p "${TASK_CACHE}/xdg" "${TASK_CACHE}/mpl"
cleanup() { rm -rf -- "${TASK_CACHE:?}"; }
trap cleanup EXIT
export XDG_CACHE_HOME=${TASK_CACHE}/xdg
export MPLCONFIGDIR=${TASK_CACHE}/mpl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "source=${FUXI_SOURCE} date=${FUXI_DATE} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -u "${SCRIPT}" stage --config "${FUXI_CONFIG}" --date "${FUXI_DATE}"
echo "source=${FUXI_SOURCE} date=${FUXI_DATE} end=$(date --iso-8601=seconds)"
