#!/bin/bash
# Mixed-source/date CPU staging array. FUXI_BATCH is exported by the submitter.

#SBATCH --job-name=fuxi-stage-a
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_stage_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_stage_%A_%a.err

set -euo pipefail
: "${FUXI_BATCH:?FUXI_BATCH must be exported by the submitter}"

REPO=${S2S_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}
PYTHON=${S2S_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}
SCRIPT=${REPO}/clean/model-runs/fuxi/scripts/fuxi_operational.py
TASK_CACHE=/tmp/fuxi_stage_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${TASK_CACHE}/xdg" "${TASK_CACHE}/mpl"
cleanup() { rm -rf -- "${TASK_CACHE:?}"; }
trap cleanup EXIT
export XDG_CACHE_HOME=${TASK_CACHE}/xdg
export MPLCONFIGDIR=${TASK_CACHE}/mpl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" -u "${SCRIPT}" array-item \
  --batch "${FUXI_BATCH}" --index "${SLURM_ARRAY_TASK_ID}" --phase stage
