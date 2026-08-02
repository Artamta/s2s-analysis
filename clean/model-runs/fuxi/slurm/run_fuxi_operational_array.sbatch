#!/bin/bash
# Mixed 5/100-member official FuXi GPU inference array.

#SBATCH --job-name=fuxi-ens-a
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_inference_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_inference_%A_%a.err

set -euo pipefail
: "${FUXI_BATCH:?FUXI_BATCH must be exported by the submitter}"

REPO=/home/raj.ayush/s2s/s2s_anlysis
PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
SCRIPT=${REPO}/clean/model-runs/fuxi/scripts/fuxi_operational.py
TASK_CACHE=/tmp/fuxi_infer_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" -u "${SCRIPT}" array-item \
  --batch "${FUXI_BATCH}" --index "${SLURM_ARRAY_TASK_ID}" --phase infer
