#!/usr/bin/env bash
#SBATCH --job-name=s2s_v1_build
#SBATCH --partition=gpu_prio
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/logs/build_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/logs/build_%A_%a.err

set -euo pipefail
: "${S2S_MATRIX:?Set S2S_MATRIX to the immutable JSON job matrix}"
ARCHIVE="/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1"
REPO="${S2S_CLEAN_ROOT:?S2S_CLEAN_ROOT must point to an archive runtime snapshot}"
DEPS="${ARCHIVE}/_deps"
export PYTHONPATH="${DEPS}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ECCODES_TMPDIR="/tmp/s2s_eccodes_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TMPDIR="${ECCODES_TMPDIR}"
mkdir -p "${ARCHIVE}/logs" "${ECCODES_TMPDIR}"
cleanup() { rm -rf -- "${ECCODES_TMPDIR:?}"; }
trap cleanup EXIT
python "${REPO}/studies/india_s2s_benchmark_v1/runtime_preflight.py"
python -u "${REPO}/studies/india_s2s_benchmark_v1/benchmark.py" matrix-task \
  --matrix "${S2S_MATRIX}" --task-index "${SLURM_ARRAY_TASK_ID}"
