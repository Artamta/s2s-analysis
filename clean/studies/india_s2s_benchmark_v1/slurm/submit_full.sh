#!/usr/bin/env bash
set -euo pipefail
REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
ARCHIVE="/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1"
MATRIX="${ARCHIVE}/indexes/full_job_matrix.json"
export PYTHONPATH="${ARCHIVE}/_deps:${PYTHONPATH:-}"
mkdir -p "${ARCHIVE}/indexes" "${ARCHIVE}/logs"
RUNTIME="$(S2S_SOURCE_ROOT="${REPO}" bash "${REPO}/studies/india_s2s_benchmark_v1/slurm/stage_runtime.sh")"
python "${RUNTIME}/studies/india_s2s_benchmark_v1/benchmark.py" make-matrix --output "${MATRIX}"
TASKS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${MATRIX}")"
CONCURRENCY="${S2S_ARRAY_CONCURRENCY:-24}"
if (( CONCURRENCY < 1 || CONCURRENCY > 32 )); then
  echo "S2S_ARRAY_CONCURRENCY must be between 1 and 32" >&2
  exit 2
fi
ARRAY_JOB="$(sbatch --parsable --array="0-$((TASKS-1))%${CONCURRENCY}" \
  --export=ALL,S2S_CLEAN_ROOT="${RUNTIME}",S2S_MATRIX="${MATRIX}" \
  "${REPO}/studies/india_s2s_benchmark_v1/slurm/run_matrix_task.sbatch")"
FINAL_JOB="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,S2S_CLEAN_ROOT="${RUNTIME}",S2S_FINALIZE_SCOPE=full,S2S_EXPECTED_TASKS="${TASKS}" \
  "${REPO}/studies/india_s2s_benchmark_v1/slurm/finalize.sbatch")"
echo "runtime=${RUNTIME} matrix=${MATRIX} tasks=${TASKS} concurrency=${CONCURRENCY} array_job=${ARRAY_JOB} finalize_job=${FINAL_JOB}"
