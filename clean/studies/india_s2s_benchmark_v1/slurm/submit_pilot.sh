#!/usr/bin/env bash
set -euo pipefail
REPO="${S2S_CLEAN_ROOT:-/home/raj.ayush/s2s/s2s_anlysis/clean}"
ARCHIVE="/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1"
mkdir -p "${ARCHIVE}/logs"
RUNTIME="$(S2S_SOURCE_ROOT="${REPO}" bash "${REPO}/studies/india_s2s_benchmark_v1/slurm/stage_runtime.sh")"
ARRAY_JOB="$(sbatch --parsable --export=ALL,S2S_CLEAN_ROOT="${RUNTIME}" \
  "${REPO}/studies/india_s2s_benchmark_v1/slurm/run_pilot_task.sbatch")"
FINAL_JOB="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,S2S_CLEAN_ROOT="${RUNTIME}",S2S_FINALIZE_SCOPE=pilot,S2S_EXPECTED_TASKS=25 \
  "${REPO}/studies/india_s2s_benchmark_v1/slurm/finalize.sbatch")"
echo "runtime=${RUNTIME} pilot_tasks=25 array_job=${ARRAY_JOB} finalize_job=${FINAL_JOB}"
