#!/usr/bin/env bash
#SBATCH --job-name=s2s_v1_finalize
#SBATCH --partition=gpu_prio
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/logs/finalize_%j.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1/logs/finalize_%j.err

set -euo pipefail
ARCHIVE="/storage/raj.ayush/s2s_final_data/final_iteration/standardized/india_s2s_benchmark_v1"
REPO="${S2S_CLEAN_ROOT:?S2S_CLEAN_ROOT must point to an archive runtime snapshot}"
export PYTHONPATH="${ARCHIVE}/_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PILOT_FLAG=()
if [[ "${S2S_FINALIZE_SCOPE:-full}" == "pilot" ]]; then PILOT_FLAG=(--pilot); fi
EXPECTED_FLAG=()
if [[ -n "${S2S_EXPECTED_TASKS:-}" ]]; then EXPECTED_FLAG=(--expected-tasks "${S2S_EXPECTED_TASKS}"); fi
python "${REPO}/studies/india_s2s_benchmark_v1/runtime_preflight.py"
python -u "${REPO}/studies/india_s2s_benchmark_v1/benchmark.py" finalize \
  "${PILOT_FLAG[@]}" "${EXPECTED_FLAG[@]}"
