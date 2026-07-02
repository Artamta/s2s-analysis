#!/bin/bash
#SBATCH --job-name=s2s_jfm_tpz500
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/verification/common/logs/slurm/%x_%j.out
#SBATCH --error=outputs/verification/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT="${S2S_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUT_ROOT="${S2S_OUTPUT_ROOT:-$ROOT/outputs/verification}"
LOG_DIR="$OUT_ROOT/jfm2026/07_logs/slurm"
COMMON_LOG_DIR="$OUT_ROOT/common/logs/slurm"
mkdir -p "$LOG_DIR" "$COMMON_LOG_DIR"

RUNTIME_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID:-manual}_runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export WORKERS="${WORKERS:-2}"
export RUN_LABEL="${RUN_LABEL:-full_jfm2026_daily_spire}"

run_step() {
  local label="$1"
  shift
  echo
  echo "===== STEP START: ${label} $(date -Is) ====="
  echo "COMMAND: $*"
  /usr/bin/time -v "$@"
  local status=$?
  echo "===== STEP END: ${label} status=${status} $(date -Is) ====="
  return "$status"
}

cd "$ROOT"
echo "START $(date -Is)"
echo "HOST $(hostname)"
echo "JOB ${SLURM_JOB_ID:-no_slurm_id}"
echo "PARTITION ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUS ${SLURM_CPUS_PER_TASK:-unknown}"
echo "WORKERS ${WORKERS}"
echo "RUN_LABEL ${RUN_LABEL}"
python --version
df -h "$ROOT" /storage/raj.ayush || true
free -h || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

# Focused rerun: headline variables only. T2M is intentionally excluded
# until daily-mean verification truth is rebuilt; grid-cell scatter is omitted
# because the result builder uses region/area summaries.
run_step jfm_daily_spire_tp_z500 python scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --variables tp z500 \
  --models spire delysm ecmwf ukmo ncep fuxi \
  --include-spire \
  --truth-source era5 \
  --run-label "$RUN_LABEL" \
  --workers "$WORKERS" \
  --no-grid-scatter

run_step jfm_daily_spire_plots python scripts/09_make_test_plots.py \
  --season jfm2026 \
  --run-label "$RUN_LABEL" \
  --max-grid-points 20000

echo "END $(date -Is)"
