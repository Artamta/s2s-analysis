#!/bin/bash
#SBATCH --job-name=s2s_jfm_gridscatter
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis/final_paper
OUT_ROOT="$ROOT/outputs/s2s_paper_outputs"
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
# Distinct run-label: do NOT touch full_jfm2026_daily_spire (the verified
# fixed-sample TP/Z500 run the paper now cites). This run exists only to
# regenerate scatter_grid_weekly.csv (with --no-grid-scatter omitted) for the
# appendix R^2 scatter figure, which the safe rerun deliberately skipped.
export RUN_LABEL="${RUN_LABEL:-full_jfm2026_gridscatter}"

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

# Same TP/Z500 fixed-sample run as full_jfm2026_daily_spire, but WITH
# grid-cell scatter enabled, under a separate run-label so it cannot
# overwrite or interfere with the already-verified headline run.
run_step jfm_gridscatter_tp_z500 python scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --variables tp z500 \
  --models spire delysm ecmwf ukmo ncep fuxi \
  --include-spire \
  --truth-source era5 \
  --run-label "$RUN_LABEL" \
  --workers "$WORKERS"

echo "END $(date -Is)"
