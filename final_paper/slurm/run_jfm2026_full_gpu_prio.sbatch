#!/bin/bash
#SBATCH --job-name=s2s_jfm2026_full
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
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
export WORKERS="${WORKERS:-4}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
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
ulimit -a || true
scontrol show job "${SLURM_JOB_ID:-0}" || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

if [ "$SKIP_PREFLIGHT" = "1" ]; then
  echo "Skipping foundation_check and scan_forecast_inits because SKIP_PREFLIGHT=1"
else
  run_step foundation_check python scripts/00_check_foundation.py
  run_step scan_forecast_inits python scripts/05_scan_forecast_inits.py --no-sample-metadata
fi

# Full JFM2026 daily-init run with SPIRE novelty product.
# TP/T2M/Z500 use ERA5 daily truth; Z500 can use the completed monthly ARCO-ERA5 truth files.
run_step jfm_daily_spire_all_models python scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --variables tp z500 t2m \
  --models spire delysm ecmwf ukmo ncep fuxi \
  --include-spire \
  --truth-source era5 \
  --run-label "$RUN_LABEL" \
  --workers "$WORKERS"

run_step jfm_daily_spire_plots python scripts/09_make_test_plots.py \
  --season jfm2026 \
  --run-label "$RUN_LABEL" \
  --max-grid-points 20000

echo
echo "OUTPUTS:"
find "$OUT_ROOT/jfm2026/03_metrics" -maxdepth 1 -type d -name 'full_jfm2026_*' -print | sort
find "$OUT_ROOT/jfm2026/04_figures" -maxdepth 1 -type d -name 'full_jfm2026_*' -print | sort
echo "END $(date -Is)"
