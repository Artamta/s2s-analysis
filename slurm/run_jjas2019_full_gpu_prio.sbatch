#!/bin/bash
#SBATCH --job-name=s2s_jjas2019_full
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --output=outputs/verification/common/logs/slurm/%x_%j.out
#SBATCH --error=outputs/verification/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT="${S2S_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUT_ROOT="${S2S_OUTPUT_ROOT:-$ROOT/outputs/verification}"
LOG_DIR="$OUT_ROOT/jjas2019/07_logs/slurm"
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
export WORKERS="${WORKERS:-16}"

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
python --version
df -h "$ROOT" /storage/raj.ayush || true
free -h || true
ulimit -a || true
scontrol show job "${SLURM_JOB_ID:-0}" || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

run_step foundation_check python scripts/00_check_foundation.py
run_step scan_forecast_inits python scripts/05_scan_forecast_inits.py

# Strict all-model comparison: current fair JJAS2019 set with all available models.
# After integrating the staged FuXi Monday dates this is 33 dates because the
# main DLESyM root still lacks usable 20190603 and 20190617 stores.
# TP uses IMD truth here; Z500/T2M use ERA5 where available.
run_step jjas_common33_all_models python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name all_usable_models \
  --variables tp z500 t2m \
  --models delysm ecmwf ukmo ncep fuxi \
  --truth-source imd \
  --run-label full_jjas2019_common33_all_models_imd \
  --workers "$WORKERS"

run_step jjas_common33_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_common33_all_models_imd \
  --max-grid-points 20000

# Operational 35-init benchmark plus FuXi. FuXi has TP/Z500 for all 35
# Monday/Thursday dates; FuXi T2M is only present in the staged Monday files.
run_step jjas_operational35_plus_fuxi_tp_both_truth python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_plus_fuxi \
  --variables tp \
  --models ecmwf ukmo ncep fuxi \
  --truth-source both \
  --run-label full_jjas2019_operational35_plus_fuxi_tp \
  --workers "$WORKERS"

run_step jjas_operational35_plus_fuxi_z500 python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_plus_fuxi \
  --variables z500 \
  --models ecmwf ukmo ncep fuxi \
  --truth-source era5 \
  --run-label full_jjas2019_operational35_plus_fuxi_z500 \
  --workers "$WORKERS"

run_step jjas_operational35_plus_fuxi_tp_era5_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_operational35_plus_fuxi_tp_era5truth \
  --max-grid-points 20000

run_step jjas_operational35_plus_fuxi_tp_imd_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_operational35_plus_fuxi_tp_imdtruth \
  --max-grid-points 20000

run_step jjas_operational35_plus_fuxi_z500_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_operational35_plus_fuxi_z500 \
  --max-grid-points 20000

# Operational 35-init rainfall benchmark without FuXi for baseline comparison.
run_step jjas_operational35_tp_both_truth python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_models \
  --variables tp \
  --models ecmwf ukmo ncep \
  --truth-source both \
  --run-label full_jjas2019_operational35_tp \
  --workers "$WORKERS"

run_step jjas_operational35_era5_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_operational35_tp_era5truth \
  --max-grid-points 20000

run_step jjas_operational35_imd_plots python scripts/09_make_test_plots.py \
  --season jjas2019 \
  --run-label full_jjas2019_operational35_tp_imdtruth \
  --max-grid-points 20000

echo
echo "OUTPUTS:"
find "$OUT_ROOT/jjas2019/03_metrics" -maxdepth 1 -type d -name 'full_jjas2019_*' -print | sort
find "$OUT_ROOT/jjas2019/04_figures" -maxdepth 1 -type d -name 'full_jjas2019_*' -print | sort
echo "END $(date -Is)"
