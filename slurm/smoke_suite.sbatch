#!/bin/bash
#SBATCH --job-name=s2s_smoke_suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/verification/common/logs/slurm/%x_%j.out
#SBATCH --error=outputs/verification/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT="${S2S_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUT_ROOT="${S2S_OUTPUT_ROOT:-$ROOT/outputs/verification}"
LOG_DIR="$OUT_ROOT/common/logs/slurm"
mkdir -p "$LOG_DIR" "$OUT_ROOT/jfm2026/07_logs/slurm" "$OUT_ROOT/jjas2019/07_logs/slurm"

RUNTIME_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

cd "$ROOT"
echo "START $(date -Is)"
echo "HOST $(hostname)"
echo "JOB ${SLURM_JOB_ID:-no_slurm_id}"
python --version

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

python scripts/00_check_foundation.py
python scripts/05_scan_forecast_inits.py --no-sample-metadata
python scripts/08_run_smoke_metric_cases.py --case all

python scripts/09_make_test_plots.py --season jjas2019 --run-label test_tp
python scripts/09_make_test_plots.py --season jjas2019 --run-label test_z500_delysm_ukmo
python scripts/09_make_test_plots.py --season jjas2019 --run-label test_tp_common17_fuxi
python scripts/09_make_test_plots.py --season jjas2019 --run-label test_z500_common17_fuxi
python scripts/09_make_test_plots.py --season jfm2026 --run-label test_tp_spire
python scripts/09_make_test_plots.py --season jfm2026 --run-label test_z500_spire_delysm_ukmo
python scripts/09_make_test_plots.py --season jfm2026 --run-label test_t2m_spire_delysm_fuxi

echo "END $(date -Is)"
