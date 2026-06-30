#!/bin/bash
#SBATCH --job-name=s2s_smoke_jfm_spire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis/final_paper
OUT_ROOT="$ROOT/outputs/s2s_paper_outputs"
LOG_DIR="$OUT_ROOT/jfm2026/07_logs/slurm"
mkdir -p "$LOG_DIR" "$OUT_ROOT/common/logs/slurm"

RUNTIME_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

cd "$ROOT"
echo "START $(date -Is)"
echo "HOST $(hostname)"
python --version

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

python scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --variables tp z500 t2m \
  --models spire fuxi delysm ecmwf ukmo ncep \
  --include-spire \
  --max-inits 2 \
  --weeks 1 \
  --workers 2 \
  --run-label slurm_smoke_jfm_daily_spire

python scripts/09_make_test_plots.py --season jfm2026 --run-label slurm_smoke_jfm_daily_spire
echo "END $(date -Is)"

