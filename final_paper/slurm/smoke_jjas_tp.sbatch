#!/bin/bash
#SBATCH --job-name=s2s_smoke_jjas_tp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis/final_paper
OUT_ROOT="$ROOT/outputs/s2s_paper_outputs"
LOG_DIR="$OUT_ROOT/jjas2019/07_logs/slurm"
mkdir -p "$LOG_DIR" "$OUT_ROOT/common/logs/slurm"

RUNTIME_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

cd "$ROOT"
echo "START $(date -Is)"
python --version

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate base || true
fi

python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_models \
  --variables tp \
  --models ecmwf ukmo ncep \
  --max-inits 2 \
  --weeks 1 2 \
  --workers 2 \
  --truth-source both \
  --run-label slurm_smoke_jjas_tp

python scripts/09_make_test_plots.py --season jjas2019 --run-label slurm_smoke_jjas_tp_era5truth
python scripts/09_make_test_plots.py --season jjas2019 --run-label slurm_smoke_jjas_tp_imdtruth
echo "END $(date -Is)"
