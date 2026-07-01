#!/bin/bash
#SBATCH --job-name=s2s_crps_conv_diag
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/common/logs/slurm/%x_%j.err

# CRPS-convention sanity check for the paper's probabilistic results.
# Scores the member-based systems (ECMWF, FuXi, NCEP) with BOTH the empirical
# ensemble CRPS and the Gaussian CRPS (from ensemble mean+spread) on identical
# fields, so we can quantify how much the Gaussian convention used for Spire
# would change a score. See analysis_pipeline.py crps_gauss_diag column.

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

# Use the analysis env explicitly (tectonic_env is LaTeX-only; base lacks deps).
PY=/home/raj.ayush/.conda/envs/s2s-hind/bin/python3
$PY --version

$PY scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --variables tp z500 \
  --models ecmwf fuxi ncep \
  --max-inits 30 \
  --weeks 1 2 3 4 5 6 \
  --workers 4 \
  --no-grid-scatter \
  --run-label crps_convention_diag

echo "END $(date -Is)"
