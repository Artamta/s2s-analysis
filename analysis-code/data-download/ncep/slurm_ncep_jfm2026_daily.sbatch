#!/bin/bash
#SBATCH --job-name=ncep_jfm_daily
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/raj.ayush/All_Model_Data/ncep/jfm2026/logs/ncep_daily_%j.log
#SBATCH --error=/storage/raj.ayush/All_Model_Data/ncep/jfm2026/logs/ncep_daily_%j.err

set -euo pipefail

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/ncep"
PYTHON="/home/raj.ayush/.conda/envs/fuxi/bin/python"
OUT_ROOT="/storage/raj.ayush/All_Model_Data/ncep/jfm2026"

mkdir -p "${OUT_ROOT}/logs"

echo "================================================================"
echo "Job        : NCEP S2S daily JFM 2026 download"
echo "Job ID     : ${SLURM_JOB_ID:-manual}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Output     : ${OUT_ROOT}"
echo "Dates      : 90 daily inits, 2026-01-01 to 2026-03-31"
echo "Leads      : daily 1-44 days"
echo "Members    : NCEP native ensemble, cf + pf (15 perturbed)"
echo "================================================================"

"${PYTHON}" "${SCRIPT_DIR}/download_ncep_jfm2026_daily.py" --start 20260101 --end 20260331

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
