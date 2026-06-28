#!/bin/bash
#SBATCH --job-name=ncep_jjas
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/raj.ayush/All_Model_Data/ncep/logs/ncep_jjas_%j.log
#SBATCH --error=/storage/raj.ayush/All_Model_Data/ncep/logs/ncep_jjas_%j.err

set -euo pipefail

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/ncep"
PYTHON="/home/raj.ayush/.conda/envs/fuxi/bin/python"
BASE="/storage/raj.ayush/All_Model_Data/ncep"

mkdir -p "${BASE}/logs" "${BASE}/jjas2019/logs" "${BASE}/jjas2025/logs"

echo "================================================================"
echo "Job        : NCEP S2S daily JJAS downloads"
echo "Job ID     : ${SLURM_JOB_ID:-manual}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Output base: ${BASE}"
echo "Seasons    : jjas2019, jjas2025"
echo "Dates      : daily 06-01 to 09-30 for each year"
echo "Leads      : daily 1-44 days"
echo "Members    : NCEP native ensemble, cf + pf (15 perturbed)"
echo "================================================================"

echo "[1/2] JJAS 2019"
"${PYTHON}" "${SCRIPT_DIR}/download_ncep_jfm2026_daily.py" \
  --out-root "${BASE}/jjas2019" \
  --start 20190601 \
  --end 20190930

echo "[2/2] JJAS 2025"
"${PYTHON}" "${SCRIPT_DIR}/download_ncep_jfm2026_daily.py" \
  --out-root "${BASE}/jjas2025" \
  --start 20250601 \
  --end 20250930

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
