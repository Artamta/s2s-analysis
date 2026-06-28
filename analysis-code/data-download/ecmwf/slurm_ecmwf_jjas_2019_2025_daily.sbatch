#!/bin/bash
#SBATCH --job-name=ecmwf_jjas
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/raj.ayush/All_Model_Data/ecmwf/logs/ecmwf_jjas_%j.log
#SBATCH --error=/storage/raj.ayush/All_Model_Data/ecmwf/logs/ecmwf_jjas_%j.err

set -euo pipefail

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/ecmwf"
PYTHON="/home/raj.ayush/.conda/envs/fuxi/bin/python"
BASE="/storage/raj.ayush/All_Model_Data/ecmwf"

mkdir -p "${BASE}/logs" "${BASE}/jjas2019/logs" "${BASE}/jjas2025/logs"

if ! "${PYTHON}" -c "import cdsapi" >/dev/null 2>&1; then
  echo "cdsapi missing in ${PYTHON}; installing user package cdsapi"
  "${PYTHON}" -m pip install --user cdsapi
fi

echo "================================================================"
echo "Job        : ECMWF operational daily JJAS downloads"
echo "Job ID     : ${SLURM_JOB_ID:-manual}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Output base: ${BASE}"
echo "Seasons    : jjas2019, jjas2025"
echo "Dates      : 2019 Mon/Thu only, 2025 daily"
echo "Leads      : daily 1-46 days"
echo "Grid       : global 1.5deg"
echo "API        : CDS/ECDS s2s-forecasts"
echo "Members    : ECMWF ENS cf + pf"
echo "================================================================"

echo "[1/2] JJAS 2019"
"${PYTHON}" "${SCRIPT_DIR}/download_ecmwf_daily_operational.py" \
  --out-root "${BASE}/jjas2019" \
  --start 20190601 \
  --end 20190930 \
  --weekdays mon-thu

echo "[2/2] JJAS 2025"
"${PYTHON}" "${SCRIPT_DIR}/download_ecmwf_daily_operational.py" \
  --out-root "${BASE}/jjas2025" \
  --start 20250601 \
  --end 20250930 \
  --weekdays all

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
