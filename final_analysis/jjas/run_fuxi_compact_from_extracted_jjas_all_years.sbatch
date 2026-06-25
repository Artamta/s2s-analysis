#!/bin/bash
#SBATCH --job-name=fuxi_raw2compact
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_raw2compact_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_raw2compact_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%2

# Build compact FuXi JJAS files from the permanent full extraction tree.
# This does not delete raw extracted data.

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
JJAS="$ROOT/final_analysis/jjas"
RAW_ROOT="${RAW_ROOT:-/storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full}"
OUT_DIR="${OUT_DIR:-/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined}"
RUN="conda run --no-capture-output -n s2s-hind python -u"

YEARS=(2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
       2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host       : $(hostname)"
echo "start      : $(date)"
echo "year       : ${YEAR}"
echo "task       : ${SLURM_ARRAY_TASK_ID}"
echo "raw root   : ${RAW_ROOT}"
echo "out dir    : ${OUT_DIR}"
echo "complete JJAS raw dates:"
python - "$RAW_ROOT" "$YEAR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
year = sys.argv[2]
months = {"06", "07", "08", "09"}
count = 0
for marker in (root / year).glob("*/*/.complete"):
    parts = marker.parts
    if len(parts) >= 3 and parts[-3] in months:
        count += 1
print(count)
PY

$RUN "$JJAS/preprocess_fuxi_from_extracted.py" \
  --raw-root "$RAW_ROOT" \
  --out-dir "$OUT_DIR" \
  --year "$YEAR" \
  --months 06,07,08,09

echo "end        : $(date)"
