#!/bin/bash
#SBATCH --job-name=fuxi_jjas_all
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_jjas_all_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_jjas_all_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4

# Compact all available FuXi-S2S JJAS hindcast archives, 2002-2021.
# One array task handles one year. Raw .7z contents are extracted only to
# node-local scratch and removed after each initialization date.

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
JJAS="$ROOT/final_analysis/jjas"
ARCHIVE_DIR=/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data
RUN="conda run --no-capture-output -n s2s-hind python -u"

YEARS=(2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
       2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)
YEAR="${YEARS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FUXI_SCRATCH="${FUXI_SCRATCH:-/tmp/${USER}/fuxi_jjas_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

INITS="$(YEAR="$YEAR" ARCHIVE_DIR="$ARCHIVE_DIR" python - <<'PY'
import os
from pathlib import Path

year = os.environ["YEAR"]
archive_dir = Path(os.environ["ARCHIVE_DIR"])
months = {"06", "07", "08", "09"}
dates = [
    p.stem for p in sorted(archive_dir.glob(f"{year}*.7z"))
    if len(p.stem) == 8 and p.stem[4:6] in months
]
print(",".join(dates))
PY
)"

if [[ -z "$INITS" ]]; then
  echo "No JJAS archives found for ${YEAR}"
  exit 1
fi

echo "host       : $(hostname)"
echo "start      : $(date)"
echo "year       : ${YEAR}"
echo "task       : ${SLURM_ARRAY_TASK_ID}"
echo "inits      : ${INITS}"
echo "scratch    : ${FUXI_SCRATCH}"
echo "init count : $(tr ',' '\n' <<< "$INITS" | wc -l)"
echo "py7zr check:"
conda run --no-capture-output -n s2s-hind python - <<'PY'
import py7zr
print("  py7zr", py7zr.__version__)
PY

$RUN "$JJAS/preprocess_fuxi.py" --init "$INITS"

echo "end        : $(date)"
