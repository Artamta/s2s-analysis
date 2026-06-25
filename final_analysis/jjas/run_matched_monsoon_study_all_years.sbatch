#!/bin/bash
#SBATCH --job-name=jjas_match_years
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_year_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_year_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4

# Run matched-valid-window JJAS verification year-by-year after FuXi compacting.

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
PLOTS="$ROOT/final_analysis/jjas/plots"
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
echo "compact nc : $(find /storage/raj.ayush/s2s_final_data/jjas/fuxi_combined -maxdepth 1 -type f -name "${YEAR}*.nc" | wc -l)"

$RUN "$PLOTS/matched_monsoon_study.py" \
  --year "$YEAR" \
  --vars TP Z500 \
  --out "$PLOTS/figs/matched_monsoon_study_${YEAR}"

echo "end        : $(date)"
