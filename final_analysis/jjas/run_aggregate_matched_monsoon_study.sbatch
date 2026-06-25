#!/bin/bash
#SBATCH --job-name=jjas_match_agg
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_aggregate_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_aggregate_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
PLOTS="$ROOT/final_analysis/jjas/plots"
RUN="conda run --no-capture-output -n s2s-hind python -u"

export PYTHONUNBUFFERED=1

echo "host  : $(hostname)"
echo "start : $(date)"

$RUN "$PLOTS/aggregate_matched_monsoon_studies.py" \
  --start-year 2002 \
  --end-year 2021 \
  --base "$PLOTS/figs" \
  --out "$PLOTS/figs/matched_monsoon_study_all_years"

echo "end   : $(date)"
