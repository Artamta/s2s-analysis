#!/bin/bash
#SBATCH --job-name=jjas_matchstudy
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_study_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/matched_study_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
PLOTS="$ROOT/final_analysis/jjas/plots"
RUN="conda run --no-capture-output -n s2s-hind python -u"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host       : $(hostname)"
echo "start      : $(date)"
echo "compact nc : $(find /storage/raj.ayush/s2s_final_data/jjas/fuxi_combined -maxdepth 1 -type f -name '2019*.nc' | wc -l)"

$RUN "$PLOTS/matched_monsoon_study.py" \
  --year 2019 \
  --vars TP Z500 \
  --out "$PLOTS/figs/matched_monsoon_study"

echo "end        : $(date)"
