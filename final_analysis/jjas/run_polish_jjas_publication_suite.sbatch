#!/bin/bash
#SBATCH --job-name=jjas_polish
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/jjas_polish_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/jjas_polish_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
PLOTS="$ROOT/final_analysis/jjas/plots"
RUN="conda run --no-capture-output -n s2s-hind python -u"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host  : $(hostname)"
echo "start : $(date)"

$RUN "$PLOTS/polish_jjas_publication_suite.py" --source auto

echo "end   : $(date)"
