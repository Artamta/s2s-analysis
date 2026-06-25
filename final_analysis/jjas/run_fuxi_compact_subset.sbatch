#!/bin/bash
#SBATCH --job-name=fuxi_jjas_compact
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_compact_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_compact_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Compact selected FuXi-S2S JJAS hindcast archives into small NetCDF files.
# This is CPU/disk heavy, so run it on SLURM, not the login node.
#
# Submit a representative monsoon sequence:
#   sbatch run_fuxi_compact_subset.sbatch
#
# Override dates:
#   sbatch --export=ALL,INITS=20190623,20190627 run_fuxi_compact_subset.sbatch

set -euo pipefail

JJAS=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas
RUN="conda run --no-capture-output -n s2s-hind python -u"

DEFAULT_INITS="20190623,20190627,20190630,20190704,20190707,20190711,20190714,20190718,20190721,20190725,20190728"
INITS="${INITS:-$DEFAULT_INITS}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FUXI_SCRATCH="${FUXI_SCRATCH:-/tmp/${USER}/fuxi_extract_${SLURM_JOB_ID}}"

echo "host       : $(hostname)"
echo "start      : $(date)"
echo "inits      : ${INITS}"
echo "scratch    : ${FUXI_SCRATCH}"
echo "py7zr check:"
conda run --no-capture-output -n s2s-hind python - <<'PY'
import py7zr
print("  py7zr", py7zr.__version__)
PY

$RUN "$JJAS/preprocess_fuxi.py" --init "$INITS"

echo "end        : $(date)"
