#!/bin/bash
#SBATCH --job-name=s2s_jfm2026
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026/verify_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026/verify_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00

# JFM2026 verification — SPIRE/FuXi/ECMWF (+MME,Persistence), TP+Z500, era5 basis.
# Submit:  sbatch run_slurm.sh
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026
cd "$SCRIPT_DIR"

# one BLAS thread per worker process (we parallelise over the 13 init dates)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   start: $(date)"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/run_verify.py" --workers 13 --vars TP Z500
echo "end: $(date)"
