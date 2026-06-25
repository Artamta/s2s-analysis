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

# JFM2026 verification — SPIRE/FuXi/ECMWF (+MME,Persistence), all config vars, era5 basis.
# Resolution via DGRID env (default 1.5 common; 0.5 = SPIRE-native).
# Submit:  sbatch run_slurm.sh                       (1.5° common)
#          sbatch --export=ALL,DGRID=0.5 run_slurm.sh   (0.5° native)
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026
cd "$SCRIPT_DIR"
DGRID="${DGRID:-1.5}"

# one BLAS thread per worker process (we parallelise over the 13 init dates)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   start: $(date)   dgrid=${DGRID}"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/run_verify.py" --workers 13 --dgrid "${DGRID}"
echo "end: $(date)"
