#!/bin/bash
#SBATCH --job-name=s2s_verify_v3
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv3/verify_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv3/verify_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00

# V3: dual clim-basis verification (era5 + model_own for FuXi/ECMWF).
# Submit: sbatch run_slurm.sh
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv3
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   start: $(date)"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/verify_s2s.py" --workers 13 --vars TP Z500
echo "end: $(date)"
