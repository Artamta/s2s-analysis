#!/bin/bash
#SBATCH --job-name=hfm_spatial_maps
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm/spatial_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm/spatial_%j.err
#SBATCH --partition=GPU-AI_prio         # allocation requested by user
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 32 parallel worker processes
#SBATCH --mem=256G
#SBATCH --time=00:30:00

# ==============================================================================
# HFM production run: Spatial maps generation for JJAS 2019.
# Submit from this directory:   sbatch run_spatial_slurm.sh
# ==============================================================================
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm
cd "$SCRIPT_DIR"

# Run inside the s2s-hind conda env (cartopy, arraylake, cfgrib).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   start: $(date)"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/plots_scr/make_spatial_maps.py" --workers 32
echo "end: $(date)"
