#!/bin/bash
#SBATCH --job-name=hfm_verify_jjas
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm/verify_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm/verify_%j.err
#SBATCH --partition=GPU-AI_prio         # allocation requested by user (CPU/IO job; GPU idle)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # 35 init-worker processes (+ headroom); IO-bound
#SBATCH --mem=512G
#SBATCH --time=04:00:00

# ==============================================================================
# HFM production run: model-own-climatology S2S verification for JJAS 2019.
#   MODELS : FuXi, ECMWF (+ MME, Persistence). SPIRE and NCEP dropped.
#   VARS   : TP, Z500 only.
#   FuXi/ECMWF scored vs their OWN hindcast clima.
#   Submit from this directory:   sbatch run_slurm.sh
# ==============================================================================
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/paper/code/hfm
cd "$SCRIPT_DIR"

# Run inside the s2s-hind conda env (cartopy, arraylake, cfgrib).
# `conda run --no-capture-output` keeps stdout UNBUFFERED so the log streams live.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   start: $(date)"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/verify_s2s.py" --workers 35 --vars TP Z500
echo "end: $(date)"
