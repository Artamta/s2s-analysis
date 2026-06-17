#!/bin/bash
#SBATCH --job-name=s2s_verify_v2
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv2/verify_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv2/verify_%j.err
#SBATCH --partition=GPU-AI_prio         # allocation requested by user (CPU/IO job; GPU idle)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # 13 init-worker processes (+ headroom); IO-bound
#SBATCH --mem=512G
#SBATCH --time=04:00:00
# #SBATCH --gres=gpu:1              # uncomment ONLY if the partition refuses CPU-only jobs

# ==============================================================================
# V2 production run: model-own-climatology S2S verification.
#   MODELS : SPIRE, FuXi, ECMWF (+ MME, Persistence). NCEP dropped.
#   VARS   : TP, Z500 only (T2M dropped).
#   FuXi/ECMWF scored vs their OWN hindcast clima; SPIRE vs ERA5 clima.
#   - One worker process per init date (13). cfgrib/zarr are NOT thread-safe,
#     so we use PROCESSES (ProcessPoolExecutor).
#   - CPU + I/O workload (xarray/numpy); NO GPU used. On GPU-AI purely for the
#     512 GB / 64-CPU allocation.
# Submit from this directory:   sbatch run_slurm.sh
# ==============================================================================
set -euo pipefail
# NOTE: SLURM runs a SPOOLED copy of this script, so "$0" is NOT the real path.
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/paper/code/presentationv2
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
    python -u "$SCRIPT_DIR/verify_s2s.py" --workers 13 --vars TP Z500
echo "end: $(date)"
