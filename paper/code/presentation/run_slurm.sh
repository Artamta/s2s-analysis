#!/bin/bash
#SBATCH --job-name=s2s_verify
#SBATCH --output=verify_%j.out
#SBATCH --error=verify_%j.err
#SBATCH --partition=GPU-AI          # allocation requested by user (CPU/IO job; GPU idle)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # 13 init-worker processes (+ headroom); IO-bound
#SBATCH --mem=512G
#SBATCH --time=04:00:00
# #SBATCH --gres=gpu:1              # uncomment ONLY if the partition refuses CPU-only jobs

# ==============================================================================
# Production run of the unified S2S verification pipeline.
#   - One worker process per init date (13). cfgrib/zarr are NOT thread-safe,
#     so we use PROCESSES (ProcessPoolExecutor), the safest/cleanest choice.
#   - This is a CPU + I/O workload (xarray/numpy); NO GPU is used. Submitted to
#     GPU-AI purely for the 512 GB / 64-CPU allocation.
#   - Keep BLAS single-threaded per process to avoid oversubscription.
# Submit from this directory:   sbatch run_slurm.sh
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Activate the same Python env you use interactively (edit if you use conda):
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate s2s
# or:  source /path/to/venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host: $(hostname)   start: $(date)"
python verify_s2s.py --workers 13 --vars TP Z500 T2M
echo "end: $(date)"
