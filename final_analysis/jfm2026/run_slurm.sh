#!/bin/bash
#SBATCH --job-name=s2s_jfm2026
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026/logs/verify_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026/logs/verify_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00

# JFM2026 verification — SPIRE/FuXi/ECMWF (+MME,Persistence), all config vars, era5 basis.
# Resolution via DGRID env (default 1.5 common; 0.5 = SPIRE-native).
# Optional FuXi override for larger ensembles:
#   FUXI_ROOT=/storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50
#   FUXI_MEMBERS=50
#   OUT_SUFFIX=ens50
# Submit:  sbatch run_slurm.sh                       (1.5° common)
#          sbatch --export=ALL,DGRID=0.5 run_slurm.sh   (0.5° native)
#          sbatch --export=ALL,FUXI_ROOT=/storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50,FUXI_MEMBERS=50,OUT_SUFFIX=ens50 run_slurm.sh
set -euo pipefail
SCRIPT_DIR=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026
cd "$SCRIPT_DIR"
DGRID="${DGRID:-1.5}"
FUXI_ROOT="${FUXI_ROOT:-}"
FUXI_MEMBERS="${FUXI_MEMBERS:-}"
OUT_SUFFIX="${OUT_SUFFIX:-}"

# one BLAS thread per worker process (we parallelise over the 13 init dates)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [[ -n "${FUXI_ROOT}" ]]; then
    EXTRA_ARGS+=(--fuxi-root "${FUXI_ROOT}")
fi
if [[ -n "${FUXI_MEMBERS}" ]]; then
    EXTRA_ARGS+=(--fuxi-members "${FUXI_MEMBERS}")
fi
if [[ -n "${OUT_SUFFIX}" ]]; then
    EXTRA_ARGS+=(--out-suffix "${OUT_SUFFIX}")
fi

echo "host: $(hostname)   start: $(date)   dgrid=${DGRID}   fuxi_root=${FUXI_ROOT:-default}   out_suffix=${OUT_SUFFIX:-none}"
conda run --no-capture-output -n s2s-hind \
    python -u "$SCRIPT_DIR/run_verify.py" --workers 13 --dgrid "${DGRID}" "${EXTRA_ARGS[@]}"
echo "end: $(date)"
