#!/bin/bash
#SBATCH --job-name=fuxi_erpas
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_%j.err

PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
SCRIPT="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/plot_erpas_style.py"

DATE="${ERPAS_DATE:-20260617}"
OUT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/erpas_style/${DATE}"
# Local ERA5 climo (much faster than WB2); unset to fall back to WB2
CLIMO_NC="${CLIMO_NC:-/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc}"

echo "================================================================"
echo "Job    : FuXi-S2S ERPAS-style plots"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $DATE"
echo "Out    : $OUT_DIR"
echo "Climo  : $CLIMO_NC"
echo "Start  : $(date)"
echo "================================================================"

CLIMO_ARG=""
[ -f "$CLIMO_NC" ] && CLIMO_ARG="--climo_nc $CLIMO_NC"

$PYTHON "$SCRIPT" \
    --date    "$DATE" \
    --out_dir "$OUT_DIR" \
    --steps   42 \
    --fps     4 \
    $CLIMO_ARG

echo "================================================================"
echo "End : $(date)"
echo "================================================================"
