#!/bin/bash
#SBATCH --job-name=fuxi_extra
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_extra_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_extra_%j.err

PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
SCRIPT="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/plot_erpas_style.py"

DATE="${ERPAS_DATE:-20260617}"
OUT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/erpas_style/${DATE}"

echo "================================================================"
echo "Job    : FuXi-S2S  Extra diagnostic plots (7 new types)"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $DATE"
echo "Out    : $OUT_DIR"
echo "Plots  : olr  tcwv  hovmoller  z500_anom  wind850  sst  shear"
echo "Start  : $(date)"
echo "================================================================"

mkdir -p "$OUT_DIR"

for MODE in olr tcwv hovmoller z500_anom wind850 sst shear; do
    echo ""
    echo "─── MODE: $MODE ───"
    $PYTHON "$SCRIPT" \
        --date    "$DATE" \
        --out_dir "$OUT_DIR" \
        --steps   42 \
        --fps     4 \
        --mode    "$MODE"
done

echo ""
echo "================================================================"
echo "End : $(date)"
echo "================================================================"
