#!/bin/bash
#SBATCH --job-name=fuxi_anim
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/fuxi_anim_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/fuxi_anim_%j.err

PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
SCRIPT="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/make_forecast_animation.py"

DATE="${ANIM_DATE:-20260618}"
RAW_DIR="/storage/raj.ayush/All_Model_Data/fuxi/test/raw"
OUT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/forecast_animation/${DATE}"

echo "================================================================"
echo "Job      : FuXi-S2S Forecast Animations"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Date     : $DATE"
echo "Mode     : ${MODE:-all}"
echo "Output   : $OUT_DIR"
echo "Start    : $(date)"
echo "================================================================"

mkdir -p "$OUT_DIR"

MODE_ARG=""
if [ -n "$MODE" ]; then
    MODE_ARG="--mode $MODE"
fi

$PYTHON "$SCRIPT" $MODE_ARG \
    --date    "$DATE" \
    --raw_dir "$RAW_DIR" \
    --out_dir "$OUT_DIR" \
    --fps 3 --members 0

echo "================================================================"
echo "End : $(date)"
echo "================================================================"
