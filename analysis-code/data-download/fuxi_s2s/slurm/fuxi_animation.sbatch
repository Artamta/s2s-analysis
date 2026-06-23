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

echo "================================================================"
echo "Job      : FuXi-S2S Forecast Animations"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Mode     : ${MODE:-all}"
echo "Start    : $(date)"
echo "================================================================"

MODE_ARG=""
if [ -n "$MODE" ]; then
    MODE_ARG="--mode $MODE"
fi

$PYTHON "$SCRIPT" $MODE_ARG --fps 3 --members 0

echo "================================================================"
echo "End : $(date)"
echo "================================================================"
