#!/bin/bash
#SBATCH --job-name=fuxi_test_inf
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/test_inf_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/test_inf_%j.err

FUXI_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"
DATE="${TEST_DATE:-20260620}"
INPUT_FILE="/storage/raj.ayush/All_Model_Data/fuxi/test/inputs/${DATE}/input.nc"
OUT_DIR="/storage/raj.ayush/All_Model_Data/fuxi/test/raw/${DATE}"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:$LD_LIBRARY_PATH"

echo "================================================================"
echo "Job         : FuXi-S2S Test Inference"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Init date   : $DATE"
echo "Input       : $INPUT_FILE"
echo "Output      : $OUT_DIR"
echo "Members     : 3   |  Steps: 42 (42-day forecast)"
echo "Start time  : $(date)"
echo "================================================================"

mkdir -p "$OUT_DIR"

$PYTHON "$FUXI_DIR/inference.py" \
    --model   "$FUXI_DIR/model/fuxi_s2s.onnx" \
    --input   "$INPUT_FILE" \
    --device  cuda \
    --total_step   42 \
    --total_member 3 \
    --save_dir     "$OUT_DIR"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
