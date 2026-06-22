#!/bin/bash
#SBATCH --job-name=fuxi_infer
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/inference_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/inference_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S Inference (JFM 2026 daily, 90 dates)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time  : $(date)"
echo "Input       : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/raw/"
echo "================================================================"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:$LD_LIBRARY_PATH"

$PYTHON "$SCRIPT_DIR/02_run_inference.py" --device cuda

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
