#!/bin/bash
#SBATCH --job-name=fuxi_ens50
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/ens50_inf_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/ens50_inf_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S Inference — 50-member ensemble (JFM 2026)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Members     : 50"
echo "Dates       : 90 (2026-01-01 to 2026-03-31)"
echo "Input       : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/raw/"
echo "Start time  : $(date)"
echo "================================================================"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:$LD_LIBRARY_PATH"

$PYTHON "$SCRIPT_DIR/02b_run_inference_50mem.py" --device cuda

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
