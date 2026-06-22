#!/bin/bash
#SBATCH --job-name=fuxi_combine
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/combine_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/combine_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S Combine Output (JFM 2026)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Input       : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/raw/"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/combined/"
echo "================================================================"

$PYTHON "$SCRIPT_DIR/03_combine_output.py"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
