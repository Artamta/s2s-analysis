#!/bin/bash
#SBATCH --job-name=fuxi_comb50
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/combine_ens50_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/combine_ens50_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S Combine — 50-member ensemble (JFM 2026)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Input       : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/raw/"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026_ens50/combined/"
echo "Start time  : $(date)"
echo "================================================================"

$PYTHON "$SCRIPT_DIR/03b_combine_50mem.py"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
