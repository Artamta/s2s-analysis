#!/bin/bash
#SBATCH --job-name=fuxi_download
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/download_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/download_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S ERA5 Input Download (JFM 2026 daily)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/"
echo "================================================================"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

$PYTHON "$SCRIPT_DIR/01_download_inputs.py"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
