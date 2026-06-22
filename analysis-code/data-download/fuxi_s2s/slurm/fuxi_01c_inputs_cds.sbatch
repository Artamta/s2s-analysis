#!/bin/bash
#SBATCH --job-name=fuxi_inputs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/fuxi_inputs_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/fuxi_inputs_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"

echo "================================================================"
echo "Job         : FuXi-S2S ERA5 Input Download via CDS (JFM 2026)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/inputs/"
echo "Dates       : 90 daily (2026-01-01 to 2026-03-31), skips existing"
echo "Grid        : Global 1.5deg — 76 channels for FuXi"
echo "================================================================"

$PYTHON "$SCRIPT_DIR/01c_download_fuxi_inputs_cds.py"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
