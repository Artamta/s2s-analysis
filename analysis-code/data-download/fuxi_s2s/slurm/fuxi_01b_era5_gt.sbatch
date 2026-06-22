#!/bin/bash
#SBATCH --job-name=era5_gt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/era5_gt_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/era5_gt_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"

echo "================================================================"
echo "Job         : ERA5 Ground Truth Download (JFM 2026, Jan-May)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth/"
echo "Vars        : z500, t2m, tp, msl  |  Domain: India 0.25deg"
echo "================================================================"

$PYTHON "$SCRIPT_DIR/01b_download_era5_gt.py"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
