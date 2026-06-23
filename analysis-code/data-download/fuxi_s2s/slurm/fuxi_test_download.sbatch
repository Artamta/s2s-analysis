#!/bin/bash
#SBATCH --job-name=fuxi_test_dl
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/test_dl_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/test_dl_%j.err

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
DATE="${TEST_DATE:-20260620}"

echo "================================================================"
echo "Job         : FuXi-S2S Test Input Download"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Init date   : $DATE"
echo "Output      : /storage/raj.ayush/All_Model_Data/fuxi/test/inputs/"
echo "Start time  : $(date)"
echo "================================================================"

$PYTHON "$SCRIPT_DIR/download_test_input.py" --date "$DATE"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
