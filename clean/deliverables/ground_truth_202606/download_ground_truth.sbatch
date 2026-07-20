#!/bin/bash
#SBATCH --job-name=rain_gt_2026
#SBATCH --partition=GPU-AI_prio
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/ground_truth_202606/download_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/ground_truth_202606/download_%j.err

set -euo pipefail

/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  /home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/ground_truth_202606/download_ground_truth.py \
  --start 2026-06-02 \
  --end 2026-07-18 \
  --output /home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/ground_truth_202606/all_india_daily_ground_truth_20260602_20260718.csv
