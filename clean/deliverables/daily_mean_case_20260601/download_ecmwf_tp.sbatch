#!/bin/bash
#SBATCH --job-name=ecmwf_tp_0601
#SBATCH --partition=GPU-AI_prio
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/daily_mean_case_20260601/ecmwf_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/daily_mean_case_20260601/ecmwf_%j.err

set -euo pipefail

/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  /home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/daily_mean_case_20260601/download_ecmwf_tp.py \
  --date 20260601 \
  --out-root /storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp
