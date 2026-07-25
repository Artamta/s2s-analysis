#!/usr/bin/env bash
#SBATCH --job-name=fuxi_review
#SBATCH --partition=gpu_prio
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/review_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/review_%j.err

set -euo pipefail
REVIEW_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024
S2S_PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
PLOT_PYTHON=/apps/compilers/anaconda3-2023.3/bin/python

"${S2S_PYTHON}" "${REVIEW_ROOT}/scripts/calculate_review_metrics.py" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260723
"${PLOT_PYTHON}" "${REVIEW_ROOT}/scripts/plot_review_figures.py" --dpi 240
