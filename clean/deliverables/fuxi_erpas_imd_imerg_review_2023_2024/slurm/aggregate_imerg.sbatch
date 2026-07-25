#!/usr/bin/env bash
#SBATCH --job-name=imerg_agg
#SBATCH --partition=gpu_prio
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/imerg_aggregate_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/imerg_aggregate_%j.err

set -euo pipefail
REVIEW_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024
S2S_PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python

"${S2S_PYTHON}" "${REVIEW_ROOT}/scripts/aggregate_imerg_climatology.py"
