#!/usr/bin/env bash
#SBATCH --job-name=imerg_clim
#SBATCH --partition=gpu_prio
#SBATCH --array=0-21%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/imerg_year_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024/logs/imerg_year_%A_%a.err

set -euo pipefail
REVIEW_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/deliverables/fuxi_erpas_imd_imerg_review_2023_2024
S2S_PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
CLIMATOLOGY_YEAR=$((2001 + SLURM_ARRAY_TASK_ID))

"${S2S_PYTHON}" "${REVIEW_ROOT}/scripts/stage_imerg_climatology_year.py" \
  --year "${CLIMATOLOGY_YEAR}" \
  --workers "${SLURM_CPUS_PER_TASK}"
