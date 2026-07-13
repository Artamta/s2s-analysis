#!/bin/bash
#SBATCH --job-name=s2s_annual
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-24%3
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/logs/production/slurm_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/logs/production/slurm_%A_%a.err

set -euo pipefail

REPO="/home/raj.ayush/s2s/s2s_anlysis"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
SCRIPT="${REPO}/clean/data-download/scripts/download_all_season_operational.py"
CALENDAR="${REPO}/clean/config/all_season_dates_2020_2024.csv"
OUTPUT="/storage/raj.ayush/s2s_final_data/final_iteration"
PROVIDERS=(ecmwf ukmo ncep cma cnrm)

PROVIDER_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
YEAR=$((2020 + SLURM_ARRAY_TASK_ID / 5))
PROVIDER="${PROVIDERS[${PROVIDER_INDEX}]}"

if [[ ! -s "${HOME}/.cdsapirc" ]]; then
  echo "Missing ${HOME}/.cdsapirc" >&2
  exit 2
fi

echo "provider=${PROVIDER} year=${YEAR} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" "${SCRIPT}" \
  --provider "${PROVIDER}" \
  --year "${YEAR}" \
  --calendar "${CALENDAR}" \
  --output-root "${OUTPUT}" \
  --retries 4 \
  --sleep-between 1.5
echo "provider=${PROVIDER} year=${YEAR} end=$(date --iso-8601=seconds)"
