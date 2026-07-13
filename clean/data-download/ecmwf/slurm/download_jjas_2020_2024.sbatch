#!/bin/bash
#SBATCH --job-name=ecmwf_core
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=2020-2024%2
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/logs/ecmwf/jjas_%A_%a.log
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/logs/ecmwf/jjas_%A_%a.err

set -euo pipefail

REPO="/home/raj.ayush/s2s/s2s_anlysis"
PYTHON="/home/raj.ayush/.conda/envs/fuxi/bin/python"
SCRIPT="${REPO}/clean/data-download/ecmwf/scripts/download_ecmwf_operational.py"
STORAGE="/storage/raj.ayush/s2s_final_data/final_iteration"
YEAR="${SLURM_ARRAY_TASK_ID}"
OUT_ROOT="${STORAGE}/raw/ecmwf/forecast/jjas${YEAR}"

mkdir -p "${STORAGE}/logs/ecmwf" "${OUT_ROOT}"

if [[ ! -s "${HOME}/.cdsapirc" ]]; then
  echo "Missing or empty ${HOME}/.cdsapirc" >&2
  exit 2
fi

if ! "${PYTHON}" -c "import cdsapi, xarray" >/dev/null 2>&1; then
  echo "The fuxi environment must contain cdsapi and xarray" >&2
  exit 2
fi

echo "ECMWF operational S2S core benchmark"
echo "year=${YEAR} cadence=Mon/Thu variables=tp,z500 leads=1-42 types=cf,pf"
echo "output=${OUT_ROOT}"

"${PYTHON}" "${SCRIPT}" \
  --year "${YEAR}" \
  --out-root "${OUT_ROOT}" \
  --cadence mon-thu \
  --variables tp,z500 \
  --forecast-types cf,pf \
  --lead-days 42 \
  --retries 5 \
  --sleep-between 2
