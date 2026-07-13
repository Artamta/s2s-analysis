#!/bin/bash
# Submit the complete hybrid ERA5 staging and FuXi forecast workflow.

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
LOCAL_STAGE="${REPO}/clean/model-runs/fuxi/slurm/stage_arco_era5_daily_2020_2022.sbatch"
REMOTE_STAGE="${REPO}/clean/model-runs/fuxi/slurm/stage_arco_hourly_2023_2025.sbatch"
FORECAST="${REPO}/clean/model-runs/fuxi/slurm/run_fuxi_2020_2025_ens50.sbatch"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50"

mkdir -p "${RUN_ROOT}/logs/era5_daily" "${RUN_ROOT}/logs/inference"

local_job=$(sbatch --parsable --array=0-35%3 "${LOCAL_STAGE}")
remote_job=$(sbatch --parsable --array=0-35%2 "${REMOTE_STAGE}")
pilot_job=$(sbatch --parsable --array=0 \
  --dependency="afterok:${local_job}_0" "${FORECAST}")

month_dependencies() {
  local job_id=$1
  local first=$2
  local last=$3
  local dependency=""
  local task
  for ((task=first; task<=last; task++)); do
    if [[ -n "${dependency}" ]]; then
      dependency+=":"
    fi
    dependency+="${job_id}_${task}"
  done
  printf '%s' "${dependency}"
}

local_2020=$(month_dependencies "${local_job}" 0 11)
local_2021=$(month_dependencies "${local_job}" 12 23)
local_2022=$(month_dependencies "${local_job}" 24 35)
remote_2023=$(month_dependencies "${remote_job}" 0 11)
remote_2024=$(month_dependencies "${remote_job}" 12 23)
remote_2025=$(month_dependencies "${remote_job}" 24 35)

forecast_2020=$(sbatch --parsable --array=1-104%2 \
  --dependency="afterok:${local_2020}:${pilot_job}" "${FORECAST}")
forecast_2021=$(sbatch --parsable --array=105-208%2 \
  --dependency="afterok:${local_2021}:${pilot_job}" "${FORECAST}")
forecast_2022=$(sbatch --parsable --array=209-312%2 \
  --dependency="afterok:${local_2022}:${pilot_job}" "${FORECAST}")
forecast_2023=$(sbatch --parsable --array=313-416%2 \
  --dependency="afterok:${remote_2023}:${pilot_job}" "${FORECAST}")
forecast_2024=$(sbatch --parsable --array=417-516%2 \
  --dependency="afterok:${remote_2024}:${pilot_job}" "${FORECAST}")
forecast_2025=$(sbatch --parsable --array=517-620%2 \
  --dependency="afterok:${remote_2025}:${pilot_job}" "${FORECAST}")

printf 'local_stage=%s\n' "${local_job}"
printf 'remote_arco_stage=%s\n' "${remote_job}"
printf 'pilot=%s\n' "${pilot_job}"
printf 'forecast_2020=%s\n' "${forecast_2020}"
printf 'forecast_2021=%s\n' "${forecast_2021}"
printf 'forecast_2022=%s\n' "${forecast_2022}"
printf 'forecast_2023=%s\n' "${forecast_2023}"
printf 'forecast_2024=%s\n' "${forecast_2024}"
printf 'forecast_2025=%s\n' "${forecast_2025}"
