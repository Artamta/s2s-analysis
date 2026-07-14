#!/bin/bash
# Submit the strict information-matched 00 UTC FuXi-S2S benchmark.

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
LOCAL_STAGE="${REPO}/clean/model-runs/fuxi/slurm/stage_arco_strict00z_2019_2022.sbatch"
REMOTE_STAGE="${REPO}/clean/model-runs/fuxi/slurm/stage_arco_hourly_strict00z_2023_2025.sbatch"
FORECAST="${REPO}/clean/model-runs/fuxi/slurm/run_fuxi_strict00z_2020_2025_ens50.sbatch"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"

mkdir -p "${RUN_ROOT}/logs/era5_daily" "${RUN_ROOT}/logs/inference"

local_job=$(sbatch --parsable --array=0-36%3 "${LOCAL_STAGE}")
remote_submit=(sbatch --parsable --array=0-35%2)
if [[ -n "${FUXI_STRICT_REMOTE_AFTER:-}" ]]; then
  remote_submit+=(--dependency="afterok:${FUXI_STRICT_REMOTE_AFTER}")
fi
remote_job=$("${remote_submit[@]}" "${REMOTE_STAGE}")
pilot_job=$(sbatch --parsable --array=0 \
  --dependency="afterok:${local_job}_0:${local_job}_1" "${FORECAST}")

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

local_2020=$(month_dependencies "${local_job}" 0 12)
local_2021=$(month_dependencies "${local_job}" 12 24)
local_2022=$(month_dependencies "${local_job}" 24 36)
remote_2023="${local_job}_36:$(month_dependencies "${remote_job}" 0 11)"
remote_2024=$(month_dependencies "${remote_job}" 11 23)
remote_2025=$(month_dependencies "${remote_job}" 23 35)

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

printf 'strict_local_stage=%s\n' "${local_job}"
printf 'strict_remote_arco_stage=%s\n' "${remote_job}"
printf 'strict_pilot=%s\n' "${pilot_job}"
printf 'strict_forecast_2020=%s\n' "${forecast_2020}"
printf 'strict_forecast_2021=%s\n' "${forecast_2021}"
printf 'strict_forecast_2022=%s\n' "${forecast_2022}"
printf 'strict_forecast_2023=%s\n' "${forecast_2023}"
printf 'strict_forecast_2024=%s\n' "${forecast_2024}"
printf 'strict_forecast_2025=%s\n' "${forecast_2025}"
