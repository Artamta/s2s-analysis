#!/bin/bash
#SBATCH --job-name=fuxi_arco
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-35%4
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50/logs/era5_daily/slurm_%A_%a.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50/logs/era5_daily/slurm_%A_%a.err

set -euo pipefail

REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
PYTHON="${FUXI_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}"
SCRIPT="${REPO}/clean/model-runs/fuxi/scripts/stage_arco_era5_daily.py"
CONFIG="${FUXI_CONFIG:-${REPO}/clean/config/fuxi_operational_2020_2025.json}"

YEAR=$((2020 + SLURM_ARRAY_TASK_ID / 12))
MONTH=$((1 + SLURM_ARRAY_TASK_ID % 12))

CHILD_PID=""
requeue_before_timeout() {
  echo "Requeueing job=${SLURM_JOB_ID} to continue direct-ARCO staging at $(date --iso-8601=seconds)"
  if [[ -n "${CHILD_PID}" ]]; then
    kill "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_before_timeout USR1

echo "year=${YEAR} month=${MONTH} job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -u "${SCRIPT}" \
  --year "${YEAR}" \
  --month "${MONTH}" \
  --config "${CONFIG}" &
CHILD_PID=$!
wait "${CHILD_PID}"
echo "year=${YEAR} month=${MONTH} end=$(date --iso-8601=seconds)"
