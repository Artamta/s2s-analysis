#!/bin/bash
# Download ERPAS Drive data directly into canonical forecast/reforecast storage.

#SBATCH --job-name=erpas_gdrive
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/logs/erpas/slurm_%j.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/logs/erpas/slurm_%j.err

set -euo pipefail

FOLDER_ID="1S58Swev_M33lAK6Aap7FJuFvG5gtv3Jf"
ROOT="/storage/raj.ayush/s2s_final_data/final_iteration"
RCLONE="${ERPAS_RCLONE_BIN:-${ROOT}/tools/rclone}"
FORECAST_ROOT="${ROOT}/raw/erpas/forecast"
CLIMATOLOGY_ROOT="${ROOT}/raw/erpas/reforecast/climatology"
STAGING_ROOT="${ROOT}/staging/erpas/google_drive_20260714_${SLURM_JOB_ID}"
MANIFEST_DIR="${ROOT}/manifests/erpas/google_drive_20260714"
LOG_DIR="${ROOT}/logs/erpas"
RCLONE_LOG="${LOG_DIR}/rclone_${SLURM_JOB_ID}.log"

mkdir -p \
  "${FORECAST_ROOT}" \
  "${CLIMATOLOGY_ROOT}" \
  "${STAGING_ROOT}" \
  "${MANIFEST_DIR}" \
  "${LOG_DIR}"

common_copy=(
  --drive-root-folder-id "${FOLDER_ID}"
  --transfers 2
  --checkers 4
  --bwlimit 20M
  --retries 10
  --low-level-retries 20
  --retries-sleep 30s
  --stats 1m
  --stats-one-line
  --log-file "${RCLONE_LOG}"
  --log-level INFO
)

echo "ERPAS transfer job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
"${RCLONE}" version

for year in 2023 2024 2025; do
  annual="${FORECAST_ROOT}/annual${year}"
  "${RCLONE}" copy gdrive: "${annual}/tp" \
    --max-depth 1 --include "/APCP_${year}*.grb" "${common_copy[@]}"
  "${RCLONE}" copy gdrive: "${annual}/surface_temperature" \
    --max-depth 1 --include "/tsfc_${year}*.grb" "${common_copy[@]}"
  "${RCLONE}" copy gdrive: "${annual}/geopotential_height" \
    --max-depth 1 --include "/gpot_${year}*.grb" "${common_copy[@]}"
  "${RCLONE}" copy gdrive: "${annual}/tp_india_0p5" \
    --max-depth 1 --include "/Ind_0.5_APCP_${year}*.grb" "${common_copy[@]}"
done

# Drive contains many folders with the same climatology name. Filtering from
# the parent folder merges every duplicate-named branch without dropping dates.
"${RCLONE}" copy gdrive: "${STAGING_ROOT}" \
  --include "/z_HINDCAST_CLIMATOLOGY/**" \
  --create-empty-src-dirs \
  "${common_copy[@]}"
"${RCLONE}" copy \
  "${STAGING_ROOT}/z_HINDCAST_CLIMATOLOGY" \
  "${CLIMATOLOGY_ROOT}" \
  --transfers 4 --checkers 8

for year in 2023 2024 2025; do
  annual="${FORECAST_ROOT}/annual${year}"
  "${RCLONE}" check gdrive: "${annual}/tp" \
    --drive-root-folder-id "${FOLDER_ID}" \
    --max-depth 1 --include "/APCP_${year}*.grb" --one-way
  "${RCLONE}" check gdrive: "${annual}/surface_temperature" \
    --drive-root-folder-id "${FOLDER_ID}" \
    --max-depth 1 --include "/tsfc_${year}*.grb" --one-way
  "${RCLONE}" check gdrive: "${annual}/geopotential_height" \
    --drive-root-folder-id "${FOLDER_ID}" \
    --max-depth 1 --include "/gpot_${year}*.grb" --one-way
  "${RCLONE}" check gdrive: "${annual}/tp_india_0p5" \
    --drive-root-folder-id "${FOLDER_ID}" \
    --max-depth 1 --include "/Ind_0.5_APCP_${year}*.grb" --one-way
done
"${RCLONE}" check \
  "${STAGING_ROOT}/z_HINDCAST_CLIMATOLOGY" \
  "${CLIMATOLOGY_ROOT}" \
  --one-way

"${RCLONE}" lsjson "${ROOT}/raw/erpas" \
  --recursive --files-only --hash > "${MANIFEST_DIR}/inventory.json"
"${RCLONE}" hashsum SHA-256 "${ROOT}/raw/erpas" \
  > "${MANIFEST_DIR}/sha256.txt"
"${RCLONE}" size "${ROOT}/raw/erpas" --json \
  > "${MANIFEST_DIR}/size.json"

rm -rf -- "${STAGING_ROOT:?}"
echo "ERPAS transfer job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds)"
