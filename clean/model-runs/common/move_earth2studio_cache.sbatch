#!/usr/bin/env bash
#SBATCH --job-name=move_e2cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/logs/move_e2cache_%j.out
#SBATCH --error=/home/raj.ayush/logs/move_e2cache_%j.err

set -euo pipefail

SOURCE="/home/raj.ayush/.cache/earth2studio"
TARGET="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/_shared_cache/earth2studio"
TRANSFER_LOG="${TARGET}/cache_move_${SLURM_JOB_ID}.log"

if [[ -L "${SOURCE}" ]]; then
  echo "Cache is already linked to $(readlink -f "${SOURCE}")"
  exit 0
fi

install -d -m 2775 "${TARGET}"
rsync -a --remove-source-files --info=stats2 --log-file="${TRANSFER_LOG}" \
  "${SOURCE}/" "${TARGET}/"

# This removes directories only when rsync has left them empty.
find "${SOURCE}" -depth -type d -empty -delete

if [[ -e "${SOURCE}" || -L "${SOURCE}" ]]; then
  echo "Source still contains data; refusing to replace it with a symlink." >&2
  exit 1
fi

ln -s "${TARGET}" "${SOURCE}"
echo "Earth2Studio cache linked: ${SOURCE} -> ${TARGET}"
du -shL "${SOURCE}"
