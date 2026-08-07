#!/bin/bash
# Publish one or more completed source cases for the same issue date.
# Required exported variables: colon-separated FUXI_CONFIGS and FUXI_DATE.

#SBATCH --job-name=fuxi-web
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_publish_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_publish_%j.err

set -euo pipefail

: "${FUXI_CONFIGS:?FUXI_CONFIGS must be exported by the submitter}"
: "${FUXI_DATE:?FUXI_DATE must be exported by the submitter}"

REPO=${S2S_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}
PYTHON=${S2S_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}
SCRIPT=${REPO}/clean/model-runs/fuxi/scripts/fuxi_operational.py
IFS=: read -r -a CONFIG_ARRAY <<< "${FUXI_CONFIGS}"
LOCK_ROOT=${S2S_FUXI_STORAGE_ROOT:-/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/operational}
mkdir -p "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/dashboard-publish.lock"
flock 9

echo "date=${FUXI_DATE} configs=${#CONFIG_ARRAY[@]} host=$(hostname) start=$(date --iso-8601=seconds)"
"${PYTHON}" -u "${SCRIPT}" publish --date "${FUXI_DATE}" --configs "${CONFIG_ARRAY[@]}"
echo "date=${FUXI_DATE} end=$(date --iso-8601=seconds)"
