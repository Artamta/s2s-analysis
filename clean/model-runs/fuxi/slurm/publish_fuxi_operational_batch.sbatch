#!/bin/bash
# Serialize and publish all successful source/date cases in one submitted batch.

#SBATCH --job-name=fuxi-web-b
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_publish_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/operational_publish_%j.err

set -euo pipefail
: "${FUXI_BATCH:?FUXI_BATCH must be exported by the submitter}"

REPO=/home/raj.ayush/s2s/s2s_anlysis
PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
SCRIPT=${REPO}/clean/model-runs/fuxi/scripts/fuxi_operational.py
LOCK_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/operational
mkdir -p "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/dashboard-publish.lock"
flock 9

"${PYTHON}" -u "${SCRIPT}" publish-batch --batch "${FUXI_BATCH}"
