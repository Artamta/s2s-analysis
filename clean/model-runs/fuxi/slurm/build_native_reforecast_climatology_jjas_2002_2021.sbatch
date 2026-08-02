#!/bin/bash
#SBATCH --job-name=fuxi_rf_climo
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021/logs/climatology_%j.out
#SBATCH --error=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021/logs/climatology_%j.err

set -euo pipefail
REPO="${FUXI_REPO_ROOT:-/home/raj.ayush/s2s/s2s_anlysis}"
PYTHON="${FUXI_DRIVER_PYTHON:-/home/raj.ayush/.conda/envs/s2s-hind/bin/python}"
RUN_ROOT="/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/native_reforecast_jjas_2002_2021"
"${PYTHON}" "${REPO}/clean/model-runs/fuxi/scripts/build_fuxi_reforecast_climatology.py" \
  --manifest "${REPO}/clean/config/fuxi_native_reforecast_jjas_2002_2021.csv" \
  --shard-dir "${RUN_ROOT}/shards" \
  --output "${RUN_ROOT}/fuxi_s2s_jjas_model_climatology_2002_2021_loyo.nc"
