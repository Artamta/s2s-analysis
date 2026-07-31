#!/bin/bash
#SBATCH --job-name=atmos42-global
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/atmos42_global_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/logs/atmos42_global_%j.err

set -euo pipefail

ANALYSIS_ROOT=/home/raj.ayush/s2s/s2s_anlysis
MODEL_ROOT=${ANALYSIS_ROOT}/analysis-code/data-download/fuxi_s2s/FuXi-S2S
INPUT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/fuxi_s2s_gfs_proxy_case_20260728_ens24/inputs/annual2026/20260728/input.nc
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/atmosphere42_global_20260728
RAW_DIR=${RUN_ROOT}/raw
PUBLIC_DIR=${RUN_ROOT}/public
INFERENCE_PYTHON=/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python
EXPORT_PYTHON=/home/raj.ayush/.conda/envs/s2s-hind/bin/python
TASK_CACHE=/tmp/atmos42_global_${SLURM_JOB_ID}

test -f "${INPUT}"
test -f "${MODEL_ROOT}/model/fuxi_s2s.onnx"
mkdir -p "${RAW_DIR}" "${PUBLIC_DIR}" "${TASK_CACHE}/xdg" "${TASK_CACHE}/mpl"

export XDG_CACHE_HOME=${TASK_CACHE}/xdg
export MPLCONFIGDIR=${TASK_CACHE}/mpl
export LD_LIBRARY_PATH=/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled

cd "${MODEL_ROOT}"
set +e
"${INFERENCE_PYTHON}" -u inference.py \
  --model model/fuxi_s2s.onnx \
  --input "${INPUT}" \
  --device cuda \
  --total_step 42 \
  --total_member 100 \
  --save_dir "${RAW_DIR}"
INFERENCE_STATUS=$?
set -e

if [[ "${INFERENCE_STATUS}" -ne 0 && "${INFERENCE_STATUS}" -ne 255 ]]; then
  exit "${INFERENCE_STATUS}"
fi
RAW_COUNT=$(find "${RAW_DIR}/member" -mindepth 2 -maxdepth 2 -type f -name '*.nc' | wc -l)
test "${RAW_COUNT}" -eq 4200

"${EXPORT_PYTHON}" \
  "${ANALYSIS_ROOT}/clean/fuxi-dashboard/scripts/export_global_forecast.py" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${PUBLIC_DIR}" \
  --members 100 \
  --lead-days 42 \
  --initialization 2026-07-28
