#!/bin/bash
#SBATCH --job-name=fuxi_jjas25
#SBATCH --partition=GPU-AI_prio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/jjas2025_ens50_inf_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/jjas2025_ens50_inf_%j.err

set -uo pipefail

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/fuxi/bin/python"

START_DATE="${FUXI_START:-20250601}"
END_DATE="${FUXI_END:-20250930}"
INPUT_DIR="${FUXI_INPUT_DIR:-/storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs}"
RAW_DIR="${FUXI_RAW_DIR:-/storage/raj.ayush/All_Model_Data/fuxi/jjas2025_ens50/raw}"
MEMBERS="${FUXI_MEMBERS:-50}"
STEPS="${FUXI_STEPS:-42}"

ALLOW_FLAG=""
if [[ "${ALLOW_MISSING_INPUTS:-0}" == "1" ]]; then
  ALLOW_FLAG="--allow-missing-inputs"
fi

mkdir -p "${SCRIPT_DIR}/logs" "${RAW_DIR}"
export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:${LD_LIBRARY_PATH:-}"

echo "================================================================"
echo "Job         : FuXi-S2S Inference - 50-member ensemble (JJAS 2025)"
echo "Job ID      : ${SLURM_JOB_ID:-manual}"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Dates       : ${START_DATE} to ${END_DATE}"
echo "Members     : ${MEMBERS}"
echo "Steps       : ${STEPS}"
echo "Input       : ${INPUT_DIR}"
echo "Output      : ${RAW_DIR}"
echo "Start time  : $(date)"
echo "================================================================"

"${PYTHON}" "${SCRIPT_DIR}/02c_run_inference_50mem_jjas2025.py" \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --input-dir "${INPUT_DIR}" \
  --raw-dir "${RAW_DIR}" \
  --members "${MEMBERS}" \
  --steps "${STEPS}" \
  --device cuda \
  ${ALLOW_FLAG}
EXIT_CODE=$?

echo "================================================================"
echo "End time : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "================================================================"
exit "${EXIT_CODE}"
