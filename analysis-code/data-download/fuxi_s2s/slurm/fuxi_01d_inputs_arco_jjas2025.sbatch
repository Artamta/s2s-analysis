#!/bin/bash
#SBATCH --job-name=fuxi_in25
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/jjas2025_inputs_arco_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/jjas2025_inputs_arco_%j.err

set -euo pipefail

SCRIPT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s"
PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"

START_DATE="${FUXI_START:-20250601}"
END_DATE="${FUXI_END:-20250930}"
INPUT_DIR="${FUXI_INPUT_DIR:-/storage/raj.ayush/All_Model_Data/fuxi/jjas2025/inputs}"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_DIR}"

echo "================================================================"
echo "Job         : FuXi-S2S ERA5 Input Download via ARCO (JJAS 2025)"
echo "Job ID      : ${SLURM_JOB_ID:-manual}"
echo "Node        : $(hostname)"
echo "Dates       : ${START_DATE} to ${END_DATE}"
echo "Output      : ${INPUT_DIR}"
echo "Grid        : global 1.5deg, 76 channels, prev+init 00 UTC"
echo "Start time  : $(date)"
echo "================================================================"

"${PYTHON}" "${SCRIPT_DIR}/01d_download_fuxi_inputs_arco_jjas2025.py" \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --input-dir "${INPUT_DIR}"

echo "================================================================"
echo "End time : $(date)"
echo "================================================================"
