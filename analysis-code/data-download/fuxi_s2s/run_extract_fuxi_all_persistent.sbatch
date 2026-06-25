#!/bin/bash
#SBATCH --job-name=fuxi_extract_all
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_extract_all_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/fuxi_extract_all_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-999%4

# Persistent full extraction of all FuXi-S2S hindcast archives.
#
# Output layout:
#   ${OUT_ROOT}/YYYY/MM/YYYYMMDD/member/NN/SS.nc
#
# This job does NOT delete extracted data. It skips complete dates, writes
# status JSON files, and marks successful dates with `.complete`.
#
# Submit:
#   sbatch run_extract_fuxi_all_persistent.sbatch
#
# Submit all 2080 archives on clusters with array-size limits:
#   sbatch --array=0-999%4 --export=ALL,ARRAY_OFFSET=0 run_extract_fuxi_all_persistent.sbatch
#   sbatch --array=0-999%4 --export=ALL,ARRAY_OFFSET=1000 run_extract_fuxi_all_persistent.sbatch
#   sbatch --array=0-79%4  --export=ALL,ARRAY_OFFSET=2000 run_extract_fuxi_all_persistent.sbatch
#
# Override output root:
#   sbatch --export=ALL,OUT_ROOT=/some/large/path run_extract_fuxi_all_persistent.sbatch

set -euo pipefail

ROOT=/home/raj.ayush/s2s/s2s_anlysis
SCRIPT="$ROOT/analysis-code/data-download/fuxi_s2s/extract_fuxi_archives_persistent.py"
ARCHIVE_DIR=/storage/raj.ayush/archive/All_Model_Data/models/fuxi/data
OUT_ROOT="${OUT_ROOT:-/storage/raj.ayush/archive/All_Model_Data/models/fuxi/extracted_full}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
ARCHIVE_INDEX=$((SLURM_ARRAY_TASK_ID + ARRAY_OFFSET))
RUN="conda run --no-capture-output -n s2s-hind python -u"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host       : $(hostname)"
echo "start      : $(date)"
echo "task       : ${SLURM_ARRAY_TASK_ID}"
echo "offset     : ${ARRAY_OFFSET}"
echo "archive idx: ${ARCHIVE_INDEX}"
echo "archive dir: ${ARCHIVE_DIR}"
echo "out root   : ${OUT_ROOT}"
echo "py7zr check:"
conda run --no-capture-output -n s2s-hind python - <<'PY'
import py7zr
print("  py7zr", py7zr.__version__)
PY

$RUN "$SCRIPT" \
  --archive-dir "$ARCHIVE_DIR" \
  --out-root "$OUT_ROOT" \
  --array-index "$ARCHIVE_INDEX" \
  --expected-members 51 \
  --expected-steps 42 \
  --mark-complete-existing \
  --extract-into-existing

echo "end        : $(date)"
