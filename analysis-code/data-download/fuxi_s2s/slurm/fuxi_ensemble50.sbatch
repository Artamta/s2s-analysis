#!/bin/bash
#SBATCH --job-name=fuxi_ens50
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/ens50_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/ens50_%j.err

# ── Config ────────────────────────────────────────────────────────────────────
FUXI_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S"
PYTHON="/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python"

DATE="${ENS_DATE:-20260617}"
MEMBERS="${MEMBERS:-50}"
INPUT_FILE="/storage/raj.ayush/All_Model_Data/fuxi/test/inputs/${DATE}/input.nc"
OUT_DIR="/storage/raj.ayush/All_Model_Data/fuxi/test/raw/${DATE}"

export LD_LIBRARY_PATH="/home/raj.ayush/.conda/envs/fuxi_s2s/lib:$LD_LIBRARY_PATH"

echo "================================================================"
echo "Job         : FuXi-S2S  ${MEMBERS}-member perturbed ensemble"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Init date   : $DATE"
echo "Members     : $MEMBERS   (member 00 = control, rest = IC-perturbed)"
echo "Steps       : 42  (42-day forecast)"
echo "Input       : $INPUT_FILE"
echo "Output      : $OUT_DIR"
echo "Disk est.   : ~$(python3 -c "print(f'{int($MEMBERS)*0.37:.0f}') 2>/dev/null || echo '~18') GB"
echo "Time est.   : ~$(python3 -c "print(f'{int($MEMBERS)*12/60:.0f}') 2>/dev/null || echo '~10') min"
echo "Start time  : $(date)"
echo "================================================================"

mkdir -p "$OUT_DIR"
cd "$FUXI_DIR"

$PYTHON inference_ensemble.py \
    --model        "$FUXI_DIR/model/fuxi_s2s.onnx" \
    --input        "$INPUT_FILE" \
    --device       cuda \
    --total_step   42 \
    --total_member "$MEMBERS" \
    --seed         42 \
    --save_dir     "$OUT_DIR"

EXIT_CODE=$?
echo "================================================================"
echo "End time  : $(date)"
echo "Exit code : $EXIT_CODE"
echo "Output dir: $(du -sh ${OUT_DIR}/member/ 2>/dev/null | cut -f1) used"
echo "================================================================"
exit $EXIT_CODE
