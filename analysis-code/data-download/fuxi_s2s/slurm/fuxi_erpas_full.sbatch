#!/bin/bash
#SBATCH --job-name=fuxi_full
#SBATCH --partition=gpu_prio
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_full_%j.log
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/logs/erpas_full_%j.err

PYTHON="/home/raj.ayush/.conda/envs/s2s-hind/bin/python"
SCRIPT="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/plot_erpas_style.py"

DATE="${ERPAS_DATE:-20260617}"
OUT_DIR="/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/erpas_style/${DATE}_ens50"
CLIMO_NC="/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc"

echo "================================================================"
echo "Job    : FuXi-S2S  FULL 24-mode ERPAS suite  (50 members)"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $DATE"
echo "Out    : $OUT_DIR"
echo "Climo  : $CLIMO_NC"
echo "Start  : $(date)"
echo "================================================================"
echo ""
echo "Modes (24 total):"
echo "  GIFs  : prec vort divg rh igpp temp_gif temp_anom hw_gif olr tcwv sst"
echo "  PNGs  : rf_weekly rf_w6 rf_anom igpp_weekly tmax_actual tmax_anom"
echo "          temp_weekly temp_anom_weekly hw"
echo "  Extra : hovmoller z500_anom wind850 shear"
echo "================================================================"

mkdir -p "$OUT_DIR"

# Ordered to front-load GIFs (heaviest), then PNGs
ALL_MODES=(
    prec vort divg rh igpp
    temp_gif temp_anom hw_gif
    olr tcwv sst
    rf_weekly rf_w6 rf_anom igpp_weekly
    tmax_actual tmax_anom temp_weekly temp_anom_weekly hw
    hovmoller z500_anom wind850 shear
)

CLIMO_ARG=""
[ -f "$CLIMO_NC" ] && CLIMO_ARG="--climo_nc $CLIMO_NC"

TOTAL=${#ALL_MODES[@]}
IDX=0

for MODE in "${ALL_MODES[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "━━━ [$IDX/$TOTAL]  MODE: $MODE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  $(date +%H:%M:%S)"
    $PYTHON "$SCRIPT" \
        --date    "$DATE" \
        --out_dir "$OUT_DIR" \
        --steps   42 \
        --fps     4 \
        --mode    "$MODE" \
        $CLIMO_ARG
    echo "    exit=$?"
done

echo ""
echo "================================================================"
echo "All $TOTAL modes complete"
echo "Output : $OUT_DIR"
echo "Files  : $(ls $OUT_DIR | wc -l)"
echo "Size   : $(du -sh $OUT_DIR | cut -f1)"
echo "End    : $(date)"
echo "================================================================"
