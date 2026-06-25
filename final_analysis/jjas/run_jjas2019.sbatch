#!/bin/bash
#SBATCH --job-name=jjas2019
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/jjas2019_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas/logs/jjas2019_%j.err
#SBATCH --partition=GPU-AI_prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00

# Unattended JJAS-2019 pipeline: FuXi extract -> verify (FuXi+ECMWF, dual basis)
# -> all plots. Each stage logs and continues even if a later stage fails.
# Submit:  sbatch run_jjas2019.sbatch
set -uo pipefail
JJAS=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jjas
ANALYSIS=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/analysis
JPLOTS=/home/raj.ayush/s2s/s2s_anlysis/final_analysis/jfm2026/plots
RES=/storage/raj.ayush/s2s_final_data/jjas/results_2019_1.5deg
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN="conda run --no-capture-output -n s2s-hind python -u"

echo "host $(hostname)  start $(date)"

echo "===== STAGE 1: FuXi extraction (JJAS 2019, ~26 inits, slow) ====="
$RUN "$JJAS/preprocess_fuxi.py" --start 20190601 --end 20190831 || echo "WARN: FuXi extraction had errors (continuing)"

echo "===== STAGE 2: verification (FuXi + ECMWF, era5 + model_own) ====="
cd "$JJAS"
$RUN "$JJAS/run_verify.py" --year 2019 --workers 13 || { echo "FATAL: verify failed"; exit 1; }

echo "===== STAGE 3: figures ====="
$RUN "$JPLOTS/make_plots.py"   --results "$RES" || echo "WARN: make_plots failed"
$RUN "$JPLOTS/meeting_figs.py" --results "$RES" || echo "WARN: meeting_figs failed"
$RUN "$ANALYSIS/monthwise.py"  --results "$RES" --out "$JJAS/plots/figs/monthwise_2019" || echo "WARN: monthwise failed"
$RUN "$JJAS/plots/monsoon_maps.py" --years 2019 --ecmwf-year 2019 || echo "WARN: monsoon_maps failed"

echo "===== DONE $(date) ====="
echo "results: $RES"
echo "figures: $JJAS/plots/figs/ , $JPLOTS/figs/"
