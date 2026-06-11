#!/bin/bash
# run_all.sh — Master script to regenerate all Spire S2S paper figures
# Usage: bash run_all.sh [step]
#   step 1: Extract MJO/extended variables (requires ArrayLake + ERA5)
#   step 2: Generate deterministic figures (01-05)
#   step 3: Generate scatter/synthesis figures (06-10)
#   step 4: Generate MJO/extended figures (11-17)
#   no args: run all steps

set -e
CONDA_ENV="s2s-hind"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

run_step() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP $1: $2"
    echo "══════════════════════════════════════════════════════════════"
    conda run -n $CONDA_ENV python -u "scripts/$3"
}

STEP="${1:-all}"

if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
    run_step 1 "Extract MJO/extended variables" "extract_mjo_extended.py"
fi

if [ "$STEP" = "2" ] || [ "$STEP" = "all" ]; then
    run_step 2 "Deterministic verification (Figs 01-05)" "fig_deterministic.py"
fi

if [ "$STEP" = "3" ] || [ "$STEP" = "all" ]; then
    run_step 3 "Scatter & synthesis (Figs 06-10)" "fig_synthesis.py"
fi

if [ "$STEP" = "4" ] || [ "$STEP" = "all" ]; then
    run_step 4 "MJO & extended diagnostics (Figs 11-17)" "fig_mjo_extended.py"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL DONE ✓"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Figures in: $(pwd)/figures/"
find figures/ -name "*.png" -o -name "*.pdf" | sort | while read f; do
    sz=$(du -h "$f" | cut -f1)
    echo "  $sz  $f"
done
