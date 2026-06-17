#!/bin/bash
# Run all presentation plotting scripts in order.
# All figures are saved to plots_results_pres/

set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "  S2S V3 Presentation Plots"
echo "  Output: plots_results_pres/"
echo "================================================================"
echo

echo "--- 1/5  Scatter: forecast vs observed (weekly) ---"
python3 plot_scatter_weekly.py

echo
echo "--- 2/5  Bias ---"
python3 plot_bias.py

echo
echo "--- 3/5  PCC / ACC ---"
python3 plot_pcc.py

echo
echo "--- 4/5  RMSE ---"
python3 plot_rmse.py

echo
echo "--- 5/5  CRPS / CRPSS ---"
python3 plot_crps.py

echo
echo "--- 6/6  Brier Skill Score ---"
python3 plot_brier.py

echo
echo "================================================================"
echo "  All done.  Figures in plots_results_pres/"
ls plots_results_pres/*.png 2>/dev/null | wc -l | xargs -I{} echo "  {} PNG files written"
echo "================================================================"
