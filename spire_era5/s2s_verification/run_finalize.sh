#!/usr/bin/env bash
# Finalize: regenerate all figures from corrected data, build main figure,
# assemble a clean paper_figures/ folder. Run after 09 recompute completes.
set -e
cd "$(dirname "$0")"

echo "=== [1/4] Regenerating P01–P15 from corrected data ==="
conda run -n s2s-hind python -u 12_master_publication_plots.py

echo "=== [2/4] Building consolidated main figure ==="
conda run -n s2s-hind python -u 13_main_figure.py

echo "=== [3/4] Assembling clean paper_figures/ ==="
PF=paper_figures
rm -rf "$PF"; mkdir -p "$PF"
# headline + companions (correct, consistent-baseline)
cp figures/paper/Fig1_main.png        "$PF/Fig1_main_maxT2m.png"
cp figures/paper/Fig1_main.pdf        "$PF/Fig1_main_maxT2m.pdf"
cp figures/pub/P02_anomaly_max.png    "$PF/Fig2_anomaly_maps_max.png"
cp figures/pub/P01_anomaly_mean.png   "$PF/Fig2b_anomaly_maps_mean.png"
cp figures/pub/P09_acc_skill_maps.png "$PF/Fig3_acc_skill_maps.png"
cp figures/pub/P10_acc_vs_lead.png    "$PF/Fig4_acc_vs_lead.png"
cp figures/pub/P06_scatter_max.png    "$PF/Fig5_scatter_max.png"
cp figures/pub/P08_scatter_initmean_max.png "$PF/Fig6_scatter_initmean_max.png"
cp figures/pub/P04_bias_max.png       "$PF/FigS1_bias_max.png"
cp figures/pub/P11_rmse_vs_lead.png   "$PF/FigS2_rmse_vs_lead.png"
cp figures/pub/P15_acc_rmse_heatmap.png "$PF/FigS3_acc_rmse_table.png"
cp figures/pub/P14_skill_dashboard.png "$PF/FigS4_skill_dashboard.png"
cp SUMMARY.md REVIEWER_CAVEATS.md     "$PF/"

echo "=== [4/4] Done. paper_figures/ contents: ==="
ls -la "$PF"
