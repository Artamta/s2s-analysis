# Midterm thesis report — Improving Regional S2S Forecasting

Overleaf: upload `thesis_overleaf.zip` and compile with pdfLaTeX (also builds with Tectonic: `~/.local/bin/tectonic-0.16.9 main.tex`).

Cover fields (name, registration number, supervisor, expert) are set at the top of `main.tex`.

Style: plain academic, ~3 body pages, no line graphs, no abstract or publication list (own papers are ordinary citations).
Results are small booktabs tables, one coverage bar chart and two spatial figures built by `tools/make_thesis_figures.py`
(serif P052 font to match the Palatino text):
- `figures/map_selection.pdf` (built but no longer used): best system per IMD region, JJAS and OND, by lead (#315 `v21_results_cube.csv`).
- `figures/improvement.pdf`: CRPS, ACC and 90% coverage by lead week for raw FuXi-S2S, simple correction and calibration network
  (#297 `generated_numbers.tex` Diag* values, 2020-2024 JJAS vs IMD; per-week ACC exists only for raw and network).
- `figures/map_adapter.pdf`: adapter CRPS improvement for Weeks 1, 2, 4, 6 (#297 reviewed `spatial_fields.npz`, SHA-256 checked).
- `figures/map_forecast.pdf` (built but not used in the report): the 27 Sep 2026 s2s.scdlds.com issue, rainfall and T2M anomaly for Weeks 1–4
  (`clean/fuxi-dashboard/public/data/forecasts/ifs/20260927.json`).

Table numbers come from the #315/#297 camera-ready sources and `clean/INDIA-S2S-BENCH/paper/tables/headline.tex`.
A short appendix (after the references): A models, B data sets, C formulas, D all-system All-India ACC, E calibration network and lead-wise CRPS/coverage against IMD. Sources: #315
`tables/model_contract.tex`, provenance table, `v21_overall_{tp,t2m}.tex` and appendix; #297 appendix and `generated_numbers.tex`.
- `figures/map_regions.pdf` (built but no longer used): the four IMD homogeneous regions from `masks/imd_region_masks_0.25deg.nc` (state lists checked by
  overlaying the Survey of India state layer on the masks).
The old `skill_summary.pdf` and product screenshots are no longer used.

Logos in `logos/` are institutional marks (Ashoka, SCDLDS from the dashboard brand assets; IISER Pune with its baked-in checkerboard removed). Do not redistribute.
