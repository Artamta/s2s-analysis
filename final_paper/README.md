# Final Paper Analysis

Clean, reviewable code for the paper-version S2S study over India.

This folder is intentionally separate from `final_analysis/`. The old folder is
the working lab notebook; this folder is the reproducible paper workspace.

## Scope

The paper track here has two case-study components:

- **JFM2026:** operational snapshot over India, using SPIRE, FuXi-S2S, ECMWF and
  supporting baselines where available. SPIRE now uses the daily
  `s2s-research.zarr` archive with 90 JFM initializations.
- **JJAS2019:** Indian monsoon case study, using the comparable operational
  systems plus DLESyM where variables are available.

For this version, anomalies and event indices use truth-matched observational
or reanalysis climatology only: IMD 1991-2020 climatology when rainfall truth is
IMD, ERA5 1990-2019 climatology when rainfall truth is ERA5, and ERA5 common
climatology for Z500/T2M. Model-own climatologies are deliberately out of scope
until the case-study pipeline is fully verified.

## Layout

```text
final_paper/
├── code/s2s_paper/       shared Python utilities
├── scripts/              small command-line checks and analysis entry points
├── masks/                copied IMD homogeneous-region masks
├── outputs/              generated tables and figures
└── docs/                 study decisions and notes
```

## Step 1 Check

Run this first:

```bash
cd final_paper
python scripts/00_check_foundation.py
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
python scripts/03_check_common_grid.py
python scripts/04_audit_model_usability.py
python scripts/05_scan_forecast_inits.py --no-sample-metadata
python scripts/06_open_comparable_forecasts.py --season jjas2019 --set-name delysm_operational --variable z500 --max-inits 1
python scripts/08_run_smoke_metric_cases.py --case all
```

This checks the path registry, IMD mask files, IMD climatology files, and the key
JFM2026/JJAS2019 data locations without doing any heavy computation. It also
checks exact hand-value formulas for ACC, RMSE, bias, MAE, CRPS, SSR, Brier
score, Brier skill score and reliability masking. The common-grid check samples
representative model/truth fields and verifies that they regrid to the same
target grid. The model usability audit writes CSV/Markdown reports under
`outputs/` showing model-variable availability, overlapping initialization
dates, lead coverage, truth coverage and required unit conversions.

The forecast init scanner writes matched-init tables under
`outputs/s2s_paper_outputs/<season>/02_processed/matched_init/`. The comparable
forecast opener verifies that forecasts can be opened with dimensions
`member, lead, lat, lon` for ACC/RMSE computation.

## Weekly Metrics Pipeline

Main entry point:

```bash
python scripts/07_run_weekly_metrics_pipeline.py --season jjas2019 --run-label full
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire --workers 4
```

Rainfall truth can be selected explicitly:

```bash
python scripts/07_run_weekly_metrics_pipeline.py --season jjas2019 --variables tp --truth-source both --run-label full_jjas_tp
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --variables tp --truth-source era5 --include-spire --run-label full_jfm_tp_era5
```

Focused smoke cases:

```bash
python scripts/08_run_smoke_metric_cases.py --case jjas_tp
python scripts/08_run_smoke_metric_cases.py --case jfm_tp_spire
python scripts/09_make_test_plots.py --season jfm2026 --run-label test_tp_spire
```

Outputs are written under
`outputs/s2s_paper_outputs/<season>/03_metrics/<run_label>/`:

- `deterministic_weekly.csv`: ACC, RMSE, bias, MAE, MSE skill vs climatology.
- `probabilistic_weekly.csv`: CRPS, CRPSS vs climatology, spread, spread-skill ratio.
- `brier_weekly.csv`: TP exceedance Brier scores and Brier skill.
- `reliability_weekly.csv`: All-India reliability bins for TP exceedance events.
- `scatter_area_weekly.csv`: region-mean forecast/truth pairs for scatter plots.
- `scatter_grid_weekly.csv`: grid-cell forecast/truth pairs inside the India mask.
- `model_status.csv`: opened/scored/skipped status and skip reasons.
- `run_metadata.json`: selected inits/models/variables and method notes.

SLURM smoke jobs:

```bash
sbatch slurm/smoke_suite.sbatch
sbatch slurm/smoke_jjas_tp.sbatch
sbatch slurm/smoke_jfm_daily_spire_mp.sbatch
```

See [docs/METHODOLOGY_OUTPUTS_SUMMARY.md](docs/METHODOLOGY_OUTPUTS_SUMMARY.md)
for the full method, metric, output and log layout.
