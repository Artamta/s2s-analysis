# Weekly Metrics Pipeline

This note documents the first clean scoring pipeline in this repository.

## Output Structure

All metric tables are written to:

```text
outputs/verification/<season>/03_metrics/<run_label>/
```

The main files are:

- `deterministic_weekly.csv`: ACC, RMSE, bias, MAE, MSE skill vs climatology.
- `probabilistic_weekly.csv`: CRPS, CRPSS vs climatology, spread, spread-skill ratio.
- `brier_weekly.csv`: TP exceedance Brier score and Brier skill.
- `reliability_weekly.csv`: All-India reliability bins for TP events.
- `scatter_area_weekly.csv`: region-mean forecast/truth pairs for simple scatter plots.
- `scatter_grid_weekly.csv`: grid-cell forecast/truth pairs inside the India mask; this also feeds spatial map diagnostics.
- `model_status.csv`: opened/scored/skipped status with reasons.
- `run_metadata.json`: selected inits/models/variables, grid, and method notes.

## Seasons and Init Sets

JJAS2019:

- `all_usable_models`: DLESyM, ECMWF, UKMO, NCEP, FuXi on 17 common Thursday inits.
- `operational_models`: ECMWF, UKMO, NCEP on 35 common Monday/Thursday inits.
- `delysm_operational`: DLESyM plus ECMWF, UKMO, NCEP on 33 common inits.
- `ai_models_present`: DLESyM and FuXi on the same 17 Thursday inits.
- TP verification can use IMD daily rainfall for 2019 or ERA5 from
  WeatherBench2. Use `--truth-source imd`, `--truth-source era5`, or
  `--truth-source both`.
- Z500/T2M verification uses WeatherBench2 ERA5.

JFM2026:

- `all_usable_models`: DLESyM, ECMWF, UKMO, NCEP, FuXi on 90 daily inits.
- `--include-spire`: adds SPIRE from the daily `s2s-research.zarr` archive; it has the same 90 daily JFM2026 init dates.
- JFM TP/T2M truth uses the daily ERA5 files under
  `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`.
- JFM Z500 truth currently uses the ERA5 CDS daily files under the FuXi
  ground-truth directory.

## Units and Aggregation

Weekly windows are lead days 1-7, 8-14, ..., 36-42, with lead day `k` valid on
`init + k days`.

TP:

- ECMWF, UKMO, and NCEP TP are cumulative `kg m-2`; weekly totals are end-minus-start differenced and divided by days to make `mm day-1`.
- FuXi combined TP is a 24-hour hourly average/rate and is multiplied by 24 to make `mm day-1`.
- SPIRE precipitation amount is `kg m-2` over a 24-hour period and is treated as `mm day-1`.
- DLESyM has no TP in the available forecast tree.
- TP climatology follows the selected TP truth source. IMD truth uses IMD
  1991-2020 daily rainfall climatology. ERA5 truth uses ERA5 1990-2019
  day-of-year TP climatology converted from m to mm/day.

Z500:

- DLESyM and FuXi Z500 are converted from geopotential to gpm using `/9.80665`.
- ECMWF, UKMO, NCEP `gh`, SPIRE geopotential height, and JFM truth are treated as gpm.
- ERA5 climatology Z500 is converted from `m2 s-2` to gpm using `/9.80665`.

T2M:

- DLESyM, FuXi, and SPIRE have weekly-capable T2M.
- ECMWF/UKMO T2M files in this tree contain only one forecast lead, so the default weekly runner skips them because `--min-leads-for-mean` defaults to 2.

## Probabilistic Scores

Raw ensembles use finite-member CRPS and sample spread after regridding to the
common India grid. SPIRE is a mean/stddev summary product, so it is scored as a
Gaussian forecast. For weekly SPIRE spread, the runner averages daily stddev
over the weekly window, matching the older analysis code.

## Parallel Runs

The runner parallelizes by init date:

```bash
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire --workers 4
```

Use `--no-grid-scatter` when you only need the smaller region-mean scatter table.
Spatial map plots require `scatter_grid_weekly.csv`, so do not use
`--no-grid-scatter` for runs where maps are needed.

TP events currently use:

- `tp_gt_1_mm_day`
- `tp_gt_10_mm_day`

The Brier climatology reference follows the selected TP truth source. IMD truth
uses IMD climatological mean/stddev with a Gaussian exceedance approximation;
ERA5 truth currently falls back to the observed base-rate reference because the
ERA5 climatology file has a daily mean but not a daily TP stddev.

## Smoke Tests

Run all focused smoke cases:

```bash
python scripts/08_run_smoke_metric_cases.py --case all
```

Individual examples:

```bash
python scripts/08_run_smoke_metric_cases.py --case jjas_tp
python scripts/08_run_smoke_metric_cases.py --case jjas_tp_common17_fuxi
python scripts/08_run_smoke_metric_cases.py --case jfm_tp_spire
python scripts/08_run_smoke_metric_cases.py --case jfm_t2m_spire_delysm_fuxi
```

Compare ERA5 and IMD rainfall truth where both are available:

```bash
python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_models \
  --variables tp \
  --models ecmwf ukmo ncep \
  --truth-source both \
  --run-label full_jjas_tp
```

Run the strict five-model JJAS common-init set with FuXi:

```bash
python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name all_usable_models \
  --variables tp z500 \
  --models delysm ecmwf ukmo ncep fuxi \
  --truth-source imd \
  --run-label full_jjas_common17_fuxi \
  --workers 4
```

Make plots for a metrics run:

```bash
python scripts/09_make_test_plots.py --season jjas2019 --run-label full_jjas_common17_fuxi
```

When grid scatter exists, the plot script also writes spatial diagnostics:

- `spatial_maps/spatial_mean_error_<variable>_weekN.png`
- `spatial_maps/spatial_rmse_<variable>_weekN.png`

Use `--no-spatial-maps` to skip map generation.

Full priority-partition SLURM runs:

```bash
sbatch slurm/run_jjas2019_full_gpu_prio.sbatch
sbatch slurm/run_jfm2026_full_gpu_prio.sbatch
```

Override worker count if the node is too memory- or I/O-bound:

```bash
sbatch --export=ALL,WORKERS=12 slurm/run_jjas2019_full_gpu_prio.sbatch
```
