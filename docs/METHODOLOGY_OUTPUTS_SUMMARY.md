# Methodology and Output Summary

This is the verification-pipeline contract for the current clean branch.

## Output Layout

All generated analysis products live under:

```text
outputs/verification/
```

Common inventory and logs:

```text
common/inventory/                 matched-init and model availability CSVs
common/logs/slurm/                scheduler stdout/stderr logs
```

Per season:

```text
<season>/02_processed/matched_init/      comparable init-date CSVs
<season>/03_metrics/<run_label>/         main metric CSVs
<season>/04_figures/<run_label>/test_plots/
                                         quick-look diagnostic PNGs
<season>/05_tables/<run_label>/          compact summary CSVs for checking/plotting
<season>/07_logs/slurm/                  runtime logs from SLURM scripts
<season>/08_final/                       reserved for final publication-ready outputs
```

The model-specific `by_model/` folders are kept only for supported models:
DLESyM, ECMWF, FuXi, NCEP, and UKMO. GenCast and NeuralGCM are deliberately
excluded from this benchmark workflow.

## Common Verification Method

- Grid: India box on the 1.5 degree common grid.
- Regions: All India plus the four IMD homogeneous regions.
- Lead windows: Week 1 through Week 6, each a 7-day mean.
- Lead-day convention: lead day `k` verifies calendar date `init + k days`.
- Deterministic forecast: ensemble mean or SPIRE mean.
- MME: arithmetic mean of available model ensemble means on the common grid.
- TP units: `mm day-1`.
  FuXi native TP is a 24-hour hourly average/rate and is multiplied by 24
  before verification.
- Z500 units: `gpm`.
- T2M units: `K`.

Climatology:

- TP/rainfall anomalies and events follow the selected truth source:
  IMD truth uses IMD 1991-2020 daily rainfall climatology, while ERA5 truth
  uses ERA5 1990-2019 day-of-year TP climatology converted from m to mm/day.
- Z500/T2M anomalies: ERA5 day-of-year climatology.

Truth:

- JJAS2019 TP: IMD daily rainfall 2019 by default, or ERA5 WeatherBench2 TP
  with `--truth-source era5`.
- JJAS2019 Z500/T2M: WeatherBench2 ERA5.
- JFM2026 TP/T2M: daily ERA5 files under
  `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`.
- JFM2026 Z500: ERA5 CDS daily files under the FuXi ground-truth directory:
  `/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth`.
- Current JFM2026 ERA5 TP coverage from the new daily file is 2025-12-25 to
  2026-05-10. Current JFM2026 ERA5 T2M coverage is 2026-01-01 to 2026-05-10.
  Current Z500 coverage in the FuXi ground-truth directory is 2026-01-01 to
  2026-03-31.
- `--truth-source both` writes separate ERA5-truth and IMD-truth runs where
  both are available. JFM2026 IMD rainfall is not available yet because
  `imd_rain_2026.nc` is absent.

## Metrics/Tables Produced

`deterministic_weekly.csv`

- ACC
- RMSE
- Bias, kept as a diagnostic for wet/dry, warm/cold, and height-error sign.
  It is not the main ranking metric.
- MAE
- MSE skill vs climatology
- Climatology RMSE
- Forecast/truth/climatology area means

`probabilistic_weekly.csv`

- CRPS
- CRPS climatology reference
- CRPSS vs climatology
- Ensemble/spread mean
- Ensemble-mean RMSE
- Spread-skill ratio

`brier_weekly.csv`

- TP event Brier score
- TP event Brier skill vs climatology
- Base rate
- Current events: `tp_gt_1_mm_day`, `tp_gt_10_mm_day`

`reliability_weekly.csv`

- All-India reliability bins for TP events
- Forecast probability bin mean
- Observed frequency
- Bin count

`scatter_area_weekly.csv`

- Region-mean forecast/truth/climatology values
- Region-mean forecast/truth anomalies
- Region-mean error
- This is the easiest table for scatter plots across init dates, regions, weeks, and models.

`scatter_grid_weekly.csv`

- Grid-cell forecast/truth/climatology values inside the India mask
- Grid-cell anomalies and errors
- This is for spatial scatter/density plots and map diagnostics.

`model_status.csv`

- Opened/scored/skipped status for each init/model/variable/week.
- This is the first file to check when a run has fewer rows than expected.

## JJAS2019 Outputs

Strict all-model common-init run:

- Set: `all_usable_models`
- Models: DLESyM, ECMWF, UKMO, NCEP, FuXi
- Common init dates: 17 Thursday inits from 2019-06-06 to 2019-09-26
- Variables: TP for ECMWF, UKMO, NCEP, FuXi; Z500 for all five models
- Truth: IMD or ERA5 for TP; ERA5 for Z500/T2M
- Metrics: deterministic, probabilistic, Brier/reliability for TP, scatter area/grid

Operational rainfall run:

- Set: `operational_models`
- Models: ECMWF, UKMO, NCEP
- Variable: TP
- Common init dates: 35 Monday/Thursday inits
- Truth: IMD 2019 rainfall by default; optional ERA5 WeatherBench2 rainfall
  with `--truth-source era5`; both can be run with `--truth-source both`.
- Climatology: follows truth source, IMD with IMD truth and ERA5 with ERA5 truth.
- Metrics: deterministic, probabilistic, Brier, reliability, scatter area/grid

Upper-air run:

- Set: `delysm_operational`
- Models available in the clean loader: DLESyM, ECMWF, UKMO, NCEP
- Primary smoke currently tests DLESyM and UKMO for Z500.
- Truth: WeatherBench2 ERA5
- Climatology: ERA5 day-of-year climatology
- Metrics: deterministic, probabilistic, scatter area/grid

T2M:

- Weekly-capable clean path is DLESyM T2M.
- ECMWF/UKMO T2M files in this tree have only one forecast lead, so they are skipped by the default weekly runner.

## JFM2026 Outputs

SPIRE novelty run:

- SPIRE source: `/storage/raj.ayush/archive/All_Model_Data/models/spire/data/s2s-research.zarr`
- Frequency: 90 daily JFM2026 initializations
- Product: processed mean/stddev, not raw members
- Scoring: Gaussian mean/std probabilistic verification

TP:

- Models: SPIRE, ECMWF, UKMO, NCEP, FuXi
- DLESyM has no TP in this tree.
- Truth: ERA5 daily TP from `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/era5_daily_tp.nc`
- Climatology: ERA5 1990-2019 day-of-year TP, converted from m to mm/day
- Metrics: deterministic, probabilistic, Brier, reliability, scatter area/grid

Z500:

- Models: SPIRE, DLESyM, ECMWF, UKMO, NCEP, FuXi
- Truth: ERA5 CDS daily ground-truth files
- Climatology: ERA5 day-of-year
- Metrics: deterministic, probabilistic, scatter area/grid

T2M:

- Models with weekly-capable clean path: SPIRE, DLESyM, FuXi
- ECMWF/UKMO T2M have only one forecast lead in this tree.
- Truth: ERA5 daily T2M from `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/era5_daily_t2m.nc`
- Metrics: deterministic, probabilistic, scatter area/grid

## Commands

Smoke suite:

```bash
python scripts/08_run_smoke_metric_cases.py --case all
```

Make plots for one run:

```bash
python scripts/09_make_test_plots.py --season jfm2026 --run-label test_tp_spire
```

When `scatter_grid_weekly.csv` is available, the same plot script writes
spatial mean-error and RMSE maps under:

```text
<season>/04_figures/<run_label>/test_plots/spatial_maps/
```

JJAS TP full-ish run:

```bash
python scripts/07_run_weekly_metrics_pipeline.py \
  --season jjas2019 \
  --set-name operational_models \
  --variables tp \
  --models ecmwf ukmo ncep \
  --truth-source both \
  --run-label full_jjas_tp \
  --workers 4
```

JFM daily SPIRE run:

```bash
python scripts/07_run_weekly_metrics_pipeline.py \
  --season jfm2026 \
  --set-name all_usable_models \
  --include-spire \
  --run-label full_daily_spire \
  --workers 4
```

Use `--no-grid-scatter` if a full run should skip the larger grid-cell scatter table.

## SLURM Smoke Jobs

```bash
sbatch slurm/smoke_suite.sbatch
sbatch slurm/smoke_jjas_tp.sbatch
sbatch slurm/smoke_jjas_common17_fuxi.sbatch
sbatch slurm/smoke_jfm_daily_spire_mp.sbatch
```

Full priority-partition runs:

```bash
sbatch slurm/run_jjas2019_full_gpu_prio.sbatch
sbatch slurm/run_jfm2026_full_gpu_prio.sbatch
```

SLURM stdout/stderr:

```text
outputs/verification/common/logs/slurm/
```

Runtime logs:

```text
outputs/verification/jjas2019/07_logs/slurm/
outputs/verification/jfm2026/07_logs/slurm/
```
