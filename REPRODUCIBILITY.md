# Reproducibility Guide

This repository is a code companion for the India S2S benchmark. It does not
mirror restricted forecast archives or track manuscript/build artifacts.

## Environment

```bash
conda env create -f environment.yml
conda activate s2s-analysis
```

## Data Roots

The workflow expects provider datasets and local intermediate products outside
the Git checkout. The main overrides are:

```bash
export S2S_STORAGE_ROOT=/path/to/storage-root
export S2S_DATA_ROOT=/path/to/All_Model_Data
export S2S_OUTPUT_ROOT=/path/to/verification-outputs
export S2S_ERA5_CLIMATOLOGY=/path/to/era5_climatology.nc
export S2S_WEATHERBENCH2_ERA5_ZARR=/path/to/weatherbench2_era5.zarr
```

If `S2S_OUTPUT_ROOT` is not set, scripts write under:

```text
outputs/verification/
```

## Lightweight Checks

These checks exercise the package and metric formulas without loading the full
forecast archive:

```bash
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
```

These checks require local/provider data roots:

```bash
python scripts/00_check_foundation.py
python scripts/03_check_common_grid.py
python scripts/04_audit_model_usability.py
python scripts/05_scan_forecast_inits.py --no-sample-metadata
python scripts/08_run_smoke_metric_cases.py --case all
```

## Verification Runs

Example full workflow entry points:

```bash
python scripts/07_run_weekly_metrics_pipeline.py --season jjas2019 --run-label full
python scripts/07_run_weekly_metrics_pipeline.py --season jfm2026 --include-spire --run-label full_daily_spire --workers 4
```

Outputs are local/generated and intentionally ignored by Git.

## Archiving

For a permanent software release, archive this code repository together with
small derived CSV outputs needed for review, if their upstream data terms allow
redistribution. Do not redistribute raw provider data unless the relevant
licenses explicitly allow it.
