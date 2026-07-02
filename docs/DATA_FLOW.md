# Data Flow

This is the current benchmark data flow before the full seasonal runners are
written.

## Storage Layer

Primary storage root:

`/storage/raj.ayush/All_Model_Data`

Current model/truth roots registered in `src/s2s_benchmark/paths.py`:

- ECMWF: `/storage/raj.ayush/All_Model_Data/ecmwf`
- FuXi: `/storage/raj.ayush/All_Model_Data/fuxi`
- UKMO: `/storage/raj.ayush/All_Model_Data/ukmo`
- DELYSM: `/storage/raj.ayush/All_Model_Data/delysm`
- DELYSM QC: `/storage/raj.ayush/All_Model_Data/delysm_qc`
- NCEP: `/storage/raj.ayush/All_Model_Data/ncep`
- SPIRE daily JFM archive:
  `/storage/raj.ayush/archive/All_Model_Data/models/spire/data/s2s-research.zarr`
- ERA5/IMD truth and climatology:
  `/storage/raj.ayush/All_Model_Data/ground_truth`
  and `/storage/raj.ayush/All_Model_Data/climatology_common`
- ERA5 daily TP/T2M truth:
  `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`

## Normalization Layer

All model-specific loading should end as an xarray `DataArray` with spatial
coordinates called `lat` and `lon`, or `latitude` and `longitude`.

Expected model-specific normalization:

- ECMWF/UKMO NetCDF:
  - Files are organized by season, variable directory, init date and `cf/pf`.
  - Surface variables use `t2m` or `tp`.
  - Z500 uses `z/500/*.nc` with data variable `gh`.
  - Ensemble dimension is usually `number`.
- FuXi JFM2026 combined:
  - Variable is stored as `forecast(member, lead_time, channel, lat, lon)`.
  - 76-channel mapping: `z500` index 5, `t2m` index 65, `tp` index 75.
  - Native TP is a 24-hour hourly average/rate and is multiplied by 24 to
    convert to `mm day-1` before weekly aggregation.
- FuXi JJAS compact:
  - Variable is stored as `forecast(member, lead_time, channel, lat, lon)`.
  - Channel labels are named, currently including `tp` and `z500`.
  - The same TP conversion is applied: native values are multiplied by 24 to
    convert to `mm day-1`.
- DELYSM:
  - Forecasts live under `init/india/forecast.zarr`.
  - Stores are Zarr with arrays shaped
    `ensemble, time, lead_time, lat, lon`.
  - DELYSM has `t2m` and `z500` for this study, but no `tp`.
  - DELYSM stores include negative lead times; verification should use
    `lead_time >= 0`.
- SPIRE:
  - Daily JFM2026 archive is `s2s-research.zarr/mean_stddev`.
  - It provides processed mean/stddev fields, not raw members.
  - Reference times cover Jan 1-Mar 31, 2026 daily; lead steps are 1-46 days.
- Truth:
  - JFM2026 TP/T2M truth uses daily ERA5 files under
    `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`.
  - JFM2026 Z500 truth uses ERA5 files under
    `/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth`.
  - JJAS2019 rainfall truth can use IMD
    `/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/netcdf/imd_rain_2019.nc`.
  - JJAS2019 rainfall truth can also use WeatherBench2 ERA5 TP with
    `--truth-source era5`.

## Common Grid Layer

All fields then go through:

`s2s_benchmark.grid.to_grid(field, GridSpec(...))`

Default grid:

- lat: 38 to 5 by 1.5 degrees, descending
- lon: 65 to 100 by 1.5 degrees, ascending
- shape: 22 x 24

Region masking happens after regridding:

`s2s_benchmark.grid.apply_region(field, region, dgrid=...)`

Available masks:

- 1.5 degree
- 0.5 degree
- 0.25 degree

## Verification Layer

After model/truth/climatology are on the same grid:

- deterministic scores use ACC, RMSE, bias and MAE
- probabilistic scores use CRPS, SSR, Brier/BSS and reliability bins
- rainfall anomalies and rainfall events follow `--truth-source`: IMD truth
  uses IMD 1991-2020 climatology, ERA5 truth uses ERA5 1990-2019 TP climatology
- Z500/T2M anomalies use the common ERA5 climatology

## Current Grid Audit

The audit script is:

`scripts/03_check_common_grid.py`

It samples representative fields from ECMWF, UKMO, FuXi, DELYSM, ERA5 truth and
IMD truth, then checks that each can be regridded to the same target grid.

Latest check passed for:

- ECMWF JJAS2019: T2M, TP, Z500
- UKMO JJAS2019: T2M, TP, Z500
- FuXi JFM2026: T2M, TP, Z500
- FuXi JJAS2019: TP, Z500
- DELYSM JJAS2019: T2M, Z500
- ERA5 truth JFM2026: T2M, TP, Z500
- IMD truth JJAS2019: TP

Detailed CSV output is written to:

`outputs/common_grid_check.csv`

The CSV is ignored by git as a generated output.
