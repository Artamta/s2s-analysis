# Data Download Plan

Goal: download organized S2S data for ECMWF, UKMO, NCEP, and CMA, with CNRM as
a weekly secondary benchmark. See `MODEL_SELECTION.md` for the selection audit
and `SMOKE_TEST_20240620.md` for the successful five-model day-1 test.

The production source of truth is `../config/all_season_dates_2020_2025.csv`.
The older `2020_2024` calendar remains frozen for the already-running array.
See `PRODUCTION_DOWNLOAD.md` for request counts, storage layout, launch, resume,
monitoring, and QC.

## Forecast vs Reforecast

Use these names consistently:

- `forecast`: operational forecast for a real initialization date/year.
- `reforecast`: hindcast/reforecast archive used for climatology and skill
  verification across historical years.

Do not mix these in the same directory.

Canonical heavy-data layout on `/storage`:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/<provider>/
    forecast/<season-or-year>/
      <variable>/<YYYYMMDD>_<cf|pf>.grib
    reforecast/<season-or-year>/
      <variable>/<MMDD>_<cf|pf>.<grib|nc>
  standardized/<provider>/<experiment>/
  manifests/<provider>/
  logs/<provider>/
```

Examples:

```text
raw/ecmwf/forecast/jjas2020/tp/20200601_pf.grib
raw/ecmwf/forecast/jjas2020/t2m/20200601_cf.grib
raw/ukmo/forecast/jjas2025/tp/20250620_pf.grib
```

## Minimum Variables

For the current India S2S benchmark:

```text
tp      total precipitation
t2m     2 m temperature
```

Useful process-science additions:

```text
u850    low-level monsoon flow / MISO
u200    tropical easterly jet / Webster-Yang shear
v850    moisture transport support
msl     monsoon trough
```

## Manifests

Every provider should write:

```text
manifests/requests.jsonl
manifests/inventory.csv
checksums/sha256_<run_label>.txt
```

Minimum manifest columns:

```text
provider,product,season,init_date,mmdd,forecast_type,variable,level,
lead_start,lead_end,file_path,size_bytes,status,request_hash,timestamp
```

## Physics-Model First Phase

The canonical operational years are `2020-2025`. Use exact common all-season
initializations for ECMWF, UKMO, NCEP, and CMA. Download CNRM on the exact
weekly subset. Every file retains its control or all native perturbed members
and the complete lead-day 1-42 window.

ECMWF `t2m` is a daily average and must be requested with interval steps such as
`0_24`, `24_48`, and `48_72`; endpoint-only steps return day 1 only. FuXi-S2S
also predicts daily means and its ERA5 inputs must come from daily statistics
derived from all 24 hourly values. For a physics issue date D at 00 UTC, the
strict FuXi run uses complete days D-2 and D-1; its lead day 1 then covers the
same D-to-D+1 period as the physics forecast. Both `tp` and `t2m` can use a
daily-statistic comparison only after this information-cutoff, unit, and
valid-period alignment.

Keep all native members in raw files. Ensemble-size matching belongs in the
standardized analysis layer, not in the downloader.

Generate all plans with:

```bash
python clean/data-download/scripts/plan_s2s_downloads.py --phase all --write
```

See `COMPARABILITY.md` for the scientific contract and `DRY_RUN_REPORT.md` for
availability, request counts, storage estimates, and the production launch gate.
