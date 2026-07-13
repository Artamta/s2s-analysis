# Data Download Plan

Goal: download organized S2S data for ECMWF, UKMO, and NCEP.

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
      <variable>/<YYYYMMDD>_<cf|pf>.<grib|nc>
    reforecast/<season-or-year>/
      <variable>/<MMDD>_<cf|pf>.<grib|nc>
  standardized/<provider>/<experiment>/
  manifests/<provider>/
  logs/<provider>/
```

Examples:

```text
raw/ecmwf/forecast/jjas2020/tp/20200601_pf.nc
raw/ecmwf/forecast/jjas2020/t2m/20200601_cf.nc
raw/ukmo/forecast/jjas2025/tp/20250620_pf.nc
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

## ECMWF First Phase

The missing operational years are `2020-2024`. Use the 35 FuXi JJAS target
starts in `../config/comparable_dates_2019_2026.csv`, paired to the minimum-lag
one-to-one ECMWF schedule that preserves the full valid-date window.
Download `tp` and `t2m`, control plus all perturbed members. ECMWF requests
extend beyond day 42 only where a shifted start needs later leads for the common
42-day window. This remains 35 initializations and 140 resumable requests per
year, or 700 requests across five years.

ECMWF `t2m` is a daily average and must be requested with interval steps such as
`0-24`, `24-48`, and `48-72`; endpoint-only steps return day 1 only. FuXi `t2m`
is a daily 00 UTC snapshot, so temperature comparisons must disclose that
temporal-statistic difference. `tp` is the strict like-for-like benchmark.

Keep all native members in raw files. Ensemble-size matching belongs in the
standardized analysis layer, not in the downloader.

See `COMPARABILITY.md` for the ECMWF/FuXi contract and
`ecmwf/README.md` for launch instructions.
