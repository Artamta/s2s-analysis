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

For pressure-level variables:

```text
raw/<provider>/<forecast|reforecast>/<season>/<variable>/<level>/<date>_<cf|pf>.<ext>
```

Examples:

```text
raw/ecmwf/reforecast/jjas/tp/0620_pf.grib
raw/ecmwf/reforecast/jjas/z/500/0620_cf.grib
raw/ukmo/forecast/jjas2025/tp/20250620_pf.nc
raw/ncep/forecast/jjas2025/z/500/20250620_cf.grib
```

## Minimum Variables

For the current India S2S benchmark:

```text
tp      total precipitation
z500    500 hPa geopotential height / height
```

Useful process-science additions:

```text
u850    low-level monsoon flow / MISO
u200    tropical easterly jet / Webster-Yang shear
v850    moisture transport support
msl     monsoon trough
t2m     temperature, if provider lead coverage is usable
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

The missing operational years are `2020-2024`. Download Monday/Thursday JJAS
initializations, `tp` and `z500`, control plus all perturbed members, daily lead
days `1-42`. This is 35 initializations per year and 700 resumable API requests
across five years.

Keep all native members in raw files. Ensemble-size matching belongs in the
standardized analysis layer, not in the downloader.

See `COMPARABILITY.md` for the ECMWF/FuXi contract and
`ecmwf/README.md` for launch instructions.
