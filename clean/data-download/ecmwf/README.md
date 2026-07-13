# ECMWF Operational JJAS 2020-2024

> Production launch is intentionally gated after the all-provider dry run. Do
> not submit the legacy SLURM file yet: the current ECDS form emits GRIB and
> uses `leadtime_hour`, while that executor still expects the older NetCDF
> request aliases. The source of truth is now
> `../manifests/download_plan/requests.jsonl`.

This phase fills the five missing operational years with the fields that are
directly usable in the ECMWF/FuXi benchmark:

- The 35 FuXi JJAS target starts, paired to ECMWF's actual initialization
  schedule.
- `tp` and `t2m`.
- Control and every perturbed member returned by ECMWF.
- The full common FuXi lead 1-42 valid window; shifted ECMWF starts may require
  native leads through day 45.
- India-centered raw request box on the 1.5 degree grid.

The output root is outside Git:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/ecmwf/forecast/jjasYYYY/
    tp/YYYYMMDD_cf.nc
    tp/YYYYMMDD_pf.nc
    t2m/YYYYMMDD_cf.nc
    t2m/YYYYMMDD_pf.nc
    manifests/requests.jsonl
  logs/ecmwf/
```

The downloader omits the ECMWF `number` selector intentionally, which retains
all available perturbed members. It validates variable, lead, member, and grid
dimensions after each download and writes one manifest record per attempt.

## Plan Check

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python \
  clean/data-download/ecmwf/scripts/download_ecmwf_operational.py \
  --year 2020 \
  --dates-file clean/config/comparable_dates_2019_2026.csv \
  --dry-run
```

Expected per year: 35 initializations and 140 requests. Expected total for the
five-year phase: 175 initializations and 700 requests.

## Cluster Launch (Blocked)

The old array script is retained for provenance only. Adapt and validate the
executor against the generated GRIB request manifest before any `sbatch` call.

`t2m` uses ECMWF's daily-average interval syntax (`0_24`, `24_48`, ...).
Using only endpoint steps (`24`, `48`, ...) silently returns the first daily
average, which is why legacy 2019/2025 `t2m` files cannot fill this benchmark.

## Scientific Boundary

These are operational forecasts, not a homogeneous model reforecast. Use them
for year-specific verification. Use the separate ECMWF and FuXi reforecast
archives for the homogeneous 20-year comparison described in
`../COMPARABILITY.md`.
