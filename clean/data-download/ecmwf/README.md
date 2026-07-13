# ECMWF Operational JJAS 2020-2024

This phase fills the five missing operational years with the fields that are
directly usable in the ECMWF/FuXi benchmark:

- Monday/Thursday initializations from June 1 through September 30.
- `tp` and 500 hPa geopotential height.
- Control and every perturbed member returned by ECMWF.
- Daily lead days 1-42.
- India-centered raw request box on the 1.5 degree grid.

The output root is outside Git:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/ecmwf/forecast/jjasYYYY/
    tp/YYYYMMDD_cf.nc
    tp/YYYYMMDD_pf.nc
    z/500/YYYYMMDD_cf.nc
    z/500/YYYYMMDD_pf.nc
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
  --year 2020 --dry-run
```

Expected per year: 35 initializations and 140 requests. Expected total for the
five-year phase: 175 initializations and 700 requests.

## Cluster Launch

```bash
sbatch clean/data-download/ecmwf/slurm/download_jjas_2020_2024.sbatch
```

The array runs at most two years concurrently. It is safe to resubmit: valid
existing files are checked and skipped.

## Scientific Boundary

These are operational forecasts, not a homogeneous model reforecast. Use them
for year-specific verification. Use the separate ECMWF and FuXi reforecast
archives for the homogeneous 20-year comparison described in
`../COMPARABILITY.md`.
