# Production Launch Status

The all-season 2020-2024 operational forecast download was submitted on
`2026-07-13 16:53 IST` from commit `16d9a7a`.

| item | value |
|---|---|
| SLURM array | `67816` |
| branch | `agent/ecmwf-final-iteration` |
| tasks | 25 provider-year workers |
| concurrency | 3 workers maximum |
| providers | ECMWF, UKMO, NCEP, CMA, CNRM |
| output root | `/storage/raj.ayush/s2s_final_data/final_iteration` |

The job runs on the cluster's `gpu` partition because that is the available
queue selected for this project. It does not request a GPU: ECDS retrieval is
network-bound and GRIB validation is CPU-bound, so an accelerator would not
make the download faster.

## Initial Health Check

Tasks `0`, `1`, and `2` started concurrently for ECMWF, UKMO, and NCEP 2020.
Their first complete day-1-to-42 files passed field, member, step, grid, and
non-null validation before being promoted from `.part` files. The remaining
tasks are held by the intentional three-worker array limit, not by an error.

Validated initial outputs include:

```text
raw/ecmwf/forecast/annual2020/tp/20200102_cf.grib
raw/ukmo/forecast/annual2020/tp/20200102_cf.grib
raw/ukmo/forecast/annual2020/tp/20200102_pf.grib
raw/ukmo/forecast/annual2020/t2m/20200102_cf.grib
raw/ncep/forecast/annual2020/surface/20200102_cf.grib
```

The corresponding append-only JSONL records are under
`manifests/<provider>/forecast/annual2020.jsonl` and include the full request,
SHA-256 request hash, file size, member count, lead count, units, and QC status.

## Monitor or Resume

```bash
squeue -j 67816
tail -f /storage/raj.ayush/s2s_final_data/final_iteration/logs/production/slurm_67816_0.err
```

To inspect completed valid records:

```bash
rg '"status": "downloaded_valid"' \
  /storage/raj.ayush/s2s_final_data/final_iteration/manifests
```

If a task is interrupted, resubmit
`clean/data-download/slurm/download_all_season_2020_2024.sbatch`. The downloader
revalidates existing final files and skips them, so it does not create duplicate
GRIB data.
