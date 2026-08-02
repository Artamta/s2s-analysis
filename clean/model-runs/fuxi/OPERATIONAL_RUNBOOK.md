# FuXi Operational Forecast Runbook

This workflow creates source-aware FuXi-S2S forecasts without copying a new
Python or Slurm launcher for each date.

## Schedule

- Wednesday and Saturday: 100 stochastic members, 42 daily leads.
- Other requested dates: 5 stochastic members, 42 daily leads, explicitly
  labeled as rapid guidance.
- GFS is a near-real-time experimental proxy initialization.
- ERA5 is a delayed reference initialization and is submitted only after all
  768 required hourly ARCO objects pass the availability probe.

The official FuXi `inference.py` is used for both ensemble sizes. No external
perturbations or assumed control member are added.

## One case

Check availability:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/fuxi_operational.py probe \
  --source both --date 20260722
```

Create a reusable case configuration:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/fuxi_operational.py create \
  --source gfs --date 20260801 --members 100
```

Submit staging, inference, and dependent web publication:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/fuxi_operational.py submit \
  --date 20260801 \
  --configs clean/config/operational/fuxi_gfs_20260801_ens100.json
```

## Mixed job array

Pass any mixture of GFS/ERA5 and 5/100-member generated configurations to:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/fuxi_operational.py submit-array \
  --concurrency 2 --configs <config1.json> <config2.json> ...
```

The command submits:

1. a bounded-concurrency CPU staging array;
2. a corresponding GPU inference array with `aftercorr` dependencies;
3. one publication job after every inference task succeeds.

Each stage is idempotent. Existing valid inputs and forecasts are checked and
skipped; invalid products are quarantined. Submission records and batch item
maps are stored under:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/
  operational/submissions/
```

## Website and comparisons

Successful publication adds source-aware issue JSON, India NetCDF/CSV/PDF
downloads, and the issue catalog. Complete 100-member issues additionally
receive six-week raw ensemble tercile probabilities and area-weighted summaries
for All India and IMD's four broad homogeneous rainfall regions. Five-member
rapid guidance deliberately withholds probability products. When both sources
exist for one issue, the publisher also writes a six-week GFS-minus-ERA5
initialization-sensitivity comparison and exposes its Week-1 summary on the
forecast page.

That comparison is not a skill score. It measures changes caused by the input
source while holding the issue date and model fixed. GFS-versus-ERA5 skill is
reported only after the complete valid weeks have observations.

## Production gates

- input shape exactly `2 × 76 × 121 × 240`;
- complete D-2/D-1 temporal contract;
- configured member count and 42 leads;
- finite values, plausible TP/T2M ranges, and non-collapsed spread;
- unique full-forecast fingerprint for every ensemble member;
- private run-manifest checksum must match before web publication.
