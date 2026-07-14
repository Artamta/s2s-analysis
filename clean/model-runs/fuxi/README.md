# FuXi-S2S Benchmarks

The primary experiment is a strict information-matched comparison against
physics forecasts issued at 00 UTC. It produces official FuXi-S2S forecasts
for the 621 exact physics-model issue dates in 2020-2025. Here, "biweekly"
means twice per week: the dates frozen in
`clean/config/all_season_dates_2020_2025.csv`.

Two experiments are intentionally isolated:

| experiment | FuXi daily inputs for physics issue D | first target period | use |
|---|---|---|---|
| `strict00z` | D-2 and D-1 | D 00 to D+1 00 UTC | primary apples-to-apples benchmark |
| `published_same_nominal_date` | D-1 and D | D+1 00 to D+2 00 UTC | published-convention sensitivity only |

The second experiment has an effective information cutoff of D+1 00 UTC. It
must not be presented as a strict forecast issued at D 00 UTC.

## Scientific Contract

- Official FuXi-S2S ONNX checkpoint and official `inference.py`.
- 50 members from the model's native stochastic latent mechanism. No external
  input noise, artificial perturbation wrapper, or assumed control member.
- Lead days 1-42.
- For a physics issue date D at 00 UTC, two complete global UTC daily means,
  D-2 and D-1, provide the 76 input channels. Their latest interval ends
  exactly at D 00 UTC.
- Instantaneous ERA5 fields average timestamps 00-23 UTC.
- Accumulated TTR and TP average the 24 intervals valid from 01 UTC through
  next-day 00 UTC. TTR is converted from J m-2 to W m-2 by dividing by 3600.
- Lead day 1 covers D 00 UTC through D+1 00 UTC. Every output stores explicit
  `forecast_period_start`, `forecast_period_end`, and bounds.
- Final variables are TP and T2M on `39-0 N, 60-99 E` at 1.5 degrees
  (`27 x 27`), matching the physics downloads.
- FuXi TP remains its native `mm h-1` mean rate. Multiply by 24 when comparing
  with daily `mm day-1` precipitation.

These remain retrospective ERA5-initialized AI forecasts. Their information
cutoff and verification windows match the physics forecasts, but each physics
center uses its own operational analysis.

## ERA5 Source

The production workflow has **zero CDS requests**.

For December 2019 through 2022, it reads the cluster-local daily
WeatherBench2/ARCO archive:

```text
/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5-daily/
  1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr
```

That archive supplies all pressure fields and eight surface fields. The three
absent surface fields (`ttr`, `100u`, `100v`) are streamed anonymously from the
public hourly ARCO store and reduced locally.

For 2023-2025, all 76 channels are streamed from:

```text
gs://gcp-public-data-arco-era5/ar/
  full_37-1h-0p25deg-chunk-1.zarr-v3
```

The reader downloads bounded hourly chunks, point-samples every sixth 0.25
degree grid cell, and accumulates one day at a time. Each variable is written
as an atomic resumable shard before monthly files are assembled. A network
failure therefore resumes at the next missing variable rather than restarting
the month.

The local daily archive was checked against the official FuXi two-day sample.
All 74 available channels agree at floating-point precision. Direct hourly
ARCO checks also match the official sample:

```text
100u RMSE = 0.000059 m s-1
100v RMSE = 0.000050 m s-1
ttr  RMSE = 0.000283 W m-2
T    RMSE = 0.000057 K (all 13 pressure levels, one test day)
```

The earlier `01d_download_fuxi_inputs_arco_jjas2025.py` implementation proved
anonymous ARCO access on this cluster, but used only 00 UTC snapshots. Those
old inputs are not valid for this daily-mean benchmark.

## Storage

```text
/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fuxi/
  fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50/
    era5-daily/monthly/<YYYYMM>/{pressure,surface}.nc
    era5-daily/monthly/<YYYYMM>/.arco/...       # resumable field shards
    inputs/annual<year>/<YYYYMMDD>/input.nc
    forecasts/annual<year>/<YYYYMMDD>.nc
    manifests/annual<year>/<YYYYMMDD>.json
    work/annual<year>/<YYYYMMDD>/member/...     # removed after successful QC
    logs/era5_daily/slurm_<job>_<task>.out
    logs/inference/annual<year>/<YYYYMMDD>.log
```

Each compact forecast contains:

```text
tp,t2m(member=50, lead_day=42, latitude=27, longitude=27)
```

The two-day model inputs are retained. Temporary full 76-channel outputs are
deleted only after the compact forecast passes dimension, coordinate,
finite-value, physical-range, and ensemble-spread checks.

## Run

Dry-run one forecast row:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/run_fuxi_forecast.py \
  --task-index 0 --dry-run --verify-large-checkpoint
```

Submit the strict resumable staging, pilot, and per-year forecast workflow:

```bash
clean/model-runs/fuxi/slurm/submit_fuxi_strict00z.sh
```

The submitter launches:

- 37 local-daily-plus-public-ARCO month tasks for December 2019 through 2022;
- 36 public-hourly-ARCO month tasks for 2023-2025, at most two concurrently;
- one official 50-member pilot after January 2020 staging;
- one forecast array per year, dependency-linked to its required months,
  including the preceding December where a January issue needs it.

Rerunning validates and skips finished monthly fields, inputs, and forecasts.
No interactive shell is needed after submission.

Summarize progress:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/audit_fuxi_run.py \
  --config clean/config/fuxi_strict00z_2020_2025.json
```

Add `--verify-outputs` for a slower full NetCDF QC pass.

## Sources

- FuXi-S2S paper: <https://arxiv.org/abs/2312.09926>
- Official FuXi archive card: <https://huggingface.co/datasets/FudanFuXi/FuXi-S2S>
- Google ARCO-ERA5: <https://github.com/google-research/arco-era5>
