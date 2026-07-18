# NeuralGCM Model Runs

The first executable milestone is a one-date smoke test, not the six-year
production run. It uses the frozen physics-calendar case `2020-06-01 00 UTC`
and writes three native NeuralGCM cumulative-precipitation frames at +0, +6,
and +12 hours.

The implementation follows Google's official
[`Forecasting quick start`](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html):
load a public checkpoint, select hourly ARCO-ERA5, shift forcing variables by
24 hours, conservatively regrid with Dinosaur, encode, and call `unroll` with
persisted forcing. It deliberately changes the notebook in these places:

- selects `v1_precip/stochastic_precip_2_8_deg.pkl`, not the notebook's default
  deterministic checkpoint;
- freezes and hashes the checkpoint and staged IC instead of downloading and
  regridding them inside every forecast task;
- uses explicit NumPy valid times and checks the exact `start_with_input=True`
  frame count;
- saves only the required precipitation diagnostic and provenance rather than
  a visualization dataset;
- fails if cumulative precipitation decreases beyond the pilot tolerance or
  if a T2M field is incorrectly claimed.

For the eventual 42-day run, six-hourly frames are `+0, +6, ..., +1008 h`.
That requires `unroll_steps=169` with `start_with_input=True`; the separately
returned advanced state is later than the final saved frame. Do not copy the
notebook's four-day arithmetic without adapting this endpoint contract.

## Initial Conditions

All models use ERA5, but they do not use the same tensor or history:

| Model | Required initialization data |
|---|---|
| FCN3 | D 00 UTC global 72-channel ERA5 state on the 0.25 degree grid |
| DLESyM v0 | Six-hourly ERA5 history from D-48 h through D, including atmosphere, SST, and TTR-derived OLR inputs |
| NeuralGCM precipitation | D 00 UTC pressure-level atmosphere plus D-1 00 UTC SST and sea ice, regridded to the checkpoint's 128 x 64 Gaussian grid |

The NeuralGCM D-1 forcing is held fixed during rollout. This is the project's
no-future-information policy. It prevents observed future SST or sea ice from
leaking into a forecast that is labeled as issued at D 00 UTC.

## First Smoke Test

Status: passed on 2026-07-17 as Slurm job `69802` on an A100 GPU in
`GPU-AI_prio`. The job completed in 1 minute 56 seconds with exit code 0. The
output SHA256 is
`903c395ad404e5bd5f7d51df65b2b357c8c3422509fce9f5f5186c61c6b64e4b`.
Scheduler logs are retained under `model-runs/neural-gcm/logs/`.

The smoke contract is tied to the exact 621-date 2020-2025 physics/FuXi
calendar. JAX compilation caches are job-local because XLA's parallel
autotuner cannot safely create its temporary files on the shared storage
filesystem. The installed precipitation checkpoint emits a singleton
`surface` dimension; the runner validates that its size is one before removing
it.

Frozen config:

```text
model-runs/configs/neuralgcm_smoke_20200601.json
```

The Slurm job performs these steps in order:

1. Validate the calendar hash, case time, forcing cutoff, checkpoint pin, and
   smoke lead arithmetic.
2. Select the D atmosphere and D-1 forcing from hourly ARCO-ERA5.
3. Verify the 24-hour forcing shift against a direct D-1 read.
4. Conservatively regrid and save the native-grid IC with a SHA256 manifest.
5. Verify JAX sees a real GPU and every package matches the frozen versions.
6. Run one stochastic member at +0, +6, and +12 hours.
7. Check finite cumulative TP, monotonic increments, absence of T2M, and write
   an atomic smoke output plus report.

Submit with:

```bash
sbatch model-runs/neural-gcm/slurm/smoke_neuralgcm_20200601.sbatch
```

Success produces:

```text
inputs/2020/20200601.nc
inputs/2020/20200601.json
forecasts/2020/20200601_smoke.nc
manifests/2020/20200601_smoke.json
audit/SMOKE_REPORT.json
```

These files are smoke evidence only and must not be used for skill scores. A
passing smoke test promotes the runner to a two-day pilot, then one complete
42-day pilot, cross-year cases, and finally a ten-date batch. The 621-date run
is forbidden until those gates pass.

## Complete 42-Day TP Pilot

Status: passed on 2026-07-17 as Slurm job `69803` on an A100 GPU in
`GPU-AI_prio`. The job completed in 3 minutes 38 seconds with exit code 0.
The standardized output is:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/neural-gcm/
  neuralgcm_v1_precip_2p8_era5_00z_pilot42d_20200601_ens1/
    forecasts/2020/20200601.nc
```

Its SHA256 is
`fe6a1f21d2f92a2ab3a036e2ae4dd5993e65c86c5639de9495ae648240fb34f3`.
It contains `tp(member=1, lead_day=42, latitude=27, longitude=27)` in
`mm day-1`, with lead day 1 covering 2020-06-01 00 UTC through 2020-06-02
00 UTC and lead day 42 ending 2020-07-13 00 UTC. The observed pilot range is
0 to 140.6408 mm/day and all values are finite and nonnegative.

NeuralGCM contributes TP only to this benchmark. The public 1.4 degree
stochastic checkpoint outputs pressure-level temperature, not 2 m
temperature. No NeuralGCM temperature job is submitted and no NeuralGCM field
may be labeled or scored as `t2m`.
