# All-Season Production Download

The current run is recorded in `LAUNCH_STATUS.md` (SLURM array `67816`).

## Scope

Operational forecasts for 2020-2024 over `0-40 N, 60-100 E`, retaining control
and all native perturbed members through lead day 42.

| provider | role | dates | requests | temperature |
|---|---|---:|---:|---|
| ECMWF | primary | 517 | 2,068 | daily-mean T2M |
| UKMO | primary | 517 | 2,068 | daily-mean T2M |
| NCEP | primary | 517 | 1,034 | proxy from four six-hour extrema intervals/day |
| CMA | primary | 517 | 2,068 | daily-mean T2M |
| CNRM | weekly secondary | 217 | 868 | daily-mean T2M |
| **total** | | | **8,106** | |

The core calendar contains `105/104/104/104/100` exact dates in 2020-2024.
CNRM contributes `11/52/52/52/50`; its sparse 2020 coverage must be disclosed.

## Storage

```text
/storage/raj.ayush/s2s_final_data/final_iteration/
  raw/<provider>/forecast/annual<year>/
    tp/<YYYYMMDD>_<cf|pf>.grib
    t2m/<YYYYMMDD>_<cf|pf>.grib
    surface/<YYYYMMDD>_<cf|pf>.grib  # NCEP TP + 6h extrema
  manifests/<provider>/forecast/annual<year>.jsonl
  logs/production/slurm_<job>_<task>.{out,err}
```

Files download to `.part`, pass GRIB field/member/step/grid/non-null QC, and
then move atomically to their final names. Resubmission validates and skips
existing files. Each provider/year has an append-only JSONL provenance log and
every request has a SHA-256 hash.

## Generate Calendar

With ECDS credentials configured:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/data-download/scripts/build_all_season_calendar.py
```

The checked-in calendar was generated from live ECDS constraints on
`2026-07-13`. Regeneration can change rows if the upstream archive is repaired.

## Launch and Resume

```bash
mkdir -p /storage/raj.ayush/s2s_final_data/final_iteration/logs/production
sbatch clean/data-download/slurm/download_all_season_2020_2024.sbatch
```

The 25-task array maps five providers across five years and allows only three
simultaneous workers. Each worker sends requests serially and sleeps between
completed requests. This is intentionally rate-limited: GPU hardware does not
accelerate ECDS transfers, and excessive concurrency makes queue throttling and
HTTP failures more likely.

Monitor with:

```bash
squeue -u "$USER"
tail -f /storage/raj.ayush/s2s_final_data/final_iteration/logs/production/slurm_<job>_<task>.out
```

Cancel only when necessary with `scancel <job-id>`. Resubmit the same `sbatch`
file after interruption; valid files are not downloaded twice.

## Reforecast Handoff

The operational executor intentionally does not mix reforecasts into these
folders. A reforecast executor should reuse its request/QC/manifest functions
but write under `raw/<provider>/reforecast/`. The provider-native climatology
periods and initialization calendars must remain explicit; do not manufacture
a common 20-year archive where none exists.
