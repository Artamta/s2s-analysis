# FuXi-S2S Launch Status

## Contract

- Calendar: 621 exact physics-model dates, 2020-2025.
- Ensemble: 50 model-native stochastic members from official `inference.py`.
- Inputs: previous/init UTC daily means from 24 hourly ERA5 values.
- Leads: days 1-42.
- Final fields: native FuXi TP and T2M on the exact 27 x 27 India grid.

## Superseded Runs

Pilot `67911_0` and production array `67923` used ERA5 00 UTC snapshots. They
were cancelled because the official FuXi sample confirms the model requires
daily means. Their artifacts remain isolated under
`invalid/00utc_snapshot_pipeline_20260713/` and must not be analyzed.

The first corrected staging design used the CDS daily-statistics service:

- `67936`: all-CDS monthly staging;
- `67940`: blocked task-0 pilot;
- `67941`: blocked production array.

These jobs were cancelled after three pressure requests remained accepted for
more than two hours without processing. Their request IDs were dismissed at
CDS and moved out of the active state tree:

```text
a13c2435-760d-4c0a-a46c-59c099a34101
3e0624c4-6533-4839-b36b-df58ecde21d2
7783db30-1691-4bc2-b883-f80d64b58ac9
```

A one-month hybrid trial (`68033`, pilot `68034`) proved that local daily ARCO
stages January pressure in about one minute, but its small CDS supplement also
remained accepted. It was cancelled, and request
`11c32cf0-df3a-406c-ac41-c890afed2221` was dismissed.

No CDS request remains active.

## Direct ARCO Launch

The current workflow uses anonymous public ARCO and the local precomputed daily
ARCO archive only:

- `68040`: January 2020 local daily ARCO plus direct hourly ARCO for
  `ttr/100u/100v`; completed successfully in `00:07:09` with 406 MB peak RSS;
- `68042`: January 2023 full 76-channel direct hourly ARCO throughput test;
- `68041`: official 50-member task-0 pilot, released by `68040_0` and running
  on `gpu1`;
- `68043`: remaining 2020-2022 local-daily/direct-ARCO months, three at a time;
- `68047`: remaining 2023-2025 full-hourly-ARCO months, two at a time, blocked
  until benchmark `68042_0` succeeds;
- `68048`, `68049`, `68050`: 2020, 2021, and 2022 forecast arrays;
- `68051`, `68052`, `68053`: 2023, 2024, and 2025 forecast arrays.

Each yearly forecast array depends on its exact 12 monthly staging tasks and
the successful pilot. Task 0 is produced only by pilot `68041`; array `68048`
starts at task 1, so no forecast date is duplicated.

The direct reader was compared with the official FuXi input before launch.
Instantaneous fields use 00-23 UTC. TTR and TP use accumulation intervals valid
01 UTC through next-day 00 UTC. The bounded reader reduced a two-day 100 m wind
test from 17 GB to about 0.53 GB peak RAM and matched the official sample.

One global pressure field/day took 34 seconds at eight workers and 1.49 GB peak
RAM. The measured January 2020 month time projects the complete 2020-2022 block
to roughly 1-2 hours at three concurrent month tasks. The expected full
2023-2025 staging time is approximately 1-2 days with two month tasks running
concurrently. All remaining staging and GPU arrays are already queued and
continue without an interactive shell.

Check the active benchmarks with:

```bash
squeue -j 68040,68041,68042,68043,68047,68048,68049,68050,68051,68052,68053
```

For a fresh launch, the same dependency-linked workflow is submitted with:

```bash
clean/model-runs/fuxi/slurm/submit_fuxi_hybrid.sh
```
