# FuXi-S2S Launch Status

## Contract

- Calendar: 621 exact physics-model dates, 2020-2025.
- Ensemble: 50 model-native stochastic members from official `inference.py`.
- No artificial input perturbations and no assumed control member.
- Inputs: two UTC daily means derived from 1-hourly ERA5.
- Leads: days 1-42, which are FuXi daily-mean forecasts.
- Final fields: native FuXi TP and T2M on the exact 27 x 27 India grid.

## Corrected Preflight

The official paper states that FuXi-S2S is trained on daily statistics and
produces global daily-mean forecasts. The official archive card states that
cumulative variables such as TP are 24-hour hourly averages. Therefore, the
first-pass use of ERA5 00 UTC snapshots was not a valid input contract.

The corrected workflow uses the CDS daily-statistics products cited by the
FuXi-S2S authors. It requests only the 1,242 unique previous/init days. The 287
requests are split by month, with every live CDS cost at or below `390/400`.
Request IDs are persisted before polling and each month task submits only one
request at a time. Per-date GPU tasks do no network retrieval.

## Cancelled Run

Pilot `67911_0` and production array `67923` used the invalid 00 UTC input
contract. Array `67923` was cancelled on 2026-07-13 after only tasks 1 and 2 had
started. The pilot output is not a valid benchmark product even though its
shape, physical range, and native stochastic spread passed mechanical QC.

Invalid artifacts from dates `20200102`, `20200106`, and `20200109` are kept at
`invalid/00utc_snapshot_pipeline_20260713/`, separate from the corrected
canonical paths, with their logs and manifests for provenance. They must not be
used in analysis.

## Next Launch

1. Complete 72 months of bounded ERA5 daily-statistics staging for 2020-2025.
2. Re-run task 0 as a full official 50-member pilot.
3. Validate daily input metadata, dimensions, TP/T2M ranges, and member spread.
4. Start tasks 1-620 with a dependency on both staging and the corrected pilot.

Progress is summarized with:

```bash
/home/raj.ayush/.conda/envs/s2s-hind/bin/python \
  clean/model-runs/fuxi/scripts/audit_fuxi_run.py
```
