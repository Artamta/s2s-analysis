# S2S Acquisition Dry Run

Status date: `2026-07-13`. No production retrieval was submitted.

## Result

ECDS authentication and catalogue access succeeded for both `s2s-forecasts`
and `s2s-reforecasts`. The live forms expose ECMWF, UKMO, and NCEP; both
forecast types; `total_precipitation`; `2_m_temperature`; and the six-hour
temperature extrema needed for NCEP. Local 2019/2025 files also prove that all
three origins have previously retrieved successfully with the configured API.

The duplicate check passed across `2,024` planned/reused assets: zero repeated
target paths and zero repeated active request hashes.

## Phase 1: Operational Forecasts

The balanced experiment uses the same 35 fixed FuXi JJAS targets in each of
2020-2024 and a common 42-day valid window.

| provider | inits/year | assets/year | five-year requests | native members | temperature field |
|---|---:|---:|---:|---:|---|
| ECMWF | 35 paired starts | 140 | 700 | 51 in older cycles; cycle dependent | daily-mean `t2m` |
| UKMO | 35 exact daily starts | 140 | 700 | 4 | daily-mean `t2m` |
| NCEP | 35 exact daily starts | 70 | 350 | 16 | proxy from four 6-hour Tmin/Tmax midranges per day |
| **total** | **525 samples** | | **1,750** | | |

Each ECMWF/UKMO asset is one field and forecast type. Each NCEP surface asset
bundles `tp`, `mx2t6`, and `mn2t6`, avoiding three redundant files and API
requests. Control and every native perturbed member are retained.

Rough storage from local regional files is `4-6 GB` for all five forecast
years. This is an order-of-magnitude planning estimate, not an ECDS quote:
ECMWF and UKMO full daily-temperature series are extrapolated from incomplete
legacy day-1 files, while NCEP is scaled from daily endpoints to 6-hour steps.
Leave at least `10 GB` free for raw files, temporary `.part` files, indexes,
manifests, and format variation.

## Phase 2: Reforecasts

| provider | historical years | native JJAS slots | total members | planned requests | local reuse |
|---|---|---:|---:|---:|---|
| ECMWF | 2000-2019 | 35 | 11 | 70 T2M | 70 TP assets |
| UKMO | 1993-2016 | 16 (1, 9, 17, 25 monthly) | 7 | 64 TP/T2M | none found |
| NCEP | 1999-2010 | 35 selected daily slots | 4 | 70 bundled surface | none found |

This is not a uniform 20-year archive. Primary calibration and skill must use
each model's native climatology. A separate `2002-2010` comparison gives the
only common FuXi/ECMWF/UKMO/NCEP historical period, but it is a nine-year
sensitivity and must not be described as a 20-year climatology.

ECMWF precipitation is already complete at
`/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data`; the plan points
to those files instead of downloading copies. Its missing reforecast field is
T2M. UKMO and NCEP reforecast roots are absent/empty and require acquisition.

## Temperature Contract

- ECMWF and UKMO `t2m` are daily averages represented by intervals such as
  `0_24`, `24_48`, and `48_72` in the current ECDS form.
- NCEP direct daily-mean T2M is unreliable in the inspected local forecast
  files. Download all six-hour `mx2t6`/`mn2t6` values and derive
  `mean((Tmax_6h + Tmin_6h) / 2)` over four intervals. Name it `t2m_proxy`, not
  `t2m`.
- FuXi T2M is a daily 00 UTC snapshot. Weekly temperature comparisons remain
  useful but are not strict like-for-like daily statistics.

## Reproduce

```bash
python clean/data-download/scripts/plan_s2s_downloads.py \
  --phase all --providers ecmwf,ukmo,ncep --years 2020-2024 --write
```

Outputs are `manifests/download_plan/summary.csv` and `requests.jsonl`. The
planner only reads inventories and writes manifests; it cannot contact ECDS or
download model data.

## Launch Gate

Do not start Phase 2 until Phase 1 files pass member, lead, coordinate, and
variable QC. Before a production launch, the existing ECMWF downloader and new
UKMO/NCEP executors must consume these exact request records, write `.part`
files, validate them, atomically rename them, and append completion manifests.

Official references: [S2S model configurations](https://confluence.ecmwf.int/spaces/S2S/pages/40796876/Models),
[parameter definitions](https://confluence.ecmwf.int/spaces/S2S/pages/26903293/Parameters),
[reforecast catalogue](https://ecds.ecmwf.int/datasets/s2s-reforecasts), and
[NCEP reforecast examples](https://confluence.ecmwf.int/display/S2S/NCEP%2Bre-forecast%2Bexamples).
