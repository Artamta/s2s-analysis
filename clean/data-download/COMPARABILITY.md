# Physics Models and FuXi Comparability Contract

## Benchmark Fields

`z500` is not part of this benchmark. Existing `z500` files remain in the
inventory because they are real data, but no new ECMWF request is planned for
them.

| Property | ECMWF | FuXi | Common contract |
|---|---|---|---|
| leads | up to 46 daily steps | 42 daily steps | same valid dates for FuXi leads 1-42 |
| ensemble | operational count varies; reforecast control + 10 perturbed | model-native stochastic members | retain native; add matched-N sensitivity |
| precipitation | cumulative `kg m-2` | `mm h-1` rate | daily `mm day-1` |
| 2 m temperature | daily-average intervals in K | daily mean in K | daily mean in degC and weekly means |
| grid | India subset, 1.5 degree | native global, compacted over India at 1.5 degree | one canonical India grid |

`tp` is comparable after differencing ECMWF accumulation and converting the
FuXi 24-hour mean rate to `mm day-1`. FuXi-S2S forecasts global daily means, so
`t2m` is also a daily-mean statistic. Align the UTC daily valid periods before
forming the common six weekly means.

The raw FuXi 2002-2021 archive contains both fields and 51 total members,
numbered `00-50`. Do not infer a deterministic control member from that
numbering: the local files do not encode a control role, and FuXi's ensemble is
model-native and stochastic. The
existing compact FuXi files contain only `tp` and `z500`, so `t2m` must be read
from the raw archive or compacted again. The local ECMWF 20-year reforecast has
`tp` and `z500`, but not `t2m`; ECMWF reforecast `t2m` must therefore be acquired
before a two-field 20-year comparison.

## Provider Differences

| provider | operational cadence | common forecast window | temperature | native reforecast archive |
|---|---|---:|---|---|
| ECMWF | native schedule, paired one-to-one | 42 days | daily mean | rolling 20 years; reuse local 2000-2019 TP |
| UKMO | daily | 42 days | daily mean | 1993-2016, only days 1/9/17/25 |
| NCEP | daily | 42 of 44 days | derived proxy | fixed 1999-2010, daily |
| FuXi | exact twice-weekly common dates for new runs | 42 days | daily mean | 2002-2021, fixed 35 JJAS MMDDs |

For UKMO and NCEP operational forecasts, use the exact FuXi target dates. For
NCEP, derive a daily temperature proxy from all four six-hour intervals:
`mean((mx2t6 + mn2t6) / 2)`. Never rename that proxy to plain `t2m`.

No 20-year period exists across every model. Use each provider's native
climatology for primary calibration and probabilistic skill. Use 2002-2010 as
a separately labeled nine-year common-period sensitivity only.

## Operational Date Pairing

The following 35-slot pairing applies only when reusing the downloaded
2002-2021 FuXi JJAS archive. The new 2020-2025 FuXi model run uses the exact 621
physics-calendar initializations and needs no shifted pairing.

Use the fixed 35 FuXi JJAS starts below. Before 2023-06-28, ECMWF operational
forecasts were initialized Monday and Thursday; from 2023-06-28 onward they are
daily. Assign starts chronologically, one to one, minimizing total lag while
requiring ECMWF to initialize no later than FuXi. Shift the ECMWF lead range so
valid dates match exactly. The one-to-one rule prevents one ECMWF forecast from
being counted as multiple independent samples.

```text
0602 0606 0609 0613 0616 0620 0623 0627 0630
0704 0707 0711 0714 0718 0721 0725 0728
0801 0804 0808 0811 0815 0818 0822 0825 0829
0901 0905 0908 0912 0915 0919 0922 0926 0929
```

| year | exact init | shifted, same valid window | availability |
|---:|---:|---:|---|
| 2019 | 0 | 35 | one-to-one plan uses a three-day shift |
| 2020 | 0 | 35 | complete target year |
| 2021 | 0 | 35 | complete target year |
| 2022 | 35 | 0 | FuXi model runs required |
| 2023 | 27 | 8 | exact from June 30 onward; FuXi model runs required |
| 2024 | 35 | 0 | FuXi model runs required |
| 2025 | 35 | 0 | FuXi model runs required |
| 2026 | 35 planned | 0 | 12 mature through July 11 as of 2026-07-13 |

An exact-only 2019 sensitivity can use `0606, 0613, 0620, 0627, 0704, 0711,
0718, 0725, 0801, 0808, 0815, 0822, 0829, 0905, 0912, 0919, 0926`. Those
17 starts are not used in the balanced 35-slot one-to-one plan because doing so
would reuse ECMWF forecasts for the intervening FuXi starts. There are no exact
target starts in 2020 or 2021. Every target is exact in 2022 and from 2024
onward; in 2023 the exact sequence begins at `0630`. For 2026, only
`0602, 0606, 0609, 0613, 0616, 0620, 0623, 0627, 0630, 0704, 0707, 0711`
is currently mature under the two-day retrieval delay.

The row-level source of truth is
`../config/comparable_dates_2019_2026.csv`. For a shifted pair with offset
`ecmwf_init - fuxi_init`, compare FuXi leads 1-42 with ECMWF leads
`1 - offset` through `42 - offset`.

Only operational years `2019-2021` overlap the downloaded FuXi 2002-2021
archive. Matching FuXi forecasts must be generated for operational years
`2022-2026`; observations alone do not turn those years into a FuXi comparison.

## Ensemble Policy

Use two views:

1. Native ensemble: each system uses all of its available native members.
2. Matched ensemble: compare the 10 ECMWF reforecast perturbed members with
   repeated 10-member FuXi subsets. The ECMWF control may be reported
   separately; FuXi has no assumed control member.

For operational ECMWF versus FuXi, keep raw member counts unchanged and apply
the matched-member rule only in derived products. Never discard members during
download.

## Canonical Derived Fields

- Initialization: 00 UTC.
- Valid period: lead days 1-42, aggregated into six non-overlapping 7-day weeks.
- Domain: India analysis box, regridded once to the shared 1.5 degree grid.
- `tp`: daily increments/rates in `mm day-1`.
- `t2m`: daily mean in `degC` for ECMWF and FuXi.
- Truth: the same ERA5 valid dates for both models.
- Provenance: source path, request hash, native member count, lead count, units,
  and conversion must be retained in every standardized file.
