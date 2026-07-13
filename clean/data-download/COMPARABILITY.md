# ECMWF and FuXi Comparability Contract

## Benchmark Fields

`z500` is not part of this benchmark. Existing `z500` files remain in the
inventory because they are real data, but no new ECMWF request is planned for
them.

| Property | ECMWF | FuXi | Common contract |
|---|---|---|---|
| leads | up to 46 daily steps | 42 daily steps | same valid dates for FuXi leads 1-42 |
| ensemble | operational count varies; reforecast control + 10 perturbed | control + 50 perturbed | retain native; add matched-N sensitivity |
| precipitation | cumulative `kg m-2` | `mm h-1` rate | daily `mm day-1` |
| 2 m temperature | daily-average intervals in K | daily 00 UTC snapshots in K | degC and weekly means, with temporal-statistic caveat |
| grid | India subset, 1.5 degree | global, 1.5 degree | one canonical India grid |

`tp` is strictly comparable after differencing ECMWF accumulation and converting
FuXi rate to `mm day-1`. `t2m` is scientifically useful but is not a strict
like-for-like daily statistic: ECMWF supplies a daily mean while FuXi supplies
one 00 UTC value per lead. Report that limitation in every temperature result.

The raw FuXi 2002-2021 archive contains both fields and 51 total members. The
existing compact FuXi files contain only `tp` and `z500`, so `t2m` must be read
from the raw archive or compacted again. The local ECMWF 20-year reforecast has
`tp` and `z500`, but not `t2m`; ECMWF reforecast `t2m` must therefore be acquired
before a two-field 20-year comparison.

## Operational Date Pairing

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

1. Native ensemble: each system uses all available perturbed members.
2. Matched ensemble: compare the 10 ECMWF reforecast perturbed members with
   repeated 10-member FuXi subsets. Controls are reported separately.

For operational ECMWF versus FuXi, keep raw member counts unchanged and apply
the matched-member rule only in derived products. Never discard members during
download.

## Canonical Derived Fields

- Initialization: 00 UTC.
- Valid period: lead days 1-42, aggregated into six non-overlapping 7-day weeks.
- Domain: India analysis box, regridded once to the shared 1.5 degree grid.
- `tp`: daily increments/rates in `mm day-1`.
- `t2m`: ECMWF daily mean and FuXi 00 UTC snapshot in `degC`, reported with the
  temporal-statistic caveat.
- Truth: the same ERA5 valid dates for both models.
- Provenance: source path, request hash, native member count, lead count, units,
  and conversion must be retained in every standardized file.
