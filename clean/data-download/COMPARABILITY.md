# ECMWF and FuXi Comparability Contract

## What Is Directly Comparable

| Property | ECMWF reforecast | FuXi archive | Common contract |
|---|---|---|---|
| years | 20 hindcast years per MMDD file | 2002-2021 | use overlapping years |
| cadence | 35 JJAS MMDD slots | 35 JJAS MMDD slots | nearest-init, same valid dates |
| leads | 46 daily steps | 42 daily steps | lead days 1-42 |
| ensemble | control + 10 perturbed | control + 50 perturbed | retain native; add matched-N sensitivity |
| precipitation | cumulative `kg m-2` | `mm h-1` rate | daily `mm day-1` |
| z500 | geopotential height (`gpm`) | geopotential (`m2 s-2`) | geopotential height (`gpm`) |
| grid | India subset, 1.5 degree | global, 1.5 degree | one canonical India grid |

The calendars are not identical. ECMWF reforecast MMDDs begin `0601, 0604,
0608, ...`; FuXi begins `0602, 0606, 0609, ...`. Pair the nearest
initializations, then shift lead windows so both forecasts verify over the exact
same dates. Do not label this an exact-initialization comparison.

## Operational ECMWF 2020-2024

The operational archive is a separate experiment from the 20-year reforecasts.
Download the provider-native Monday/Thursday cycle and preserve every member.
Model upgrades and changing ensemble sizes are metadata, not errors, and must be
recorded per file.

Only `2019-2021` overlap the downloaded FuXi 2002-2021 archive. ECMWF
operational years `2022-2024` can be verified against observations, but they
cannot be called a direct FuXi-archive comparison unless matching FuXi forecasts
are generated for those years.

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
- `z500`: instantaneous geopotential height in `gpm`.
- Truth: the same ERA5 valid dates for both models.
- Provenance: source path, request hash, native member count, lead count, units,
  and conversion must be retained in every standardized file.
