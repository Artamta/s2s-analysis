# Methods And Units

## Forecast Window

- Initialization: 23 Jun 2026 (`20260623`)
- Valid period: 2026-06-24 to 2026-08-04
- Lead days: 1 to 42
- Spatial domain shown: India only, using the union of the four IMD homogeneous rainfall-region masks.

## Units

- FuXi-S2S package input: `tp(member, lead_time, lat, lon)` in `mm/day`.
- ECMWF-S2S package input: `tp(member, lead_time, lat, lon)` in `mm/day`.
- IMD climatology input: `rain_mean(day, lat, lon)` in `mm/day`; the dataset global attribute documents `units = mm/day`.
- IMD masks: binary 0/1 masks; 1 means grid cell belongs to the region.
- Figure and CSV totals: `mm` over the 42-day valid window.

## Core Formulas

Daily model rainfall is already in `mm/day` in the prepared package inputs.

For each model member:

```text
P_member_total(lat, lon) = sum_lead=1..42 tp(member, lead, lat, lon)
```

For the ensemble mean spatial field:

```text
P_ensmean(lat, lon) = mean_member(P_member_total(lat, lon))
```

For the IMD climatology spatial field:

```text
P_imd_clim(lat, lon) = sum_valid_days rain_mean(day, lat, lon)
```

Area means use cosine-latitude weighting over the selected IMD region mask:

```text
AreaMean(P) = sum(P_i * cos(lat_i) * mask_i) / sum(cos(lat_i) * mask_i)
```

Bias maps use:

```text
Bias(lat, lon) = Forecast_42day(lat, lon) - IMD_42day_climatology(lat, lon)
```

For bias maps, IMD climatology is linearly interpolated to each forecast grid before subtraction.

## Ensemble Ranges

- Forecast bands are member-wise percentiles across the 50 members.
- IMD climatology bands in the provided CSVs come from the prepared IMD climatology workflow.
- The plotted central lines are ensemble/climatological means.

## Verification

I independently recomputed all final 42-day summary totals from the NetCDF inputs and compared them with `data/20260623_summary_final_totals.csv`.

- Number of checks: 30
- Maximum absolute difference: `0.00018 mm`
- Interpretation: differences are only floating-point/rounding noise.

## ERA5

ERA5 is not included in these finalized plots. A candidate file exists at:

`/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc`

It contains `tp(dayofyear, latitude, longitude)` with units `m`, so precipitation must be converted using:

```text
tp_mm = tp_m * 1000
```

That file is technically usable for a later ERA5 sensitivity plot, but its baseline years are not documented in the NetCDF metadata. For the current professor-facing package, IMD 1991-2020 remains the main observation climatology.
