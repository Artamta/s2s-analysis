# ERPAS Forecast Archive: Model And Data Summary

This folder documents and analyses the ERPAS forecast archive stored locally.
All values below were read from the downloaded GRIB files rather than inferred
from filenames alone.

For a short presentation, use the
[`10-minute meeting brief`](MEETING_BRIEF.md), which orders the key figures and
includes speaker notes, example values, caveats, and likely questions.

> **Communication-ready summary:** This ERPAS archive contains weekly,
> Wednesday 00 UTC extended-range forecasts from 2023 through 29 October 2025.
> Each initialization has 33 daily lead times. The downloaded fields are
> precomputed unweighted means of four source forecasts, not four accessible
> ensemble members. Precipitation, surface air temperature (T2m convention), and pressure-level
> geopotential height are available globally at 1 degree; a broad
> 30-150 E, 30 S-50 N precipitation product is also available at 0.5 degree.

![ERPAS archive overview](outputs/dataset_overview.png)

## At A Glance

| Property | Value in this archive |
| --- | --- |
| System | ERPAS extended-range forecast product |
| Provider tree | IITM/IMD ERPAS archive |
| GRIB provenance | NCEP centre `kwbc`; generating process identifier 96 (GFS) |
| Forecast period downloaded | 4 January 2023 to 29 October 2025 |
| Initialization cadence | Weekly, Wednesday at 00 UTC |
| Forecast horizon | 33 daily leads: day 1 to day 33 |
| Forecast representation | Precomputed unweighted mean of 4 source forecasts |
| Raw ensemble members | Not present in these files |
| Global horizontal grid | 1 x 1 degree; 181 latitude x 360 longitude |
| Higher-resolution precipitation grid | 0.5 x 0.5 degree; 161 x 241; 30-150 E, 30 S-50 N |
| File format | GRIB2, CF-1.7 metadata when opened with `cfgrib` |
| Forecast files | 591 |
| Provider-climatology files | 432: 144 calendar start dates x 3 products |
| Total local archive | 1,023 GRIB files, 9.83 GiB |

## What ERPAS Means Here

ERPAS is the Extended Range Prediction System described by IITM for extended
range forecasting over India. The local files sit in the ERPAS provider tree,
while their GRIB headers identify NCEP (`kwbc`) and generating process 96,
which is GFS. These statements describe different layers: ERPAS is the forecast
system/product being distributed, while the GRIB centre and process identify
the NCEP model provenance encoded in this archive.

This README describes the **downloaded archive**, not every possible ERPAS
configuration or operational product. The `tsfc` field is treated as surface
air temperature/T2m following the project convention and domain-expert
guidance. Its GRIB metadata encode `surface`, level 0, rather than an explicit
`heightAboveGround=2` coordinate. Raw ensemble members are not present.

## Variables And Units

| Product and filename | GRIB variable | Units | Time meaning | Grid and coverage |
| --- | --- | --- | --- | --- |
| Global precipitation, `APCP_YYYYMMDD.grb` | `tp` | `kg m-2` | 24-hour accumulation at each daily lead; numerically equal to mm/day | 1 degree global, 181 x 360 |
| Regional precipitation, `Ind_0.5_APCP_YYYYMMDD.grb` | `tp` | `kg m-2` | 24-hour accumulation at each daily lead; numerically equal to mm/day | 0.5 degree, 30-150 E and 30 S-50 N, 161 x 241 |
| Surface air temperature (T2m convention), `tsfc_YYYYMMDD.grb` | `t` | K | Instantaneous surface air temperature at each daily endpoint; GRIB level is `surface`, level 0 | 1 degree global, 181 x 360 |
| Geopotential height, `gpot_YYYYMMDD.grb` | `gh` | gpm | Instantaneous pressure-level field at each daily endpoint | 1 degree global; 1000, 925, 850, 700, 500, 300, and 200 hPa |

The regional precipitation product covers a broad Asia/Indian Ocean domain;
it is not clipped to India's political boundary. The plotting script clips the
field for display and overlays the repository's India state-boundary shapefile.

## Ensemble Interpretation

The GRIB product metadata use derived-forecast templates and report:

- `derivedForecast = 0`: unweighted mean of all members;
- `numberOfForecastsInEnsemble = 4` for dated forecasts;
- `numberOfForecastsInEnsemble = 20` in the sampled provider climatology;
- no `member`, `number`, `ensemble`, or `realization` dimension.

Therefore, each dated field is one **ensemble-mean field made from four source
forecasts**. The archive cannot support member-wise trajectories, ensemble
spread, probabilities, rank histograms, CRPS from members, or reliability
analysis. Do not describe it as four downloaded members.

## Forecast Time Semantics

Lead `step=1 day` is forecast day 1 and `step=33 days` is forecast day 33. The
valid time is initialization time plus the lead step.

The forecast-period figures use these definitions:

| Label | Lead steps | Processing |
| --- | --- | --- |
| Day 1 | 1 | Field at the first daily lead |
| Week 1 | 1-7 | Mean of seven daily fields |
| Week 2 | 8-14 | Mean of seven daily fields |
| Week 3 | 15-21 | Mean of seven daily fields |
| Week 4 | 22-28 | Mean of seven daily fields |
| Week 5 | 29-33 | Mean of five daily fields; not a complete seven-day week |

For precipitation, the plotted weekly value is the **mean daily accumulation
in mm/day**. Use a sum instead when the scientific question requires total
rainfall accumulated over the forecast week.

## Archive Coverage

| Year | Initializations | Global TP | Surface T | Geopotential height | 0.5 degree TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 52 | 52 | 52 | 52 | 52 |
| 2024 | 52 | 52 | 52 | 52 | 51 |
| 2025 | 44 | 44 | 44 | 44 | 44 |

The missing product is 0.5-degree precipitation for initialization
`2024-12-18`. The 2025 snapshot ends on `2025-10-29`; this is archive coverage,
not evidence that the forecast system itself stopped on that date.

## Comparison With IMD And ERA5

| ERPAS field | Suitable reference | Required preparation |
| --- | --- | --- |
| Daily `tp` | IMD gridded rainfall or ERA5 total precipitation | Match the exact valid 24-hour period; convert ERA5 metres to mm with `x 1000`; apply a common mask and regrid before pointwise scores |
| Surface-air `t` (T2m convention) | ERA5/IMD near-surface air temperature | Convert K to degree C if desired; compare with reference T2m sampled at the same valid hour because ERPAS contains instantaneous daily endpoints, not daily means |
| 500 hPa `gh` | ERA5 geopotential/geopotential height at 500 hPa | If ERA5 variable is geopotential `z` in m2 s-2, divide by standard gravity to obtain geopotential metres; regrid before bias/RMSE |

Use the same initialization, lead, and valid date on both sides. Compare an
ensemble mean with an equivalent mean forecast or clearly state the mismatch.
Recommended deterministic diagnostics for this archive are spatial bias,
MAE/RMSE, anomaly correlation, pattern correlation, and regional time series.

## Forecast Plot Gallery

The example initialization is `2023-07-05`, during the Indian summer monsoon.
India panels use
`/storage/raj.ayush/archive/s2s-forecast-/91/STATE_BOUNDARY.shp`, reprojected
from Lambert Conformal Conic to WGS84.

### India Day 1 And Week 1

![India day 1 and week 1 variables](outputs/forecast-periods/india_day1_week1_all_variables_20230705.png)

### India Precipitation Through All Forecast Weeks

![India weekly precipitation evolution](outputs/forecast-periods/tp_india_weekly_evolution_20230705.png)

### Global, Asia, And India Precipitation

![Precipitation domain comparison](outputs/forecast-periods/tp_day1_week1_domains_20230705.png)

### Surface Air Temperature And 500 hPa Height

![Surface air temperature domains](outputs/forecast-periods/surface_temperature_day1_week1_domains_20230705.png)

![500 hPa height domains](outputs/forecast-periods/gh500_day1_week1_domains_20230705.png)

## Reproduce The Analysis

Generate the inventory, metadata tables, overview, and basic diagnostics:

```bash
python data-download/erpas/data-analysis/analyze_erpas.py
```

Generate the Cartopy maps with India national and state boundaries:

```bash
python data-download/erpas/data-analysis/plot_forecast_periods.py
```

Choose another available initialization or shapefile:

```bash
python data-download/erpas/data-analysis/plot_forecast_periods.py \
  --init-date 20240703 \
  --lead-day 1 \
  --india-shapefile /path/to/STATE_BOUNDARY.shp
```

Both scripts accept `--data-root`. The default is:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/raw/erpas
```

## Generated Outputs

- [`analysis_report.md`](outputs/analysis_report.md): generated archive report;
- [`inventory_summary.csv`](outputs/inventory_summary.csv): counts, sizes, and date ranges;
- [`product_metadata.csv`](outputs/product_metadata.csv): variables, grids, leads, levels, units, and ensemble metadata;
- [`dataset_overview.png`](outputs/dataset_overview.png): one-page archive summary;
- [`forecast-periods/README.md`](outputs/forecast-periods/README.md): forecast-map definitions and file index.
- [`MEETING_BRIEF.md`](MEETING_BRIEF.md): 10-minute figure order and speaker notes.

## References

- [IITM ERPAS project description](https://www.tropmet.res.in/6-project_details)
- [NCEP GRIB2 product definition template 4.2](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_temp4-2.shtml)
- [NCEP GRIB2 product definition template 4.12](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_temp4-12.shtml)
- [NCEP derived-forecast code table 4.7](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-7.shtml)
