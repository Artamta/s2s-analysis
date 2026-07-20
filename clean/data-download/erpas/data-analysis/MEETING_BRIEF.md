# ERPAS: Brief

## One-Sentence Description

The downloaded ERPAS archive contains weekly Wednesday 00 UTC extended-range
forecasts with 33 daily leads, supplied as precomputed four-forecast ensemble
means for precipitation, surface air temperature (T2m convention), and
pressure-level geopotential height.

## 0:00-2:00 | What Data Do We Have?

![ERPAS archive overview](outputs/dataset_overview.png)

Say:

> We have 1,023 GRIB2 files occupying 9.83 GiB: 591 dated forecast files and
> 432 provider-climatology files. The forecast snapshot covers 2023 through 29
> October 2025, initialized weekly on Wednesday at 00 UTC, with daily lead times
> from day 1 to day 33.

| Item | Meeting-ready description |
| --- | --- |
| Forecast representation | One precomputed unweighted mean of four source forecasts |
| Raw members | Not downloaded; ensemble spread and probabilities cannot be calculated |
| Global grid | 1 x 1 degree, 181 latitude x 360 longitude |
| Regional precipitation grid | 0.5 x 0.5 degree, 161 x 241, covering 30-150 E and 30 S-50 N |
| Precipitation | `tp`, 24-hour accumulation, `kg m-2`, numerically mm/day |
| Temperature | `t`, K, surface air temperature/T2m convention, instantaneous daily endpoint |
| Geopotential height | `gh`, gpm, at 1000, 925, 850, 700, 500, 300, and 200 hPa |
| Climatology | 144 calendar start dates for precipitation, temperature, and geopotential height; sampled GRIB reports a mean of 20 forecasts |

Mention the coverage caveat: the 0.5-degree precipitation file for
`2024-12-18` is missing, and the 2025 snapshot is incomplete after 29 October.

## 2:00-5:00 | Spatial Forecast Structure

![India forecast summary](outputs/forecast-periods/india_day1_week1_all_variables_20230705.png)

This example is initialized **5 July 2023 at 00 UTC**.

Say:

> The top row is lead day 1 and the bottom row is the Week-1 mean. The
> precipitation field uses the 0.5-degree regional product, while T2m and Z500
> use the 1-degree global grid. The state and national boundaries are from our
> India shapefile. Weekly averaging smooths daily synoptic variability and
> makes the larger-scale spatial pattern easier to compare.

Visible features to describe without overinterpreting one forecast:

- precipitation is spatially heterogeneous, with strong west-coast and
  surrounding-ocean rainfall and regional maxima over parts of India;
- surface air temperature is cooler over the Himalayan/northern high terrain
  and warmer over much of central and peninsular India;
- Z500 shows the broad mid-tropospheric height gradient and circulation-scale
  structure rather than local rainfall detail.

Do not call this a verification result. It is a demonstration of the forecast
fields and their spatial scales.

## 5:00-7:00 | Temporal Evolution Through Day 33

![India lead evolution](outputs/forecast-periods/india_country_mean_lead_evolution_20230705.png)

The lines are daily values averaged over grid cells inside India's national
outline. Thick black segments are forecast-period means.

| Period | Lead days | Rainfall (mm/day) | T2m (degC) | Z500 (gpm) |
| --- | ---: | ---: | ---: | ---: |
| Week 1 | 1-7 | 11.73 | 23.71 | 5851.7 |
| Week 2 | 8-14 | 9.66 | 23.26 | 5843.8 |
| Week 3 | 15-21 | 11.86 | 23.19 | 5846.7 |
| Week 4 | 22-28 | 9.25 | 23.04 | 5843.6 |
| Week 5* | 29-33 | 8.69 | 23.23 | 5843.0 |

Say:

> The daily curves retain lead-to-lead variability, while the weekly segments
> summarize the extended-range signal. In this example rainfall strengthens
> again in Week 3 before weakening in Weeks 4 and 5. Week 5 contains only five
> days, so it should not be presented as a full seven-day mean.

These are spatial means for one initialization. They do not measure model
skill or forecast uncertainty.

## 7:00-9:00 | Week-To-Week Spatial Evolution

![Weekly precipitation evolution](outputs/forecast-periods/tp_india_weekly_evolution_20230705.png)

Say:

> This figure retains the spatial information that is hidden by an India mean.
> It shows where the weekly rainfall pattern shifts from Week 1 through days
> 29-33. All panels use one common color scale, so changes can be compared
> directly.

Important definition: weekly precipitation here is the **mean daily
accumulation in mm/day**. Summing the daily fields would instead give total
rainfall over each period.

## 9:00-10:00 | Validation Plan And Takeaway

Say:

> The next step is forecast verification against IMD rainfall and ERA5. We
> will match valid dates and accumulation windows, convert units, apply a
> common India mask, and regrid to a common resolution before computing bias,
> RMSE, correlation, and weekly regional diagnostics.

Comparison rules:

| ERPAS field | Reference and alignment |
| --- | --- |
| Precipitation | IMD gridded rainfall or ERA5 TP; match the same 24-hour valid period and regrid |
| T2m | ERA5/IMD near-surface air temperature sampled at the same valid hour; ERPAS is an instantaneous endpoint, not a daily mean |
| Z500 | ERA5 500 hPa geopotential height; convert ERA5 geopotential `z` to gpm by dividing by standard gravity when needed |

## Likely Questions

**Is this four ensemble members?**  
No. The files contain one derived mean made from four source forecasts; the raw
member dimension is absent.

**Is the high-resolution precipitation India-only?**  
No. It is a broad 30-150 E, 30 S-50 N regional product. The plots clip it and
overlay the India boundary.

**Is `tsfc` T2m?**  
For this project it is treated as WMO surface air temperature/T2m following
domain-expert guidance. The raw GRIB encodes `surface`, level 0. It is
instantaneous at each daily endpoint.

**Does this show ERPAS is accurate?**  
No. These figures describe one forecast's data structure and evolution. Skill
requires multi-initialization comparison with observations or reanalysis.

**What is the main takeaway?**  
ERPAS provides a useful 33-day ensemble-mean forecast archive with global
1-degree atmospheric fields and a 0.5-degree regional precipitation product,
but member-wise uncertainty analysis is impossible with this download.

