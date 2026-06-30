# Current State Summary

This is the clean paper branch for the India S2S case-study analysis. It is
separate from `final_analysis/`, which remains the older working analysis tree.

## Study Scope

- JFM2026: operational winter/spring S2S case over India.
- JJAS2019: Indian monsoon case study.
- Current framing: case-study paper, not a definitive multi-year hindcast
  ranking.
- Climatology basis: observational climatology only. Model-own climatology is
  not used in this clean branch yet.

## Data Sources

- All model data root:
  `/storage/raj.ayush/All_Model_Data`
- Common ERA5 climatology:
  `/storage/raj.ayush/All_Model_Data/climatology_common/era5_climatology.nc`
- ERA5 daily TP/T2M truth:
  `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`
- IMD 30-year rainfall climatology:
  `/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/`
- JJAS2019 IMD rainfall truth:
  `/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/netcdf/imd_rain_2019.nc`
- JJAS2019 FuXi compact:
  `/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined`
- JJAS2019 ECMWF new data:
  `/storage/raj.ayush/All_Model_Data/ecmwf/jjas2019`
- JFM2026 FuXi:
  `/storage/raj.ayush/All_Model_Data/fuxi/jfm2026`
- JFM2026 ECMWF:
  `/storage/raj.ayush/All_Model_Data/ecmwf/jfm2026`
- JFM2026 ground truth:
  `/storage/raj.ayush/All_Model_Data/fuxi/jfm2026/ground_truth`
- JFM SPIRE zarr:
  `/storage/raj.ayush/archive/All_Model_Data/models/spire/data/s2s-research.zarr`

Important current caveat: JFM2026 TP/T2M truth now uses the daily ERA5 files in
`/storage/raj.ayush/s2s-forecast-data-prev/era5/daily`. TP covers 2025-12-25 to
2026-05-10 and T2M covers 2026-01-01 to 2026-05-10. For a 2026-03-31 init,
weeks 1-5 are fully covered; week 6 has only 5 of 7 truth days because
2026-05-11 and 2026-05-12 are absent.

JFM2026 Z500 still uses the local ERA5 CDS daily files under the FuXi
ground-truth directory, covering 2026-01-01 to 2026-03-31.

IMD rainfall NetCDF exists for 2019, but `imd_rain_2026.nc` is not present in
the IMD NetCDF folder as of this check. The file
`/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/quick_test/2026.grd`
is currently zero bytes, so JFM2026 IMD rainfall verification is not score-ready.

## Climatology Policy

- Rainfall/TP climatology follows the requested truth source. IMD truth uses
  IMD 1991-2020 30-year rainfall climatology. ERA5 truth uses ERA5 1990-2019
  day-of-year TP climatology converted from m to mm/day.
- Z500 and T2M verification uses the common ERA5 climatology file.
- ERA5 climatology file variables:
  - `t2m(dayofyear, latitude, longitude)`, units `K`
  - `tp(dayofyear, latitude, longitude)`, units `m`
  - `z500(dayofyear, latitude, longitude)`, units `m**2 s**-2`
- IMD daily climatology:
  - baseline `1991-2020`
  - 365-day climatology, Feb 29 removed
  - `rain_mean(day, lat, lon)`, `rain_std(day, lat, lon)`,
    `rain_count(day, lat, lon)`
  - units `mm/day`
- IMD seasonal climatology:
  - seasons `JFM` and `JJAS`
  - includes seasonal total mean/std/p10/p90 and seasonal daily mean.

## Grid And Regions

- Standard verification grid: India box, descending latitude and ascending
  longitude.
- Weekly lead windows:
  - Week 1: lead days 1-7
  - Week 2: lead days 8-14
  - Week 3: lead days 15-21
  - Week 4: lead days 22-28
  - Week 5: lead days 29-35
  - Week 6: lead days 36-42
- Masks available:
  - `masks/imd_region_masks_1.5deg.nc`
  - `masks/imd_region_masks_0.5deg.nc`
  - `masks/imd_region_masks_0.25deg.nc`
- Regions:
  - All India
  - Northwest India
  - Central India
  - South Peninsula
  - East & Northeast India
- All India is the union of the four IMD homogeneous rainfall regions.

## Deterministic Metrics

The deterministic score wrapper returns:

- ACC
- RMSE
- Bias
- MAE

ACC is not raw PCC. It is the area-weighted spatial correlation between forecast
anomaly and observed anomaly:

`ACC = corr_w(forecast - climatology, observation - climatology)`

The correlation is spatially centered after anomaly calculation. ACC requires an
explicit climatology argument, so scripts cannot accidentally report raw PCC as
ACC.

RMSE, bias and MAE are computed on the original field values after coordinate
alignment and masking.

## Probabilistic Metrics

The probabilistic score helpers include:

- finite-member ensemble CRPS
- Gaussian CRPS
- CRPSS
- ensemble mean and spread
- SSR
- exceedance/below-threshold event probability
- Brier score
- Brier skill score
- reliability bins

Finite-member ensemble CRPS is:

`CRPS = mean_i |x_i - y| - 0.5 * mean_ij |x_i - x_j|`

Gaussian CRPS is:

`sigma * [z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]`, where
`z = (obs - mean) / sigma`.

SSR is:

`SSR = mean ensemble spread / RMSE(ensemble mean, observation)`

Ensemble spread uses sample standard deviation across members (`ddof=1`), which
matches the older `final_analysis/core/aggregate.py` convention.

Event probabilities ignore missing ensemble members and keep fully masked cells
as `NaN`, so Brier and reliability calculations do not count outside-region or
fully missing cells.

## Preflight Checks

Run these before overnight jobs:

```bash
cd /home/raj.ayush/s2s/s2s_anlysis/final_paper
python scripts/00_check_foundation.py
python scripts/01_check_core.py
python scripts/02_check_metrics_formulas.py
python -m compileall -q code scripts
```

Current checks validate path existence, mask cell counts, core grid/mask/metric
behavior, exact hand-value metric formulas, and Python syntax.
