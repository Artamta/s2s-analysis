# Study Decisions

## Framing

This folder supports a **case-study paper**:

- JFM2026 is an operational winter/spring S2S benchmark over India.
- JJAS2019 is a monsoon case study with deeper process diagnostics.

This is not framed as a definitive 18-year climatological model ranking unless
the multi-year branch is explicitly regenerated and reviewed.

## Climatology

Use the climatology that matches the verification truth source:

- TP uses the climatology matching the requested truth source. IMD truth uses
  IMD 1991-2020 30-year rainfall climatology. ERA5 truth uses ERA5 1990-2019
  day-of-year TP climatology.
- JJAS2019 can be run against both IMD and ERA5 TP using `--truth-source both`.
- JFM2026 currently has ERA5 TP/T2M truth through 2026-05-10. IMD 2026 daily
  rainfall NetCDF is not available, so JFM IMD rainfall verification is not
  score-ready yet.
- ERA5 common climatology for Z500 and T2M anomalies.

ERA5 common climatology is read from:

`/storage/raj.ayush/All_Model_Data/climatology_common/era5_climatology.nc`

IMD 30-year rainfall climatology is read from:

`/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/`

Do not use model-own climatology in this clean paper branch yet. That avoids
mixing a case-study paper with a hindcast-calibrated multi-year paper.

## Metrics

Use ACC, not raw PCC, for spatial correlation skill.

- ACC: weighted spatial correlation between forecast anomaly and observed
  anomaly, both relative to the appropriate observational climatology field
  (IMD 1991-2020 for IMD rainfall truth, ERA5 common climatology for ERA5
  rainfall truth, and ERA5 common climatology for Z500/T2M).
- RMSE: weighted root-mean-square error on the original field.
- Bias: weighted mean forecast minus observation.
- MAE: weighted mean absolute error.
- CRPS: finite-member ensemble CRPS for raw ensembles.
- SSR: mean ensemble spread divided by RMSE of the ensemble mean. Ensemble
  spread uses sample standard deviation across members (`ddof=1`).
- Brier score: event-probability error for defined threshold events.
- Reliability bins: event-probability calibration counts and observed
  frequency, using only finite masked cells.

## Model Variables

Current clean-branch variable availability:

- ECMWF: TP, Z500, T2M.
- FuXi: TP, Z500, T2M.
- UKMO: TP, Z500, T2M.
- DELYSM: Z500, T2M only. DELYSM does not provide TP in the current stores.

## JJAS2019 Calendar Handling

FuXi-S2S and ECMWF-S2S do not have identical initialization calendars for
JJAS2019.

- FuXi compact files are mostly Sunday/Thursday.
- New ECMWF operational files are Monday/Thursday.
- Exact same-init comparison is therefore only the Thursday subset.
- Main JJAS2019 comparison should match forecasts by the same ERA5 valid date
  window.

Both comparisons are useful:

- **matched valid-window:** main fair case-study comparison.
- **exact Thursday-only:** sensitivity check for skeptical readers.

## Region Masks

The copied masks in `masks/` are the same validated IMD homogeneous-region masks
used by `final_analysis/`, plus a 0.25 degree mask generated with the same SOI
shapefile workflow:

- `imd_region_masks_1.5deg.nc`
- `imd_region_masks_0.5deg.nc`
- `imd_region_masks_0.25deg.nc`

All-India is defined as the union of the four IMD homogeneous rainfall regions.

## IMD Climatology Products

IMD rainfall climatology products are read from:

`/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/`

The current clean paper branch expects the 1991-2020 30-year daily and seasonal
NetCDF climatologies plus the all-India cumulative JJAS/JFM CSV products.
