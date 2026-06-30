# FuXi-S2S / ECMWF-S2S / IMD Rainfall Package

Initialization: 23 Jun 2026 (`20260623`)

Valid period: 2026-06-24 to 2026-08-04 (42 lead days)

This folder is intended to be zipped and shared. It contains PNG figures only
under `figures/`, lightweight CSV summaries under `data/`, methods/units notes,
and this script for regenerating the plots.

## Figures

- `figures/01_all_india_cumulative_rainfall_fuxi_ecmwf_imd.png`
- `figures/02_imd_regions_cumulative_rainfall_fuxi_ecmwf_imd.png`
- `figures/03_spatial_42day_accumulation_india.png`
- `figures/04_spatial_bias_vs_imd_climatology_india.png`

## Data And Sources

- FuXi-S2S: `/storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp/comparable_fuxi_op2026_ens50/fuxi_20260623_tp_ens50_lead42_india_1p5deg_daily_mm.nc`
- ECMWF-S2S: `/storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp/comparable_fuxi_op2026_ens50/ecmwf_20260623_tp_ens50_lead42_india_1p5deg_daily_mm.nc`
- IMD daily rainfall climatology: `/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/imd_rain_1991_2020_daily_climatology.nc`
- IMD homogeneous-region masks: `/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc`
- Summary CSVs copied from: `/storage/raj.ayush/All_Model_Data/ecmwf/jjas2026/tp/comparable_fuxi_op2026_ens50`

## Methods

See `METHODS_AND_UNITS.md` for formulas, units, mask handling, and the final
math verification. Independent recomputation from the NetCDF inputs matched the
summary CSV totals to within `0.00018 mm`.

## ERA5 Note

ERA5 climatology is not plotted here. A usable candidate exists at
`/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc`; it has `tp` in meters
on a 366-day climatological calendar. Its baseline years are not documented in
the file metadata, so this package leaves ERA5 for a later optional sensitivity
plot and keeps IMD 1991-2020 as the main observation climatology.

## Re-run

```bash
python make_prof_plots.py
```
