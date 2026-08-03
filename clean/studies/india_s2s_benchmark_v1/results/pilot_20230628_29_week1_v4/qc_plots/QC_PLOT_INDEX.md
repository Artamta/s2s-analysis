# Pilot visual-QC plot index

Catalog ID: `d3ce84adc9f1a7ca`

These figures verify preprocessing, metadata, valid-time alignment, grids, and plausible values. They are not model-skill figures.

## Required cautions

- NCEP `t2m_proxy` and ERPAS instantaneous `tsfc` are not daily-mean T2M.
- ERPAS is a provider-precomputed mean and is not a probabilistic one-member ensemble.
- Small negative accumulated-field increments are preserved and shown explicitly.
- Cross-model daily panels use the intersection of `valid_time`, not equal lead indices.

## Figures

- [00_spatial_support.png](00_spatial_support.png)
- [01_tp_week1_total_common.png](01_tp_week1_total_common.png)
- [01b_tp_matched_6day_total_common.png](01b_tp_matched_6day_total_common.png)
- [02_t2m_week1_mean_common.png](02_t2m_week1_mean_common.png)
- [03_temperature_noncomparable.png](03_temperature_noncomparable.png)
- [04_erpas_tp_native_vs_common.png](04_erpas_tp_native_vs_common.png)
- [05_tp_ensemble_spread.png](05_tp_ensemble_spread.png)
- [06_member_inventory.png](06_member_inventory.png)
- [07_negative_tp_diagnostics.png](07_negative_tp_diagnostics.png)
- [08_erpas_gh_pressure_levels.png](08_erpas_gh_pressure_levels.png)
- [09_valid_time_alignment.png](09_valid_time_alignment.png)
- [tp_valid_2023-06-30.png](daily_tp_matched_valid_time/tp_valid_2023-06-30.png)
- [tp_valid_2023-07-01.png](daily_tp_matched_valid_time/tp_valid_2023-07-01.png)
- [tp_valid_2023-07-02.png](daily_tp_matched_valid_time/tp_valid_2023-07-02.png)
- [tp_valid_2023-07-03.png](daily_tp_matched_valid_time/tp_valid_2023-07-03.png)
- [tp_valid_2023-07-04.png](daily_tp_matched_valid_time/tp_valid_2023-07-04.png)
- [tp_valid_2023-07-05.png](daily_tp_matched_valid_time/tp_valid_2023-07-05.png)
- [10_tp_india_area_mean_timeseries.png](10_tp_india_area_mean_timeseries.png)
