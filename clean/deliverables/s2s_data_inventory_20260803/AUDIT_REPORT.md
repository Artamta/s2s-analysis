# S2S storage inventory

Generated: `2026-08-03T18:18:55.849159+00:00`

This is a read-only, manifest-first inventory. `manifest_valid` means the file passed its producing pipeline's validation and still has the expected identity metadata. Only records with `open_qc.status=opened` were reopened during this audit.

## Dataset summary

| experiment | product | years | ICs | files | members | leads | variables | ACC-ready |
|---|---|---:|---:|---:|---|---|---|---:|
| model-run/dlesym/dlesym_v0_isccp_era5_tpdiag_t2m_00z_2020_2024_ens1 | operational_forecast | 2020,2021,2022,2023,2024 | 517 | 517 | 1 | 42 | t2m,tp | 517 |
| model-run/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens16 | operational_forecast | 2020 | 1 | 1 | 16 | 42 | t2m | 1 |
| model-run/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens4 | operational_forecast | 2020,2021,2022,2023,2024 | 517 | 517 | 4 | 42 | t2m | 517 |
| model-run/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3 | operational_forecast | 2020,2021,2022,2023,2024 | 516 | 516 | 3 | 42 | t2m | 516 |
| model-run/fuxi/fuxi_s2s_gfs_proxy_case_20260722_ens100 | case_or_pilot_forecast | 2026 | 1 | 1 | 100 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_gfs_proxy_case_20260728_ens100 | case_or_pilot_forecast | 2026 | 1 | 1 | 100 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_gfs_proxy_case_20260728_ens24 | case_or_pilot_forecast | 2026 | 1 | 1 | 24 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_gfs_proxy_case_20260730_31_ens5 | case_or_pilot_forecast | 2026 | 2 | 2 | 5 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260601_ens50 | case_or_pilot_forecast | 2026 | 1 | 1 | 50 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260623_ens50 | case_or_pilot_forecast | 2026 | 1 | 1 | 50 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260715_ens100 | case_or_pilot_forecast | 2026 | 1 | 1 | 100 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260715_wind_ens100 | case_or_pilot_forecast | 2026 | 1 | 1 | 100 | 42 | u850,v850,tp | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260722_ens100 | case_or_pilot_forecast | 2026 | 1 | 1 | 100 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_case_20260722_ens1_trial | case_or_pilot_forecast | 2026 | 1 | 1 | 1 | 42 | tp,t2m | 0 |
| model-run/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 621 | 621 | 50 | 42 | tp,t2m | 621 |
| model-run/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50 | operational_forecast | 2020,2021,2022,2023 | 373 | 373 | 50 | 42 | tp,t2m | 369 |
| model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens1 | operational_forecast | 2020 | 1 | 1 | 1 | 42 | tp | 1 |
| model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens10 | operational_forecast | 2020,2021,2022,2023,2024 | 517 | 517 | 10 | 42 | tp | 517 |
| model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_pilot42d_20200601_ens1 | case_or_pilot_forecast | 2020 | 1 | 1 | 1 | 42 | tp | 0 |
| model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_smoke_20200601_ens1 | case_or_pilot_forecast | 2020 | 1 | 1 | unknown | unknown | unknown | 0 |
| physics/cma_operational_2020_2025 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 621 | 2484 | 1,3 | 42 | t2m,tp | 621 |
| physics/cnrm_operational_2020_2025 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 269 | 1076 | 1,24 | 42 | t2m,tp | 269 |
| physics/ecmwf_operational_2020_2025 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 621 | 2484 | 1,100,50 | 42 | t2m,tp | 621 |
| physics/ncep_operational_2020_2025 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 621 | 1242 | 1,15 | 42 | surface | 621 |
| physics/ukmo_operational_2020_2025 | operational_forecast | 2020,2021,2022,2023,2024,2025 | 621 | 2484 | 1,3 | 42 | t2m,tp | 621 |
| provider/erpas_forecast_2023_2025 | provider_ensemble_mean_forecast | 2023,2024,2025 | 148 | 591 | 1 | 33 | geopotential_height,surface_temperature,tp,tp_india_0p5 | 148 |
| reforecast/cnrm_native_climatology | native_reforecast_or_smoke_test | 2019 | 1 | 1 | unknown | 1 | tp | 0 |
| reforecast/ecmwf_native_climatology | native_reforecast_or_smoke_test | 2020 | 1 | 1 | unknown | 1 | tp | 0 |
| reforecast/fuxi_native_2002_2021 | native_reforecast_archive | 2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021 | 2080 | 2080 | 51 | 42 | 76_native_channels | 0 |
| reforecast/ukmo_native_climatology | native_reforecast_or_smoke_test | 2020 | 1 | 1 | unknown | 1 | tp | 0 |

## Integrity findings

Files with inventory issues: **4**.

- `missing`: 4

## Sampled metadata-open checks

- `not_run`: 15466
- `opened`: 49
- `timed_out`: 5

A timeout means the decoder exceeded the bounded audit window; it is not classified as corruption.

## Calendar gaps

- `model-run/dlesym/dlesym_v1_era5_t2m_00z_2020_2024_ens16` 2020: 104 missing expected IC(s): 2020-01-02, 2020-01-06, 2020-01-09, 2020-01-13, 2020-01-16, 2020-01-20, 2020-01-23, 2020-01-27, 2020-01-30, 2020-02-03, 2020-02-06, 2020-02-10 (+92 more in inventory.json)
- `model-run/fcn3/fcn3_v1_t2m_00z_2020_2024_ens3` 2021: 1 missing expected IC(s): 2021-03-08
- `model-run/fuxi/fuxi_s2s_twice_weekly_2020_2025_ens50` 2023: 44 missing expected IC(s): 2023-06-19, 2023-06-22, 2023-06-26, 2023-06-29, 2023-07-03, 2023-07-06, 2023-07-10, 2023-07-13, 2023-07-17, 2023-07-20, 2023-07-24, 2023-07-27 (+32 more in inventory.json)
- `model-run/neural-gcm/neuralgcm_v1_precip_2p8_era5_00z_2020_2024_ens1` 2020: 104 missing expected IC(s): 2020-01-06, 2020-01-09, 2020-01-13, 2020-01-16, 2020-01-20, 2020-01-23, 2020-01-27, 2020-01-30, 2020-02-03, 2020-02-06, 2020-02-10, 2020-02-13 (+92 more in inventory.json)

See `inventory.json` for initialization- and file-level records, `coverage_matrix.csv` for year coverage, and `file_issues.csv` for exact exclusions.
