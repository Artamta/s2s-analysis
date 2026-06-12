# Reproducibility code — *Subseasonal forecast benchmarking over India (JFM 2026)*

This directory contains the **curated, review-ready code** that produces every
figure and table in the manuscript. It is a clean snapshot of the working
scripts in the `../../analysis-code/analysis/` working folder, reorganised so each manuscript figure maps to exactly
one script.

```
manuscript_code/
├── README.md                 ← this file
├── run_all.sh                ← runs the whole pipeline (compute → figures)
├── utils/                    ← shared verification helpers (imported by all scripts)
│   ├── verification_wmo.py     cosine-weighted ACC / RMSE / bias (WMO formulas)
│   ├── verification_extra.py    land mask, bootstrap + paired-bootstrap CIs
│   └── spatial_masking.py
├── compute/                  ← data pipeline: raw forecasts → skill tables (run first)
└── figures/                  ← plotting: skill tables → manuscript PDFs (run second)
```

## Data flow

```
 raw forecasts (/storage/.../s2s-forecast-data)   ARCO-ERA5 (cloud)
            │                                            │
            ▼                                            ▼
        compute/*  ───────────►  ../../analysis-code/analysis/*.csv, *.nc, *.npz
                                          │
                                          ▼
        figures/*  ───────────►  ../figs/fig*.pdf  +  ../../analysis-code/analysis/tables_regional.tex
```

- **Inputs.** Forecasts (SPIRE Zarr, FuXi NetCDF, ECMWF/NCEP GRIB) live under
  `/storage/raj.ayush/s2s-forecast-data/`; the ERA5 daily references are built
  from the public ARCO-ERA5 archive. Paths are set at the top of each script.
- **Intermediates** (skill tables, anomaly fields) are written to `../../analysis-code/analysis/`.
- **Outputs** (figure PDFs/PNGs, the regional LaTeX table) go to `../figs/`.

## Method summary (what the code implements)

- Verification on Indian **land points only** (coastline mask), **cosine-latitude
  weighted**, on a common **1.5° grid**; **ensemble means** for deterministic scores.
- Skill metrics: spatially **centered** anomaly pattern correlation (PCC), RMSE,
  mean bias; anomalies use an **in-sample 2026 daily climatology** for all variables.
- **Precipitation** is verified against a **true 24-h daily** ERA5 total (built from
  ARCO hourly), and FuXi-S2S precipitation is **unit-harmonised** (mean rate
  mm h⁻¹ → mm day⁻¹, ×24) before scoring.
- **Confidence intervals** are bootstrap over the 13 init dates; system
  differences use a **paired** bootstrap.
- **Probabilistic** scores treat each system as a Gaussian(ensemble-mean, spread)
  forecast and report CRPS/CRPSS, the spread-skill ratio, and reliability.

## Compute pipeline (run order)

| # | Script | Produces (in `../../analysis-code/analysis/`) |
|---|--------|------------------------------|
| 1 | `compute/01_build_era5_daily_tp.py`          | `era5_daily_tp.nc` (true 24-h ERA5 precip) |
| 2 | `compute/02_build_era5_daily_t2m.py`         | `era5_daily_t2m.nc` |
| 3 | `compute/03_compute_skill_z500_tp.py`        | `skill_per_init_full.csv`, `weekly_anom_fields.nc` |
| 4 | `compute/04_correct_precip_vs_daily_era5.py` | `skill_tp_corrected.csv` (TP vs true daily ERA5) |
| 5 | `compute/05_compute_skill_t2m.py`            | `skill_t2m.csv` |
| 6 | `compute/06_compute_probabilistic.py`        | `prob_skill.csv`, `reliability.npz`, `prob_daily_regional.npz`, `prob_fields.nc` |
| 7 | `compute/07_dump_scatter_points.py`          | `scatter_points.npz` |

## Figure → script map

| Manuscript | Script (`figures/`) | Output file(s) |
|---|---|---|
| Fig 1 (domain)                       | `make_domain_and_bias.py`        | `fig01_domain` |
| Fig 2 (skill horizon)                | `make_skill_horizon.py`          | `fig02_skill_horizon` |
| Fig 3 (SPIRE − FuXi)                  | `make_skill_horizon.py`          | `fig03_spire_vs_fuxi` |
| Fig 4 (regional scorecard)           | `make_skill_horizon.py`          | `fig04_regional_scorecard` |
| Fig 5 (bias diagnostics)             | `make_domain_and_bias.py`        | `fig05_bias` |
| Fig 6 (MME + persistence)            | `make_mme_spatial_regional.py`   | `fig06_mme_persistence` |
| Fig 7 (spatial PCC, Z500)            | `make_mme_spatial_regional.py`   | `fig07_spatial_pcc_z500` |
| Fig 8 (spatial bias, Z500)           | `make_mme_spatial_regional.py`   | `fig08_spatial_bias_z500` |
| Fig 9 (regional horizon)             | `make_mme_spatial_regional.py`   | `fig09_regional_horizon` |
| Fig 10 (circulation case study)      | `make_circulation_case_study.py` | `fig10_case_study` |
| Fig 11 (forecast–obs density scatter)| `make_density_scatter.py`        | `fig11_scatter_density` |
| Figs 12–13 (appendix error maps)     | `make_appendix_error_maps.py`    | `fig12_*`, `fig13_*` |
| Figs 15–17 (CRPSS, spread-skill, reliability) | `make_probabilistic_skill.py` | `fig15_*`, `fig16_*`, `fig17_*` |
| Fig 18 (event spread)                | `make_event_spread.py`           | `fig18_event_spread` |
| Fig 19 (region-wise PCC/RMSE)        | `make_regional_bars.py`          | `fig19_regional_bars` |
| Figs 20–22 (week-6 case studies)     | `make_week6_case_studies.py`     | `fig20_*`, `fig21_*`, `fig22_*` |
| Figs 23–25 (regional case series)    | `make_regional_case_series.py`   | `fig23_*`, `fig24_*`, `fig25_*` |
| Figs 26–27 (SPIRE distribution gallery) | `make_spire_gallery.py`       | `fig26_*`, `fig27_*` |
| Tables (regional RMSE/PCC)           | `make_regional_tables.py`        | `tables_regional.tex` |

*(Figure 14, the Taylor diagram, was removed from the manuscript and is not
produced here.)*

## How to reproduce

```bash
# from paper_results_pipeline/
bash manuscript_code/run_all.sh          # full pipeline: compute then all figures
```

Or run a single figure once its inputs exist, e.g.:

```bash
python manuscript_code/figures/make_skill_horizon.py    # Figs 2, 3, 4
```

## Dependencies

Python 3.11 with `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`, `cartopy`,
`cfgrib`/`eccodes`, `zarr`, `gcsfs` (ARCO-ERA5), and `global-land-mask`.
