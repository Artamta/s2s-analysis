# Spire AI-S2S Paper — Figures & Analysis

## Overview
Publication-quality verification of **Spire AI-S2S** subseasonal forecasts against **ERA5** reanalysis for **JFM 2026** (January–March), 90 daily initializations over the Indian domain.

## Directory Structure

```
spire-s2s-paper/
├── data/                          # Processed datasets
│   ├── weekly_anomalies_v2.nc     # → symlink to original (T2m, Precip, Z500)
│   ├── weekly_anomalies_extended.nc  # OLR, U850, U200 anomalies
│   └── equatorial_hovmoller.nc    # Equatorial-band daily data (MJO diagnostics)
│
├── scripts/                       # All Python/bash scripts
│   ├── extract_mjo_extended.py    # Step 1: Extract OLR/U850/U200 from ArrayLake+ERA5
│   ├── fig_deterministic.py       # Step 2: Figs 01-05 (ACC/RMSE/Bias line + maps)
│   ├── fig_synthesis.py           # Step 3: Figs 06-10 (scatter, scorecard, dashboard)
│   ├── fig_mjo_extended.py        # Step 4: Figs 11-17 (MJO/OLR/U850 diagnostics)
│   └── run_all.sh                 # Master script (bash run_all.sh [step])
│
├── figures/                       # All outputs (PNG 300dpi + PDF)
│   ├── deterministic/             # Figs 01-05
│   ├── scatter/                   # Figs 06-07
│   ├── synthesis/                 # Figs 08-10
│   ├── mjo/                       # Figs 11-17
│   └── *.png / *.pdf              # (flat copies also kept here)
│
└── README.md                      # This file
```

## Figure Catalogue

### Deterministic Verification (Figs 01–05)
| Fig | File | Description |
|-----|------|-------------|
| 01 | `fig01_acc_vs_lead` | ACC vs forecast lead (T2m-mean, T2m-max, Precip, Z500) |
| 02 | `fig02_rmse_vs_lead` | RMSE vs lead: (a) Temperature, (b) Precip & Z500 |
| 03 | `fig03_bias_vs_lead` | Bias vs lead: (a) Temperature, (b) Precip & Z500 |
| 04 | `fig04_acc_skill_maps` | Spatial ACC maps (3 vars × W1/W3/W6) with stippling |
| 05 | `fig05_bias_maps` | Spatial bias maps (T2m-mean/max × W1/W3/W6) |

### Scatter & Synthesis (Figs 06–10)
| Fig | File | Description |
|-----|------|-------------|
| 06 | `fig06_scatter_gridcell` | Grid-cell hexbin scatter (4 vars × W1/W3/W6, R²/RMSE) |
| 07 | `fig07_scatter_initmean` | Init-mean scatter, month-colored (Jan/Feb/Mar) |
| 08 | `fig08_scorecard_heatmap` | ACC + RMSE scorecard (4 vars × 6 weeks) |
| 09 | `fig09_anomaly_timeseries` | Spire vs ERA5 anomaly time series with ±1σ |
| 10 | `fig10_verification_dashboard` | 6-panel synthesis dashboard |

### MJO & Extended Diagnostics (Figs 11–17)
| Fig | File | Description |
|-----|------|-------------|
| 11 | `fig11_acc_vs_lead_mjo` | ACC vs lead for OLR, U850, U200 |
| 12 | `fig12_rmse_vs_lead_mjo` | RMSE vs lead for MJO variables |
| 13 | `fig13_acc_maps_mjo` | Spatial ACC maps (OLR/U850/U200 × W1/W3/W6) |
| 14 | `fig14_hovmoller_olr` | Equatorial OLR Hovmöller (Spire vs ERA5) |
| 15 | `fig15_hovmoller_u850` | Equatorial U850 Hovmöller |
| 16 | `fig16_mjo_propagation_skill` | MJO equatorial pattern correlation vs lead |
| 17 | `fig17_extended_scorecard` | 7-variable × 6-week ACC scorecard |

## Data Sources
- **Spire AI-S2S**: ArrayLake repo `artamta/s2s-research` (90 inits, 46-day leads, global 0.5°)
- **ERA5**: ARCO-ERA5 GCS Zarr store (hourly, 0.25°, interpolated to Spire grid)
- **Climatology**: WB2 1990-2019 (T2m/Z500/Precip), ERA5 1991-2020 Tmax

## Quick Start
```bash
# Generate all figures (if data already extracted):
cd spire-s2s-paper
bash scripts/run_all.sh

# Generate specific step only:
bash scripts/run_all.sh 4   # MJO figures only
```

## Notes on MJO/SST/IOD
- **MJO**: Full RMM index computation is not possible with India-only domain.
  Instead, we provide equatorial-band Hovmöller diagrams (OLR, U850) and
  propagation skill metrics over the Indian sector.
- **SST**: Not available in Spire AI-S2S archive (no sea surface temperature variable).
- **IOD**: Cannot compute DMI — domain does not extend south of equator (Southern IO poles missing).
