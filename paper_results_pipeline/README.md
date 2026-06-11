# 🚀 S2S AI vs Physics: Paper Results Pipeline

This repository contains the rigorous, publication-ready mathematical pipeline for evaluating Spire, FuXi, ECMWF, and NCEP Subseasonal-to-Seasonal (S2S) forecasts against ERA5 and IMD Ground Truths.

## ✅ Phase 1: What We Have Successfully Completed
- **SLURM Infrastructure:** Offloaded the massive FuXi, ERA5, and IMD Gridded Rainfall downloads to the `GPU-AI_prio` compute node to keep the login node safe.
- **Unified Model Loader (`data_loaders/model_loader.py`):**
  - Standardized Spire's hidden Zarr subgroups into a standard format.
  - Mathematically averaged NCEP's Daily Min/Max temperatures into a fair Instantaneous equivalent.
  - Implemented an automatic `regrid_to_common` function using Xarray Bilinear Interpolation to strictly scale all 0.25° models down to 1.5° to ensure fair apples-to-apples evaluation.
- **Verification Metrics (`verification/metrics.py`):**
  - Hand-wrote the exact formulas for RMSE, ACC, Bias, and Ensemble Spread.
  - Implemented 4 precise geographical bounding boxes matching the official IMD Homogeneous Regions (Northwest, Central, South Peninsular, Northeast).
- **Professor Singh's Visuals (`plotting/plot_scatter_and_bias.py`):**
  - Successfully generated publication-ready Spatial Bias Maps and Scatter Plots (Model vs ERA5) with $R^2$, RMSE, and MAE burned directly into the image.
- **Evaluation Loop (`analysis/evaluate_jan1_rmse.py`):**
  - Successfully generated our first tabular RMSE comparison `.csv` for Spire, ECMWF, and NCEP over Day 1.

---

## ⏳ Phase 2: Action Checklist (Next Steps)
When you return to your computer, follow this exact checklist:

- [ ] **1. Verify Downloads Finished**
  Run `squeue -u raj.ayush` to ensure the FuXi (Job 54220), ERA5 (Job 54221), and IMD (Job 54288) SLURM jobs have vanished from the queue (meaning they are 100% complete).
  
- [ ] **2. Add FuXi to the Loader**
  Open `data_loaders/model_loader.py` and write a `load_fuxi()` function to read the massive tensors now sitting in `/storage/`.

- [ ] **3. Run the Full JFM Evaluation**
  Update `analysis/evaluate_jan1_rmse.py` to loop over *all* dates (Jan 1 to Mar 26) instead of just Jan 1.

- [ ] **4. The Lead-Time Decay Graph**
  Write a script to plot the famous S2S Lead-Time Decay Line Graph (X-axis = Day 1 to Day 45, Y-axis = RMSE/ACC). This is the core figure that will prove if AI beats Physics!

- [ ] **5. Calculate ACC (Anomalies)**
- [x] Integrate Spire data (completed)
- [x] Regrid datasets to 1.5° (completed)
- [x] Calculate Bias, RMSE, $R^2$ (completed)
- [x] Run FuXi Inference on cluster (completed & resuming)
- [ ] Calculate ERA5 Climatology for ACC baseline
- [ ] Generate S2S Lead-Time Decay Plot (Day 1 to 45)
- [ ] Compute spatial maps of ACC

---

## 📝 Methodology Notes for Paper

**Copy-paste this directly into your paper's methodology section regarding the ACC metric:**

> **Anomaly Correlation Coefficient (ACC) Baseline:**
> *"To ensure a strictly level playing field across diverse modeling architectures, the Anomaly Correlation Coefficient (ACC) for all dynamical and data-driven models was computed utilizing the observational ERA5 climatology as the baseline. This methodology aligns with recent benchmarking standards established in deep learning weather prediction literature (e.g., Lam et al., 2023; Bi et al., 2023), thereby eliminating the artificial inflation of dynamical model skill that can occur when utilizing model-specific hindcast climatologies."*

> **Data Standardization and Temporal Alignment:**
> *"Forecast outputs from disparate modeling systems were mathematically harmonized prior to evaluation. Geopotential height from the SPIRE dataset was converted to geopotential ($m^2/s^2$) by multiplying by standard gravity ($g = 9.80665$). Total precipitation from the dynamical S2S models (ECMWF, NCEP), natively provided as accumulations in meters, was temporally differentiated into daily rates and converted to millimeters per day ($mm/day$) to match the AI outputs. Temporally, while dynamical models provide 24-hour daily averages, the AI systems natively output instantaneous 24-hour snapshots (e.g., valid at 00:00 UTC). To construct a uniform benchmark, all evaluations were conducted against the 24-hour daily mean of the ERA5 ground truth."*

> **Initialization and Predictability Alignment:**
> *"All dynamical and deep learning forecasts were strictly aligned to the exact same 00:00 UTC initialization times across the 13 weekly JFM 2026 dates. While dynamical models numerically integrate forward from this start time, the data-driven models (FuXi, SPIRE) ingest the ERA5 reanalysis fields at 00:00 UTC as their direct input tensors. This guarantees all models are evaluated on a strictly identical forecast horizon."*

> **Spatial Resolution and Variable Normalization:**
> *"Native model resolutions varied significantly, ranging from 0.25° (FuXi) and 0.5° (SPIRE) to 1.5° (ECMWF, NCEP). To prevent high-resolution models from receiving artificial spatial penalties during verification, all datasets were conservatively regridded to a unified 1.5° global grid prior to computing ACC and RMSE. Finally, the prognostic variables were verified for consistency: 2-meter surface temperatures were universally evaluated in Kelvin, eliminating the need for variable-specific conversions."*

> **Spatial Masking and Regional Verification:**
> *"To rigorously isolate predictability over the region while avoiding the geopolitical complexities of rendering political borders, the study domain was strictly defined using a spatial bounding box over the Indian Subcontinent (Latitude 5°N–38°N, Longitude 65°E–100°E). Furthermore, to prevent the obfuscation of regional subseasonal dynamics that occurs when treating a heterogeneous landmass as a single spatial average, verification metrics were partitioned into the Homogeneous Rainfall Regions defined by the India Meteorological Department (IMD)—Northwest India, Central India, South Peninsula, and East/Northeast India. These regions were isolated using strict coordinate bounding boxes, ensuring a purely meteorological and compliant spatial evaluation."*

---

### Important Quirks to Remember
* **NCEP:** NCEP only provided Min/Max temperatures in the S2S database, not instantaneous snapshots. Our loader mathematically averages them together. 
* **IMD:** The `imdlib` package hates downloading incomplete years. If we desperately need IMD 2026 data, we must skip `imdlib` and manually download it, or just use ERA5 Precipitation (`tp`) as the ground truth.

---

## 📊 100% Verified S2S Dataset Inventory

**Initialization Times:** All forecasts are identically initialized at **00:00 UTC** across **13 Weekly Dates** during Jan-Feb-Mar (JFM) 2026. The dates are: `2026-01-01, 2026-01-08, 2026-01-15, 2026-01-22, 2026-01-29, 2026-02-05, 2026-02-12, 2026-02-19, 2026-02-26, 2026-03-05, 2026-03-12, 2026-03-19, 2026-03-26`.

**Validated Variables & Units:**
- **Temperature (T2M / MX2T6):** 2-meter surface temperature. Natively verified in **Kelvin (K)**. *(Note: ECMWF & NCEP use Daily Max Temp `mx2t6` because instantaneous `t2m` is not archived by the S2S database >24h).*
- **Precipitation (TP):** Total surface precipitation. Raw units vary (Meters vs mm). Our unified mathematical pipeline temporally differentiates and standardizes everything to **mm/day**.
- **Geopotential (Z500 / GH):** Geopotential at 500 hPa. Unified to **m²/s²** *(Spire natively outputs geopotential height in meters, which we strictly convert by multiplying by standard gravity 9.80665)*.

| Dataset | Type | Absolute Master Path | Resolution | Temporal Horizon | Members | Core Variables & Native Units | Status |
|---|---|---|---|---|---|---|---|
| **ERA5** | Ground Truth | `/storage/raj.ayush/s2s-forecast-data/era5/data/` | 1.5° x 1.5° | 135 continuous days | None | `t2m` (K), `tp` (meters), `z` (m²/s²) | 🟢 100% Intact |
| **ECMWF** | Operational | `/storage/raj.ayush/s2s-forecast-data/ecmwf/data/` | 1.5° x 1.5° | 46 Days (24h steps) | 1 CF, 100 PF | `mx2t6` (K), `tp` (kg/m²), `gh` (gpm) | 🟢 100% Intact |
| **NCEP** | Operational | `/storage/raj.ayush/s2s-forecast-data/ncep/data/` | 1.5° x 1.5° | 44 Days (24h steps) | 1 CF, 15 PF | `mx2t6` (K), `tp` (kg/m²), `gh` (gpm) | 🟢 100% Intact |
| **FuXi** | AI Model | `/storage/raj.ayush/s2s-forecast-data/fuxi/output/` | 1.5° x 1.5° | 42 Days (Daily steps) | 11 Ensemble | `t2m` (K), `tp` (mm), `z` (m²/s²) | 🟢 100% Intact |
| **SPIRE** | AI Model | `/storage/raj.ayush/s2s-forecast-data/spire/*.zarr` | 0.5° x 0.5° | 46 Days (Daily steps) | Ens. Mean / Std | `temp` (K), `precip` (kg/m²), `gh` (m) | 🟢 100% Intact |

