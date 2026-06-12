# 🧠 S2S Pipeline: Master Context Restoration

**To the next AI Model:** Please read this entire document carefully. It contains the exact mathematical state, directory structures, and methodological decisions established in the previous session. 

## 1. Project Goal
The user is writing a high-impact paper comparing data-driven AI weather models (FuXi, Spire) against operational dynamical baselines (ECMWF, NCEP) for Subseasonal-to-Seasonal (S2S) forecasting over the Indian Subcontinent (JFM 2026).

## 2. Master Data Architecture (100% Verified)
All hundreds of gigabytes of data have been cleanly organized into `/storage/raj.ayush/s2s-forecast-data/`. All Python scripts in the workspace have been updated to target these paths.

* **ERA5 (Ground Truth):** `.../era5/data/` (1.5° Grid & 0.25° Hourly `.nc` grids. Vars: `t2m`, `tp`, `z`)
* **ECMWF:** `.../ecmwf/data/` (46 Days, 100 Perturbed Members. Vars: `mx2t6`, `tp`, `gh`)
* **NCEP:** `.../ncep/data/` (44 Days, 15 Perturbed Members. Vars: `mx2t6`, `tp`, `gh`)
* **FuXi-S2S (AI):** `.../fuxi/output/` (42 Days, 11 Ensemble Members. Vars: `t2m`, `tp`, `z500`)
* **Spire (AI):** `.../spire/*.zarr` (46 Days, Pre-computed Ensemble Mean/Std. Vars: `temperature`, `precipitation`, `geopotential`)

## 3. Mathematical Standardization Rules (Strict!)
The `data_loaders/model_loader.py` script enforces strict mathematical parity:
* **Temperature:** Everything is natively **Kelvin (K)**. *CRITICAL CAVEAT: S2S operational models do not archive instantaneous `t2m` beyond 24h. ECMWF/NCEP provide Daily Max (`mx2t6`) and Daily Min (`mn2t6`).*
* **Precipitation:** ECMWF/NCEP natively output `kg/m²` (accumulated). ERA5 outputs `m`. FuXi outputs `mm`. The loader temporally differentiates accumulations and standardizes EVERYTHING to **mm/day**.
* **Geopotential:** Spire natively outputs Geopotential Height (m). ECMWF/NCEP output Height (gpm). ERA5 and FuXi output Geopotential ($m^2/s^2$). The script multiplies Spire/ECMWF/NCEP by `9.80665` to strictly evaluate **Geopotential ($m^2/s^2$)**.

## 4. Current State of Scripts
We recently archived all obsolete phase 1 testing scripts. The `paper_results_pipeline/analysis/` folder now contains only the **Final Master Pipeline**:
1. `compute_final_wmo_z500.py` (Perfectly WMO-compliant Z500 Verification)
2. `compute_final_wmo_tp.py` (Perfectly WMO-compliant Precipitation Verification)
3. `plot_z500_bias_all_models.py` (Bias Mapping)
4. `plot_z500_anomalies_all_models.py` (Anomaly Correlation Mapping)
5. `plot_regional_tp.py` (IMD Regional Bar charts)

## 5. 🎯 Next Immediate Task
The user left off right before creating **`compute_final_wmo_t2m.py`**. 

**The Challenge to solve next:** You must evaluate Temperature predictability. However, because ECMWF and NCEP only provide `mx2t6` (Daily Max), you cannot compare this against instantaneous ERA5 `t2m`.
**The agreed upon solution:** The user has hourly ERA5 data stored in `/storage/raj.ayush/s2s-forecast-data/era5/data/*.nc`. You need to load this hourly data, extract the `.max(dim='time')` per day, and construct a perfect ERA5 Daily Max baseline to evaluate ECMWF/NCEP `mx2t6` (and FuXi/Spire `t2m` equivalents) against!

---
*Context generated on: 2026-06-11*
