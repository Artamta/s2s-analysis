# S2S Verification Pipeline — Complete Documentation
## JFM 2026 | Model-Own Climatology (V2)

---

## 1. What This Pipeline Does (Big Picture)

We have **3 forecast models** (SPIRE, FuXi, ECMWF) that each issued hindcasts
starting every ~7 days from **1 Jan to 26 Mar 2026** (13 initialisation dates).
Each forecast runs for **42 days ahead**.

The pipeline asks: **"How well did each model predict the actual weather
(ERA5 reanalysis) over India?"**

It answers this for:
- **2 variables:** Total Precipitation (TP) and 500 hPa Geopotential Height (Z500)
- **6 weekly windows:** Week 1 (days 1–7), Week 2 (days 8–14), … Week 6 (days 36–42)
- **42 individual lead days** (daily track)
- **5 regions:** All India + 4 IMD homogeneous regions

Outputs are tidy CSV files of skill scores + reliability data.

---

## 2. Setup: Season, Inits, Grid

| Item | Value |
|---|---|
| Season | JFM 2026 (January–March 2026) |
| Init dates | 13 dates: 2026-01-01, 01-08, 01-15, 01-22, 01-29, 02-05, 02-12, 02-19, 02-26, 03-05, 03-12, 03-19, 03-26 |
| Forecast length | 42 days per init |
| Verification grid | 1.5° × 1.5°, lat 38°N–6.5°N (22 pts), lon 65°E–99.5°E (24 pts) |
| Land mask | Applied — ocean points set to NaN before scoring |
| Truth | ERA5 reanalysis (daily) |
| ERA5 clim | 30-year WMO daily climatology (DOY-mean) |

**Regions scored** (IMD homogeneous regions):
- All India (union of the 4 below)
- Northwest India
- Central India
- South Peninsula
- East & Northeast India

---

## 3. Models and Data Sources

### 3.1 SPIRE
- Delivered as **ensemble summary**: per-step mean + per-step stddev (zarr store)
- NOT individual members
- Anomalies in the zarr are **ERA5 1991–2020 referenced** (SPIRE's own reference)
- Deterministic track: uses the ensemble mean
- Probabilistic track: models as **Gaussian(mean, stddev)**
- TP units: mm/day (already in zarr)
- Z500 units: gpm (`geopotential_height_at_isobaric_levels` already in gpm)

### 3.2 FuXi
- **11 ensemble members**, individual netCDF files per member per day
- File path: `fuxi/output/{YYYYMMDD}/member/{mm}/{dd}.nc`
- TP units: **mm/hour** → multiplied by **24** → mm/day
- Z500 units: **geopotential (m²/s²)** → divided by **g = 9.80665** → gpm
- All members loaded at once (optimised: one file open per member-day, all channels)

### 3.3 ECMWF
- **~100 perturbed ensemble members** (pf), GRIB format
- TP variable: `tp`, GRIB units: `kg/m²` = **mm, cumulative from init**
- Z500 variable: `gh`, already in **gpm** (geopotential height)
- TP is processed as cumulative → differenced to get daily/weekly rates

### 3.4 MME (Multi-Model Ensemble)
- Simple average of SPIRE + FuXi + ECMWF **anomaly fields**
- Reconstructed as absolute field: `MME = mean_anomaly + ERA5_clim`
- Note: anomalies are on slightly mixed baselines (SPIRE~ERA5, FuXi/ECMWF=model-own)

### 3.5 Persistence
- Observed ERA5 mean over the **7 days immediately before init**
- Used as a naive reference forecast (does the last week predict the next week?)
- For init = 2026-01-01: uses Dec 25–31 2025 (special patch file)
- Scored against ERA5 clim as baseline (same as observations)

---

## 4. Unit Conversions (Critical)

All variables must be in the same units before any scoring.

| Variable | Model | Raw Units | Conversion | Final Units |
|---|---|---|---|---|
| TP | ERA5 truth | mm/day | none | mm/day |
| TP | SPIRE | mm/day | none | mm/day |
| TP | FuXi | mm/hour | × 24 | mm/day |
| TP | ECMWF | mm cumulative | differenced | mm/day |
| TP | ERA5 clim | m/day | × 1000 | mm/day |
| TP | FuXi clim | mm/hour cumul mean | × 24 | mm/day |
| TP | ECMWF clim | mm cumulative | differenced | mm/day |
| Z500 | ERA5 truth | m²/s² | ÷ 9.80665 | gpm |
| Z500 | SPIRE | gpm | none | gpm |
| Z500 | FuXi | m²/s² | ÷ 9.80665 | gpm |
| Z500 | ECMWF | gpm | none | gpm |
| Z500 | FuXi clim | m²/s² | ÷ 9.80665 | gpm |
| Z500 | ECMWF clim | gpm | none | gpm |

**Key rule:** The climatology file goes through the **exact same unit transform and
weekly aggregation** as the forecast. This guarantees `fcst - clim` is always
in the same units as `obs - clim`.

---

## 5. Weekly Aggregation

Forecast step numbers are **1-based lead days** (step 1 = day after init).

| Week | Lead Days | Step indices (0-based) | Obs calendar dates (for init Jan 1) |
|---|---|---|---|
| Week 1 | 1–7 | 0–6 | Jan 2 – Jan 8 |
| Week 2 | 8–14 | 7–13 | Jan 9 – Jan 15 |
| Week 3 | 15–21 | 14–20 | Jan 16 – Jan 22 |
| Week 4 | 22–28 | 21–27 | Jan 23 – Jan 29 |
| Week 5 | 29–35 | 28–34 | Jan 30 – Feb 5 |
| Week 6 | 36–42 | 35–41 | Feb 6 – Feb 12 |

**For Z500:** weekly mean = simple mean over the 7 daily fields

**For TP (ECMWF/clim, cumulative):**
```
Week 1:   cum[step=7]  / 7                          (mm/day)
Week N:  (cum[step=de] - cum[step=ds-1]) / 7        (mm/day)
```

**For TP (FuXi, rate):** mean of daily rates × 24

**Observation window** matches the forecast: `date_range(start=init+1day)[ds-1:de]`
(Bug fix: init day itself is NOT included — step 1 forecasts the day AFTER init.)

---

## 6. Anomaly Definition (Core Design Choice)

The key question: **what baseline do we subtract to get anomalies?**

Different models have different hindcast climatologies. We use:

| Model | Forecast anomaly | Obs anomaly |
|---|---|---|
| SPIRE | `SPIRE_mean − ERA5_clim(DOY)` | `ERA5 − ERA5_clim(DOY)` |
| FuXi | `FuXi_weekly_mean − FuXi_own_clim(lead)` | `ERA5 − ERA5_clim(DOY)` |
| ECMWF | `ECMWF_weekly_mean − ECMWF_own_clim(lead)` | `ERA5 − ERA5_clim(DOY)` |
| Persistence | `obs_pre_init − ERA5_clim(DOY)` | `ERA5 − ERA5_clim(DOY)` |

**Why model-own clim for FuXi/ECMWF?**
These models have systematic biases in their climatological mean that vary with
lead time. Scoring against their OWN hindcast climatology removes this bias
and measures pure **anomaly pattern skill** — the signal above and beyond what
the model's own mean state would predict.

**Why ERA5 clim for SPIRE?**
SPIRE's hindcast anomalies are already ERA5-1991–2020 referenced, so SPIRE's
own clim ≈ ERA5 clim by construction. Using ERA5 clim is consistent.

**ERA5 clim:** 30-year WMO daily climatology, DOY-averaged over the 7 valid days.

---

## 7. All Verification Metrics — Formulas

Let:
- `f` = forecast field (on verification grid, land-masked)
- `o` = ERA5 observation field
- `cf` = forecast climatology (model-own or ERA5 depending on model)
- `co` = ERA5 observation climatology
- `af = f − cf` = forecast anomaly
- `ao = o − co` = obs anomaly
- `w(φ)` = cosine latitude weight, normalised so mean(w) = 1.0
- `<x>` = area-weighted spatial mean = `Σ w(φ) x / Σ w(φ)`

### 7.1 Pattern Correlation (PCC / ACC)

WMO spatially-centred anomaly pattern correlation:

```
PCC = <(af − <af>)(ao − <ao>)> / sqrt(<(af−<af>)²> × <(ao−<ao>)²>)
```

- Range: −1 to +1. PCC > 0.6 generally considered "useful skill"
- Spatially centred (subtracts area-mean) → measures pattern shape, not mean shift
- Uses **model-own cf** for FuXi/ECMWF, **ERA5 co** for SPIRE/MME/Persistence

### 7.2 Root Mean Square Error (RMSE)

```
RMSE = sqrt( <(f − o)²> )
```

- Measures absolute error magnitude in forecast units (mm/day or gpm)
- Applied to **raw** (not anomaly) fields — includes both bias and pattern error

### 7.3 Bias

```
Bias = <f − o>
```

- Signed: positive = model too wet/high, negative = model too dry/low
- Applied to **raw** fields

### 7.4 MSE Skill Score (MSSS)

```
MSSS = 1 − MSE(f, o) / MSE(ERA5_clim, o)
```

- Reference is always **ERA5 climatology** (not model-own) → fair comparison across models
- MSSS > 0: model beats climatology
- MSSS = 1: perfect forecast
- MSSS < 0: model worse than just predicting climatology

### 7.5 Standard Deviation Ratio (Amplitude Fidelity)

```
std_ratio = std_w(af) / std_w(ao)
```

where `std_w(x) = sqrt(<(x − <x>)²>)` = weighted spatial std of anomaly

- std_ratio = 1: model anomaly amplitude matches observations
- std_ratio < 1: model under-dispersed (anomalies too weak)
- std_ratio > 1: model over-dispersed

---

## 8. Probabilistic Metrics

SPIRE, FuXi, ECMWF all treated as **Gaussian(μ, σ)** forecasts where:
- SPIRE: μ = provider mean, σ = provider stddev (from zarr)
- FuXi: μ = ensemble mean (11 members), σ = ensemble std
- ECMWF: μ = ensemble mean (~100 members), σ = ensemble std

### 8.1 Continuous Ranked Probability Score (CRPS)

For a Gaussian forecast N(μ, σ) vs scalar observation y (Gneiting & Raftery 2007):

```
CRPS(N(μ,σ), y) = σ × [ z(2Φ(z)−1) + 2φ(z) − 1/√π ]

where z = (y − μ) / σ
      Φ = standard normal CDF
      φ = standard normal PDF
```

- Lower CRPS = better forecast
- Units same as the variable (mm/day or gpm)
- Computed **per grid point**, then area-weighted mean

**Climatology reference CRPS:**
```
CRPS_clim = CRPS( N(ERA5_clim_mean, σ_clim), ERA5_obs )
```
where `σ_clim` = temporal std of **ERA5 anomalies** over the verification period
(seasonal trend removed — see Bug 3 fix)

**Persistence reference CRPS:**
```
CRPS_pers = CRPS( N(pers_field, sig_floor), ERA5_obs )
```
where sig_floor = 0.05 mm/day (TP) or 1.0 gpm (Z500) — effectively deterministic

### 8.2 CRPS Skill Score (CRPSS)

```
CRPSS = 1 − CRPS_model / CRPS_clim
```

- CRPSS > 0: model beats climatology in probabilistic sense
- CRPSS = 1: perfect probabilistic forecast
- CRPSS < 0: model worse than climatology

### 8.3 Spread-Skill Ratio (SSR)

```
SSR = mean(σ_ensemble) / RMSE(μ, obs)
```

- SSR = 1: perfectly calibrated (spread matches error)
- SSR < 1: under-dispersed (overconfident)
- SSR > 1: over-dispersed

---

## 9. Brier Score and Events

Binary events are defined from the Gaussian forecast:

**TP exceedance events:**
- `tp_gt_1mm`: P(TP > 1 mm/day)
- `tp_gt_10mm`: P(TP > 10 mm/day)

**Tercile events (all variables):**
- Above-normal: obs > clim_mean + 0.4307 × σ_clim (upper tercile boundary)
- Below-normal: obs < clim_mean − 0.4307 × σ_clim (lower tercile boundary)

Note: 0.4307 = norm.ppf(2/3) — the Gaussian z-score that puts exactly 1/3 of
probability above it.

**Forecast probability:**
```
P(above) = 1 − Φ( (hi − μ) / σ )
P(below) = Φ( (lo − μ) / σ )
```

### 9.1 Brier Score

```
BS = <(P_fcst − 1{obs ∈ event})²>
```

- Lower = better. Range 0 to 1.

**Climatology reference Brier:**
```
BS_clim = <(p_clim − 1{obs ∈ event})²>
```
where p_clim = 1/3 for tercile events (climatological frequency),
or `P(obs > thr | Gaussian clim)` for exceedance events.

### 9.2 Brier Skill Score (BSS)

```
BSS = 1 − BS_model / BS_clim
```

- BSS > 0: better than climatology
- BSS < 0: worse than climatology

### 9.3 Reliability

Forecast probabilities binned into 10 bins (0–0.1, 0.1–0.2, … 0.9–1.0).
For each bin: observed frequency vs mean forecast probability.
A perfectly reliable forecast: observed frequency = forecast probability (diagonal line).

---

## 10. Cosine Latitude Weighting

All spatial averages use **area-weighted** means:

```
w(φ) = cos(φ) / mean(cos(φ))    [mean normalised to 1.0]
```

This corrects for the convergence of meridians at higher latitudes
(grid cells near the equator cover more area than those near the poles).

Applied to: PCC, RMSE, Bias, MSSS, CRPS, Brier, SSR.

---

## 11. Pipeline Execution Flow

```
For each init date (13) — run in PARALLEL (13 worker processes):
│
├── Load SPIRE mean+stddev (zarr)
├── Load FuXi 11 members × 42 days (netCDF, all channels in one read)
├── Load ECMWF ~100 members × 42 steps (GRIB)
├── Load ERA5 truth (tp_daily.nc + z500.grib)
├── Load ERA5 30-yr clim (era5_climatology.nc)
├── Load model-own clim (FuXi: tp/z500_clima_{MMDD}.nc, ECMWF: same)
├── Compute SPIRE implied clim = mean_stddev − anomalies (zarr)
├── Compute persistence = mean(ERA5, 7 days before init)
├── Compute σ_clim = std(ERA5_anomaly, all available days)
│
├── WEEKLY LOOP (Weeks 1–6):
│   ├── Aggregate forecasts to 7-day means
│   ├── Aggregate clims to same 7-day means (same transforms)
│   ├── Compute obs = ERA5 mean over matching 7 days (init+ds to init+de)
│   ├── Compute ERA5 clim_o = mean of 30-yr DOY clims for those 7 days
│   └── For each region × each model:
│       ├── Deterministic: PCC, RMSE, Bias, MSSS, std_ratio
│       └── Probabilistic: CRPS, CRPSS, SSR, Brier, BSS, reliability
│
└── DAILY LOOP (days 1–42):
    ├── Extract single-day forecast fields
    └── Score: PCC, RMSE, Bias, MSSS (no Brier for daily)

Aggregate all 13 inits → write:
  skill_deterministic.csv   (31,200 rows)
  skill_probabilistic.csv   (18,720 rows)
  skill_brier.csv           (7,020 rows)
  reliability.npz
  RUN_METADATA.txt
```

---

## 12. Bugs Found and Fixed

| # | File | Bug | Effect | Fix |
|---|---|---|---|---|
| 1 | `loaders.py`, `verify_s2s.py` | Obs window started at init date instead of init+1 day | Forecast compared to wrong obs day | `date_range(start=init+1)` |
| 2 | `verify_s2s.py` | MSSS denominator used model-own clim instead of ERA5 clim | MSSS not comparable across models | Changed to always use `clim_o` |
| 3 | `loaders.py` | σ_clim computed from raw Z500 (seasonal trend inflated by ~15%) | CRPSS slightly too high for all models | Now uses `std(ERA5 − ERA5_clim)` |

---

## 13. What Each Output Column Means

### skill_deterministic.csv

| Column | Meaning |
|---|---|
| `pcc` | Pattern correlation (anomaly) — main skill metric |
| `rmse` | RMSE of raw forecast vs raw obs (mm/day or gpm) |
| `bias` | Mean(fcst − obs), signed |
| `msss_clim` | MSE skill score vs ERA5 climatology |
| `msss_pers` | MSE skill score vs persistence |
| `fcst_std` | Spatial std of forecast anomaly |
| `obs_std` | Spatial std of obs anomaly |
| `std_ratio` | fcst_std / obs_std (amplitude fidelity) |
| `fcst_mean` | Area-weighted mean of raw forecast |
| `obs_mean` | Area-weighted mean of raw ERA5 obs |

### skill_probabilistic.csv

| Column | Meaning |
|---|---|
| `crps` | CRPS of model (lower = better) |
| `crps_clim` | CRPS of climatology reference |
| `crps_pers` | CRPS of persistence reference |
| `crpss_clim` | 1 − crps/crps_clim (>0 = beats clim) |
| `crpss_pers` | 1 − crps/crps_pers (>0 = beats persistence) |
| `spread` | Mean ensemble spread (σ) |
| `rmse` | RMSE of ensemble mean |
| `ssr` | Spread-skill ratio = spread/rmse |

### skill_brier.csv

| Column | Meaning |
|---|---|
| `brier` | Brier score of model |
| `brier_clim` | Brier score of climatology reference |
| `briss_clim` | 1 − brier/brier_clim (BSS, >0 = beats clim) |
| `base_rate` | Observed frequency of the event in this sample |

---

## 14. Key References

- **PCC/MSSS:** Murphy (1988), *J. Climate* — skill scores for deterministic forecasts
- **CRPS (Gaussian):** Gneiting & Raftery (2007), *J. Am. Stat. Assoc.* — proper scoring rules
- **WMO verification:** WMO-No. 1186 (2021) — standard verification methods
- **Cosine weighting:** Standard practice for global/regional verification
- **S2S verification practice:** Vitart et al. (2017), *BAMS* — S2S prediction project
