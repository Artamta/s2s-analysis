# S2S Hindcast Verification — Implementation Guide

**Goal:** Reproduce BAMS (Vitart 2017) S2S verification framework using Spire JFM 2026 hindcast vs ERA5.

---

## What we know about the data

### Spire (arraylake `artamta/s2s-research`)
- 90 daily init dates: 2026-01-01 → 2026-03-31
- Lead: 1–46 days, global 0.5° grid (361 lat × 720 lon)
- **Key group: `mean_stddev`** — raw ensemble mean + stddev
  - `air_temperature` (2m, K)
  - `air_temperature_max`, `air_temperature_min` (K)
  - `precipitation_amount` (kg/m²/day = mm/day)
  - `geopotential_height_at_isobaric_levels` (m, 4 isobar levels — check which ones)
  - `eastward_wind_at_isobaric_levels`, `northward_wind_at_isobaric_levels` (m/s)
  - `specific_humidity_at_isobaric_levels` (kg/kg)
  - `ttr` (top thermal radiation, W/m² — OLR proxy for MJO)
  - `air_pressure_at_sea_level` (Pa)
  - `wind_speed` (10m, m/s)
- **Key group: `anomalies`** — anomalies already relative to ERA5 1991–2020 climo
  - `geopotential_height` and `temperature` (pressure-level vars confirmed anomaly)
  - Surface vars may also be anomaly — check by comparing with `mean_stddev`
- **Key group: `probabilities`** — tercile/quintile/decile probs vs ERA5 1991–2020
  - `precipitation_amount_prob_upper_tercile`, `_lower_tercile` etc.
  - `air_temperature_prob_upper_tercile` etc.
  - Use these for BSS/RPSS computation directly — no manual tercile calc needed

### ERA5 (ARCO zarr, public)
- URL: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- Storage: `{"token": "anon"}`
- 0.25° grid — **must interpolate to Spire 0.5° grid before comparing**
- Key variables: `total_precipitation` (m/hr → *1000 for mm/day), `2m_temperature` (K → -273.15 for °C), `geopotential` (J/kg → /9.80665 for gpm)

### ⚠️ Dimension mismatch fix (always do this)
```python
era5_da.interp(latitude=spire_lat, longitude=spire_lon, method="linear")
```

---

## Scripts to write (in order)

### Script 1: `01_era5_jfm_climatology.py`
**What:** ERA5 1991–2020 JFM day-of-year (DOY) climatology for India domain.

**Why needed:** To compute ERA5 obs anomalies for the verification step.

**Domain:** lat 0–50°N, lon 55–105°E

**Variables to include:**
- `total_precipitation` → daily sum (mm/day) = resample('1D').sum() * 1000
- `2m_temperature` → daily mean (°C) = resample('1D').mean() - 273.15
- `geopotential` at level=500 → daily mean (gpm) = resample('1D').mean() / 9.80665

**Output:** `era5_jfm_climatology_1991_2020.nc`  
Dimensions: `(dayofyear, latitude, longitude)` — latitude on ERA5 native 0.25° grid

**Logic:**
```python
ds_jfm = ds.sel(time=slice("1991-01-01","2020-12-31"))
ds_jfm = ds_jfm.where(ds_jfm.time.dt.month.isin([1,2,3]), drop=True)
# compute daily aggregates, then:
climo = daily_ds.groupby("time.dayofyear").mean("time").compute()
climo.to_netcdf("era5_jfm_climatology_1991_2020.nc")
```

**NOTE:** Can also try Earth2 Studio's climatology if available. Skip this script if a pre-computed ERA5 climatology file already exists (check `IMD_Fuxi/era5_master_climatology/` first).

---

### Script 2: `02_compute_weekly_anomalies.py`
**What:** For all 90 Spire init dates, compute weekly-mean Spire forecast anomalies and paired ERA5 obs anomalies.

**Weekly windows:**
- W1 = days 1–7  (lead days 1 to 7, mean)
- W2 = days 8–14
- W3 = days 15–21
- W4 = days 22–28

**Spire anomaly computation:**
```python
# Load Spire mean_stddev
session = client.get_repo("artamta/s2s-research").readonly_session("main")
ds_spire = xr.open_zarr(session.store, group="mean_stddev")

# Select India domain
ds_india = ds_spire.sel(
    latitude=slice(50, 0),   # check if lat is descending or ascending!
    longitude=slice(55, 105)
)

# Weekly mean: e.g. W1
steps_w1 = [np.timedelta64(d, 'D') for d in range(1, 8)]
spire_w1 = ds_india["air_temperature"].sel(step=steps_w1).mean("step")
# shape: (reference_time=90, lat, lon)

# Spire anomaly = forecast - mean over all 90 inits for same DOY
# Simple: subtract the mean over all reference_times (climo = mean across inits for same DOY band)
spire_climo_w1 = spire_w1.mean("reference_time")  # rough climo
spire_anom_w1 = spire_w1 - spire_climo_w1
```

**ERA5 obs anomaly computation:**
```python
# For each init date + weekly window, compute valid date range
# Load ERA5 for valid dates, compute weekly mean, subtract ERA5 DOY climo

climo = xr.open_dataset("era5_jfm_climatology_1991_2020.nc")

# For init_date + W1 (days 1-7):
valid_dates = pd.date_range(init_date + timedelta(1), init_date + timedelta(7))
era5_obs = ds_era5["2m_temperature"].sel(time=valid_dates).mean("time") - 273.15
era5_obs_on_spire_grid = era5_obs.interp(latitude=spire_lat, longitude=spire_lon)

# ERA5 climo for those DOYs
doys = [d.day_of_year for d in valid_dates]
era5_climo_vals = climo["t2m"].sel(dayofyear=doys).mean("dayofyear")
era5_climo_on_spire = era5_climo_vals.interp(latitude=spire_lat, longitude=spire_lon)

era5_anom = era5_obs_on_spire_grid - era5_climo_on_spire
```

**Output:** `weekly_anomalies.nc`
```
Dimensions: reference_time=90, week=4, lat, lon
Variables: spire_t2m_anom, era5_t2m_anom, spire_precip_anom, era5_precip_anom, spire_z500_anom, era5_z500_anom
```

---

### Script 3: `03_skill_maps.py`
**What:** ACC, RMSE, bias maps from `weekly_anomalies.nc`.

**ACC computation (Pearson CC across 90 init dates at each grid point):**
```python
from scipy.stats import pearsonr
import numpy as np

ds = xr.open_dataset("weekly_anomalies.nc")
# For T2M, W1:
spire = ds["spire_t2m_anom"].sel(week=1).values  # shape (90, lat, lon)
era5  = ds["era5_t2m_anom"].sel(week=1).values   # shape (90, lat, lon)

acc = np.zeros((len(lat), len(lon)))
for i in range(len(lat)):
    for j in range(len(lon)):
        cc, _ = pearsonr(spire[:, i, j], era5[:, i, j])
        acc[i, j] = cc
```

OR more efficiently using xarray:
```python
# Vectorized via scipy/numpy
from scipy.stats import pearsonr
acc_t2m_w1 = xr.apply_ufunc(
    lambda x, y: np.corrcoef(x, y)[0, 1],
    ds["spire_t2m_anom"].sel(week=1),
    ds["era5_t2m_anom"].sel(week=1),
    input_core_dims=[["reference_time"], ["reference_time"]],
    vectorize=True,
)
```

**Output:** `skill_metrics.nc`
```
Variables: acc_t2m(week, lat, lon), acc_precip, acc_z500, rmse_t2m, bias_t2m, ...
```

---

### Script 4: `04_plot_bams_fig1.py` ← FIRST FIGURE TO MAKE
**What:** BAMS Fig 1 equivalent — T2M anomaly spatial maps.

**Layout: 3 rows × 4 cols**
- Row 1: Spire T2M anomaly forecast (W1, W2, W3, W4)
- Row 2: ERA5 T2M anomaly verification (same valid period)
- Row 3: Bias = Spire - ERA5

**Pick a good init date:** Find init date in Jan-Mar 2026 when there was an MJO event (check Indian Ocean convection in TTR anomaly). Suggest: check init dates around Jan 15-20 or Feb 10-15 2026.

**Map style:** Copy from `IMD_Fuxi/imd_plots.py`:
- `cfeature.COASTLINE`, `cfeature.BORDERS`, NaturalEarth state lines
- `ccrs.PlateCarree()` projection
- Colormap: `RdBu_r` for anomalies (symmetric around 0, e.g. ±4°C for T2M)

**Key code pattern (from existing `imd_plots.py:50-79`):**
```python
ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', facecolor='none'), linewidth=0.6, edgecolor='black')
```

---

### Script 5: `05_plot_acc_skill_maps.py`
**What:** ACC skill maps for T2M and precipitation (W1–W4).

**Layout: 2 rows × 4 cols**
- Row 1: T2M ACC W1, W2, W3, W4
- Row 2: Precip ACC W1, W2, W3, W4

**Colormap:** `RdYlGn`, range -0.2 to 1.0  
**Stipple:** where CC < 0.3 (below 95% significance threshold for n=90)  
**Panel title:** "W{n} | CC={area_mean:.2f}" (area mean over India box 8-35°N, 68-98°E)

**Also produce:** Line plot of India-mean ACC vs lead day for T2M and Precip.

---

### Script 6: `06_mjo_phase_diagram.py` ← BAMS Fig 2 (do later)
**What:** MJO phase diagram from Spire forecasts.

**Data needed:**
- Spire TTR (OLR proxy) — band-average over equatorial strip (15°S–15°N)
- Spire U850 and U200 — from `eastward_wind_at_isobaric_levels` (check isobar values)
- Wheeler-Hendon (2004) EOF basis vectors — need to download from BoM or compute from ERA5

**Approach:**
1. Extract equatorial-mean TTR, U850, U200 anomaly as longitude vector (360 points)
2. Project onto first 2 combined EOFs → RMM1, RMM2
3. Plot (RMM1, RMM2) trajectory for each lead day (color-coded by lead time)
4. Overlay observed RMM1, RMM2 from ERA5 or NOAA CPC dataset

**Reference:** Wheeler & Hendon (2004, MWR). RMM indices available pre-computed from BoM website.

---

## Rough expected results (sanity check)

For JFM 2026, 90 init dates, India domain:

| Variable | W1 ACC | W2 ACC | W3 ACC | W4 ACC |
|----------|--------|--------|--------|--------|
| T2M      | >0.80  | 0.50-0.70 | 0.30-0.50 | 0.20-0.40 |
| Precip   | 0.50-0.70 | 0.30-0.50 | 0.10-0.30 | ~0.10 |
| Z500     | >0.85  | 0.60-0.80 | 0.40-0.60 | 0.20-0.40 |

If values are way outside these ranges, check unit conversions and sign conventions.

---

## Common pitfalls to avoid

1. **Latitude direction**: Spire may have lat descending (90 to -90). Always check before slicing.
   ```python
   lat_desc = float(ds['latitude'][0]) > float(ds['latitude'][-1])
   lat_slice = slice(lat_max, lat_min) if lat_desc else slice(lat_min, lat_max)
   ```

2. **Step coordinate**: Spire step is `timedelta64[D]`. Select as:
   ```python
   ds.sel(step=np.timedelta64(7, 'D'), method='nearest')
   ```

3. **Precip units**: Spire = kg/m²/day = mm/day. ERA5 = m/hour accumulated. Convert ERA5:
   ```python
   tp_daily_mm = ds_era5["total_precipitation"].resample(time="1D").sum("time") * 1000.0
   ```

4. **Geopotential vs Geopotential Height**: ERA5 `geopotential` is in J/kg. Divide by 9.80665 for gpm.

5. **Interpolation**: Always interp ERA5 to Spire grid, not the other way. ERA5 is truth, Spire is forecast.

6. **Memory**: Don't load all 90 init dates at once into RAM. Process in chunks or use dask.

---

## Files structure
```
spire_era5/s2s_verification/
  PLAN.md                          ← this file
  01_era5_jfm_climatology.py
  02_compute_weekly_anomalies.py
  03_skill_maps.py
  04_plot_bams_fig1.py
  05_plot_acc_skill_maps.py
  06_mjo_phase_diagram.py
  era5_jfm_climatology_1991_2020.nc   (output of Script 1)
  weekly_anomalies.nc                  (output of Script 2)
  skill_metrics.nc                     (output of Script 3)
  figures/                             (outputs of Scripts 4, 5, 6)
```
