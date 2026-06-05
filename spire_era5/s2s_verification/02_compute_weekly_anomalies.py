"""
02_compute_weekly_anomalies.py

Compute weekly-mean anomalies for all 90 Spire JFM 2026 init dates verified
against ERA5 observations.  Variables: T2M, precipitation, Z500.

Weekly windows:
  W1 = lead days  1– 7
  W2 = lead days  8–14
  W3 = lead days 15–21
  W4 = lead days 22–28

Anomaly strategy:
  Spire precip / Z500 : use `anomalies` group directly (pre-computed vs ERA5
                         1991-2020 climo by Spire).
  Spire T2M surface   : `mean_stddev/air_temperature` minus mean over 90 inits
                        (surface T2M is absent from the anomalies group).
  ERA5 obs anomaly    : ARCO weekly mean  −  WB2Climatology DOY mean.
                        WB2 climo is fetched ONCE for all 90 JFM days and
                        stored as a DOY-indexed numpy array — no per-iteration
                        remote calls.

Output: weekly_anomalies.nc
  dims : (init_time=90, week=4, latitude, longitude)
  vars : spire_t2m_anom, era5_t2m_anom,
         spire_precip_anom, era5_precip_anom,
         spire_z500_anom,  era5_z500_anom   [float32, °C / mm/day / gpm]

Coordinate notes:
  - Spire lat DESCENDING  (90→-90)  → flip on load.
  - WB2   lat DESCENDING  (90→-90)  → slice(50,0) then flip.
  - ARCO  lat DESCENDING  (90→-90)  → slice with max-first.
  - ERA5 geopotential in m²/s²  → / 9.80665 = gpm.
  - ERA5 precip in m (hourly)   → daily sum × 1000 = mm/day.
  - WB2 t2m climo in K          → − 273.15 to match ERA5 °C obs.
"""

import gc
import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client
from earth2studio.data import WB2Climatology   # type: ignore[import-untyped]

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "weekly_anomalies.nc"

LAT_MIN, LAT_MAX =  0.0, 50.0
LON_MIN, LON_MAX = 55.0, 105.0

WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28)}

# ── OPEN SPIRE ────────────────────────────────────────────────────────────────
print("Opening Spire arraylake store …")
client  = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")

ds_anom = xr.open_zarr(session.store, group="anomalies")
ds_mean = xr.open_zarr(session.store, group="mean_stddev")

def crop_spire(ds):
    """Flip descending lat to ascending, then crop to India domain."""
    return (ds.isel(latitude=slice(None, None, -1))
              .sel(latitude=slice(LAT_MIN, LAT_MAX),
                   longitude=slice(LON_MIN, LON_MAX)))

ds_anom = crop_spire(ds_anom)
ds_mean = crop_spire(ds_mean)

spire_lat  = ds_mean["latitude"].values    # ascending 0.5°, shape (101,)
spire_lon  = ds_mean["longitude"].values   # 55–105°E,  shape (101,)
n_inits    = ds_mean.sizes["reference_time"]
init_times = pd.DatetimeIndex(ds_mean["reference_time"].values)

print(f"Spire: {n_inits} inits  |  grid {len(spire_lat)} lat × {len(spire_lon)} lon")

# ── FETCH WB2 CLIMO ONCE FOR ALL 90 JFM DOYs ─────────────────────────────────
# Use 2001 (non-leap) as dummy year; WB2Climatology maps by DOY internally.
print("Fetching WB2 climatology for DOY 1-90 (single remote call) …")
wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr", verbose=False)
jfm_times = pd.date_range("2001-01-01", "2001-03-31", freq="D").to_list()
da_wb2 = wb2(jfm_times, ["t2m", "z500", "tp24"])
# da_wb2 dims: (time=90, variable=3, lat=721, lon=1440)  lat DESCENDING

# Crop to India and flip lat to ascending
da_wb2 = (da_wb2
          .sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
          .isel(lat=slice(None, None, -1)))

# Interpolate to Spire 0.5° grid once
da_wb2 = da_wb2.interp(lat=spire_lat, lon=spire_lon, method="linear")
# da_wb2 shape now: (90, 3, 101, 101)

# Build DOY-indexed numpy arrays  (index 0 = DOY 1, index 89 = DOY 90)
doy_offset = 1   # DOY 1 is index 0
wb2_t2m  = da_wb2.sel(variable="t2m").values  - 273.15  # K→°C, shape (90, lat, lon)
wb2_z500 = da_wb2.sel(variable="z500").values / 9.80665 # m²/s²→gpm
wb2_tp   = da_wb2.sel(variable="tp24").values           # mm/day (tp24 already per day)
print(f"WB2 climo cached: {da_wb2.shape}")

def wb2_window_mean(arr_doy: np.ndarray, doys: np.ndarray) -> np.ndarray:
    """Mean of WB2 climo over a set of DOYs.  arr_doy shape: (90, lat, lon)."""
    idx = doys - doy_offset          # DOY 1 → index 0
    return arr_doy[idx].mean(axis=0) # (lat, lon)

# ── OPEN ARCO-ERA5 (lazy) ─────────────────────────────────────────────────────
print("Opening ARCO-ERA5 zarr (lazy) …")
ds_era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
)

def era5_weekly_mean(var: str, dates: pd.DatetimeIndex,
                     level: int | None = None) -> np.ndarray:
    """Fetch ERA5, aggregate to daily, return weekly-mean numpy array on Spire grid.

    Args:
        var   : ARCO variable name
        dates : the 7 valid dates
        level : pressure level in hPa (geopotential only)
    Returns:
        numpy array (lat, lon) on Spire 0.5° India grid
    """
    t0 = f"{dates[0].date()}T00:00"
    t1 = f"{dates[-1].date()}T23:00"
    da = ds_era5[var].sel(
        latitude=slice(LAT_MAX + 1, LAT_MIN - 1),   # descending → max first
        longitude=slice(LON_MIN - 1, LON_MAX + 1),
        time=slice(t0, t1),
    )
    if level is not None:
        da = da.sel(level=level)

    if var == "total_precipitation":
        daily = da.resample(time="1D").sum("time").compute() * 1000.0  # m→mm/day
    else:
        daily = da.resample(time="1D").mean("time").compute()

    if var == "2m_temperature":
        daily = daily - 273.15           # K→°C
    elif var == "geopotential":
        daily = daily / 9.80665          # m²/s²→gpm

    weekly = daily.mean("time")          # (lat, lon) on ERA5 0.25° grid

    # Interp to Spire 0.5° grid
    return weekly.interp(latitude=spire_lat, longitude=spire_lon,
                         method="linear").values

# ── COMPUTE SPIRE WEEKLY MEANS FOR ALL 90 INITS (batch, before loop) ──────────
print("Pre-computing Spire weekly means for all 90 inits …")

sp_precip = {}   # wk → DataArray (reference_time, lat, lon)  — already anomaly
sp_z500   = {}   # wk → DataArray (reference_time, lat, lon)  — already anomaly
sp_t2m    = {}   # wk → DataArray (reference_time, lat, lon)  — raw K

for wk, (d0, d1) in WEEKS.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    sp_precip[wk] = ds_anom["precipitation_amount"].sel(step=steps).mean("step").compute()
    sp_z500[wk]   = (ds_anom["geopotential_height"]
                     .sel(isobar=50000.0, step=steps).mean("step").compute())
    sp_t2m[wk]    = ds_mean["air_temperature"].sel(step=steps).mean("step").compute()
    print(f"  W{wk} done")

# Spire T2M hindcast climo = mean over 90 inits
sp_t2m_climo = {wk: sp_t2m[wk].mean("reference_time").values for wk in WEEKS}

# ── ALLOCATE OUTPUT ───────────────────────────────────────────────────────────
shape = (n_inits, len(WEEKS), len(spire_lat), len(spire_lon))
out = {k: np.full(shape, np.nan, dtype=np.float32) for k in [
    "spire_t2m_anom", "era5_t2m_anom",
    "spire_precip_anom", "era5_precip_anom",
    "spire_z500_anom", "era5_z500_anom",
]}

# ── MAIN LOOP: iterate over init dates, fetch ERA5 per week ───────────────────
print(f"\nProcessing {n_inits} init dates × {len(WEEKS)} weeks …")

for i, init_date in enumerate(init_times):
    print(f"  [{i+1:02d}/{n_inits}] {init_date.date()}", end="", flush=True)

    for wk_idx, (wk, (d0, d1)) in enumerate(WEEKS.items()):
        valid_dates = pd.date_range(init_date + pd.Timedelta(d0, "D"),
                                    init_date + pd.Timedelta(d1, "D"))
        doys = valid_dates.day_of_year.values   # shape (7,)

        # ── Spire anomalies ────────────────────────────────────────────────
        out["spire_t2m_anom"][i, wk_idx] = (
            sp_t2m[wk].isel(reference_time=i).values - sp_t2m_climo[wk]
        )
        out["spire_precip_anom"][i, wk_idx] = sp_precip[wk].isel(reference_time=i).values
        out["spire_z500_anom"][i, wk_idx]   = sp_z500[wk].isel(reference_time=i).values

        # ── ERA5 obs − WB2 climo ───────────────────────────────────────────
        out["era5_t2m_anom"][i, wk_idx] = (
            era5_weekly_mean("2m_temperature", valid_dates)
            - wb2_window_mean(wb2_t2m, doys)
        )
        out["era5_precip_anom"][i, wk_idx] = (
            era5_weekly_mean("total_precipitation", valid_dates)
            - wb2_window_mean(wb2_tp, doys)
        )
        out["era5_z500_anom"][i, wk_idx] = (
            era5_weekly_mean("geopotential", valid_dates, level=500)
            - wb2_window_mean(wb2_z500, doys)
        )

        gc.collect()

    print("  ✓")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\nPacking dataset …")

week_vals = list(WEEKS.keys())

def make_da(arr, long_name, units):
    return xr.DataArray(
        arr,
        dims=["init_time", "week", "latitude", "longitude"],
        coords={"init_time": init_times, "week": week_vals,
                "latitude": spire_lat,   "longitude": spire_lon},
        attrs={"long_name": long_name, "units": units},
    )

ds_out = xr.Dataset({
    "spire_t2m_anom":    make_da(out["spire_t2m_anom"],    "Spire T2M anomaly",    "°C"),
    "era5_t2m_anom":     make_da(out["era5_t2m_anom"],     "ERA5 T2M anomaly",     "°C"),
    "spire_precip_anom": make_da(out["spire_precip_anom"], "Spire precip anomaly", "mm/day"),
    "era5_precip_anom":  make_da(out["era5_precip_anom"],  "ERA5 precip anomaly",  "mm/day"),
    "spire_z500_anom":   make_da(out["spire_z500_anom"],   "Spire Z500 anomaly",   "gpm"),
    "era5_z500_anom":    make_da(out["era5_z500_anom"],    "ERA5 Z500 anomaly",    "gpm"),
}, attrs={
    "description":  "Weekly-mean anomalies, Spire JFM 2026 vs ERA5, India domain",
    "spire_climo":  "precip/Z500: Spire anomalies group (vs ERA5 1991-2020). "
                    "T2M: mean over 90 JFM 2026 inits.",
    "era5_climo":   "WeatherBench2 1990-2019 DOY climatology",
    "domain":       f"lat {LAT_MIN}–{LAT_MAX}°N  lon {LON_MIN}–{LON_MAX}°E  0.5°",
})

ds_out.to_netcdf(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}")
print(ds_out)
