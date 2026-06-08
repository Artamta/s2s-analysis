"""
02_compute_weekly_anomalies.py

Compute weekly-mean anomalies for all 90 Spire JFM 2026 init dates verified
against ERA5 observations.  Variables: T2M, precipitation, Z500.

Weekly windows (Spire provides 46 lead days → six full 7-day weeks):
  W1 = lead days  1– 7
  W2 = lead days  8–14
  W3 = lead days 15–21
  W4 = lead days 22–28
  W5 = lead days 29–35
  W6 = lead days 36–42
  (days 43–46 do not form a complete 7th week and are dropped)

Anomaly strategy:
  Spire precip / Z500 : use `anomalies` group directly (pre-computed vs ERA5
                         1991-2020 climo by Spire).
  Spire T2M surface   : `mean_stddev/air_temperature` minus mean over 90 inits
                        (surface T2M is absent from the anomalies group).
  ERA5 obs anomaly    : ARCO weekly mean  −  WB2Climatology DOY mean.

PERFORMANCE — why this version is fast:
  The 90 JFM init dates × 6 weekly windows have heavily OVERLAPPING valid dates,
  all falling inside one contiguous window: 2026-01-02 → 2026-05-12 (131 days).
  Instead of 90×6×3 = 1620 cloud fetches (the slow naive approach), we download
  each ERA5 variable ONCE for that whole window, aggregate to daily, interpolate
  to the Spire grid once, and cache as a date-indexed numpy array.  Every weekly
  mean is then a pure in-memory numpy slice.  →  3 cloud fetches total.

  Likewise WB2 climatology is fetched ONCE for DOY 1-118 and cached.

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

import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client
from earth2studio.data import WB2Climatology   # type: ignore[import-untyped]

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "weekly_anomalies.nc"

LAT_MIN, LAT_MAX =  0.0, 50.0
LON_MIN, LON_MAX = 55.0, 105.0

WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21),
         4: (22, 28), 5: (29, 35), 6: (36, 42)}
MAX_LEAD = 42   # longest lead day used (= end of W6); Spire offers 46


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

spire_lat  = ds_mean["latitude"].values    # ascending 0.5°
spire_lon  = ds_mean["longitude"].values   # 55–105°E
n_inits    = ds_mean.sizes["reference_time"]
init_times = pd.DatetimeIndex(ds_mean["reference_time"].values)

print(f"Spire: {n_inits} inits  |  grid {len(spire_lat)} lat × {len(spire_lon)} lon")

# ── DEFINE THE SINGLE CONTIGUOUS VALID-DATE WINDOW ────────────────────────────
# Earliest valid date = first init + lead 1.  Latest = last init + lead 28.
window_start = init_times[0]  + pd.Timedelta(1, "D")
window_end   = init_times[-1] + pd.Timedelta(MAX_LEAD, "D")
all_dates    = pd.date_range(window_start, window_end, freq="D")
print(f"ERA5 valid-date window: {window_start.date()} → {window_end.date()} "
      f"({len(all_dates)} days)")

# Map each calendar date → its index in the cached daily arrays.
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# ── FETCH WB2 CLIMO ONCE FOR ALL DOYs IN THE WINDOW ───────────────────────────
# Window spans DOY 2 → 118 (covers into April).  Fetch the union of DOYs needed.
# Use 2001 (non-leap) so DOY == day-of-year cleanly; WB2 maps internally by DOY.
print("Fetching WB2 climatology for the valid-date window (single remote call) …")
wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr", verbose=False)

# Representative datetime per DOY in our window (dummy year 2001).
doys_needed = sorted(set(all_dates.day_of_year.values.tolist()))
wb2_times   = [pd.Timestamp("2001-01-01") + pd.Timedelta(int(doy) - 1, "D")
               for doy in doys_needed]
da_wb2 = wb2(wb2_times, ["t2m", "z500", "tp24"])
# dims: (time, variable, lat=721, lon=1440)  lat DESCENDING

# Crop to India, flip lat ascending, interpolate to Spire grid — all once.
da_wb2 = (da_wb2
          .sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
          .isel(lat=slice(None, None, -1))
          .interp(lat=spire_lat, lon=spire_lon, method="linear"))

# Cache as DOY-keyed dict of (lat, lon) arrays, with unit conversions.
wb2_t2m_by_doy  = {}
wb2_z500_by_doy = {}
wb2_tp_by_doy   = {}
for k, doy in enumerate(doys_needed):
    wb2_t2m_by_doy[doy]  = da_wb2.isel(time=k).sel(variable="t2m").values  - 273.15
    wb2_z500_by_doy[doy] = da_wb2.isel(time=k).sel(variable="z500").values / 9.80665
    wb2_tp_by_doy[doy]   = da_wb2.isel(time=k).sel(variable="tp24").values
print(f"WB2 climo cached for {len(doys_needed)} DOYs.")

# ── FETCH ERA5 ONCE PER VARIABLE FOR THE WHOLE WINDOW ─────────────────────────
print("Opening ARCO-ERA5 zarr …")
ds_era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
)

def fetch_era5_daily(var: str, level: int | None = None) -> np.ndarray:
    """
    Download one ERA5 variable for the FULL valid-date window, aggregate to
    daily, convert units, interpolate to the Spire grid.

    Returns a numpy array of shape (n_days, lat, lon) aligned with `all_dates`.
    This is the only place we touch the network for ERA5 — called 3 times total.
    """
    t0 = f"{window_start.date()}T00:00"
    t1 = f"{window_end.date()}T23:00"
    da = ds_era5[var].sel(
        latitude=slice(LAT_MAX + 1, LAT_MIN - 1),    # descending → max first
        longitude=slice(LON_MIN - 1, LON_MAX + 1),
        time=slice(t0, t1),
    )
    if level is not None:
        da = da.sel(level=level)

    print(f"  fetching ERA5 '{var}'"
          + (f" @ {level}hPa" if level else "")
          + f"  ({da.sizes['time']} hourly steps) …", flush=True)

    if var == "total_precipitation":
        daily = da.resample(time="1D").sum("time").compute() * 1000.0  # m→mm/day
    else:
        daily = da.resample(time="1D").mean("time").compute()

    if var == "2m_temperature":
        daily = daily - 273.15           # K→°C
    elif var == "geopotential":
        daily = daily / 9.80665          # m²/s²→gpm

    # Interpolate every daily field to the Spire grid in one vectorised call.
    daily = daily.interp(latitude=spire_lat, longitude=spire_lon, method="linear")

    # Reindex onto our master date axis so index i == all_dates[i].
    daily = daily.reindex(time=all_dates)
    return daily.values.astype(np.float32)   # (n_days, lat, lon)

print("Pre-fetching ERA5 daily fields for the whole window …")
era5_t2m_daily  = fetch_era5_daily("2m_temperature")
era5_tp_daily   = fetch_era5_daily("total_precipitation")
era5_z500_daily = fetch_era5_daily("geopotential", level=500)
print("ERA5 daily fields cached in memory.")

# ── PRE-COMPUTE SPIRE WEEKLY MEANS FOR ALL 90 INITS (batch, before loop) ──────
print("Pre-computing Spire weekly means for all 90 inits …")

sp_precip = {}   # wk → DataArray (reference_time, lat, lon)  — already anomaly
sp_z500   = {}   # wk → DataArray (reference_time, lat, lon)  — already anomaly
sp_t2m    = {}   # wk → DataArray (reference_time, lat, lon)  — raw °C-offset (K)

for wk, (d0, d1) in WEEKS.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    sp_precip[wk] = ds_anom["precipitation_amount"].sel(step=steps).mean("step").compute()
    sp_z500[wk]   = (ds_anom["geopotential_height"]
                     .sel(isobar=50000.0, step=steps).mean("step").compute())
    sp_t2m[wk]    = ds_mean["air_temperature"].sel(step=steps).mean("step").compute()
    print(f"  W{wk} done")

# Spire T2M hindcast climo = mean over the 90 inits, per week.
sp_t2m_climo = {wk: sp_t2m[wk].mean("reference_time").values for wk in WEEKS}

# ── HELPERS FOR THE NUMPY-ONLY MAIN LOOP ──────────────────────────────────────
def era5_window_mean(daily_arr: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean of a cached ERA5 daily array over a set of dates. → (lat, lon)."""
    idx = [date_to_idx[d] for d in dates]
    return daily_arr[idx].mean(axis=0)

def wb2_window_mean(by_doy: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean of cached WB2 climo over the DOYs of `dates`. → (lat, lon)."""
    return np.mean([by_doy[int(d.day_of_year)] for d in dates], axis=0)

# ── ALLOCATE OUTPUT ───────────────────────────────────────────────────────────
shape = (n_inits, len(WEEKS), len(spire_lat), len(spire_lon))
out = {k: np.full(shape, np.nan, dtype=np.float32) for k in [
    "spire_t2m_anom", "era5_t2m_anom",
    "spire_precip_anom", "era5_precip_anom",
    "spire_z500_anom", "era5_z500_anom",
]}

# ── MAIN LOOP: pure in-memory numpy, no network ───────────────────────────────
print(f"\nProcessing {n_inits} init dates × {len(WEEKS)} weeks (in-memory) …")

for i, init_date in enumerate(init_times):
    for wk_idx, (wk, (d0, d1)) in enumerate(WEEKS.items()):
        valid_dates = pd.date_range(init_date + pd.Timedelta(d0, "D"),
                                    init_date + pd.Timedelta(d1, "D"))

        # ── Spire anomalies ────────────────────────────────────────────────
        out["spire_t2m_anom"][i, wk_idx] = (
            sp_t2m[wk].isel(reference_time=i).values - sp_t2m_climo[wk]
        )
        out["spire_precip_anom"][i, wk_idx] = sp_precip[wk].isel(reference_time=i).values
        out["spire_z500_anom"][i, wk_idx]   = sp_z500[wk].isel(reference_time=i).values

        # ── ERA5 obs − WB2 climo ───────────────────────────────────────────
        out["era5_t2m_anom"][i, wk_idx] = (
            era5_window_mean(era5_t2m_daily,  valid_dates)
            - wb2_window_mean(wb2_t2m_by_doy,  valid_dates)
        )
        out["era5_precip_anom"][i, wk_idx] = (
            era5_window_mean(era5_tp_daily,    valid_dates)
            - wb2_window_mean(wb2_tp_by_doy,   valid_dates)
        )
        out["era5_z500_anom"][i, wk_idx] = (
            era5_window_mean(era5_z500_daily,  valid_dates)
            - wb2_window_mean(wb2_z500_by_doy, valid_dates)
        )

    if (i + 1) % 10 == 0 or i == n_inits - 1:
        print(f"  [{i+1:02d}/{n_inits}] {init_date.date()} done")

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
