"""
09_compute_anomalies_v2.py

CORRECTED anomaly computation — both Spire and ERA5 use the SAME climatology
baseline so the comparison is fully apples-to-apples.

Fixes the bug in 02_compute_weekly_anomalies.py where Spire T2M anomaly was
computed as (raw - mean_over_90_inits), which forces the 90-init mean anomaly
to be identically ZERO. Here we subtract a proper external climatology instead.

Two physically-matched variants are produced:

  VARIANT "mean"  (mean-vs-mean):
    Spire air_temperature (daily mean)  −  WB2 1990-2019 mean-T2m DOY climo
    ERA5  daily-mean T2m                −  WB2 1990-2019 mean-T2m DOY climo

  VARIANT "max"  (max-vs-max):
    Spire air_temperature_max (daily max) −  ERA5 Tmax DOY climo (computed here)
    ERA5  daily-max  T2m                  −  ERA5 Tmax DOY climo (computed here)

  ERA5 Tmax climatology: daily max of hourly T2m, averaged by day-of-year over a
  reference period (default 1991-2020). This is the only heavy compute step; it
  is cached to disk (era5_tmax_climo_india.nc) and reused.

Precip & Z500 anomalies are taken from Spire's `anomalies` group (already vs
ERA5 1991-2020) and ERA5 vs WB2 — unchanged, they were already correct.

Output: weekly_anomalies_v2.nc
  dims : (init_time=90, week=6, latitude, longitude)
  vars : spire_t2m_mean_anom, era5_t2m_mean_anom,
         spire_t2m_max_anom,  era5_t2m_max_anom,
         spire_precip_anom,   era5_precip_anom,
         spire_z500_anom,     era5_z500_anom
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client
from earth2studio.data import WB2Climatology

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = "weekly_anomalies_v2.nc"
TMAX_CLIMO_NC = "era5_tmax_climo_india.nc"
CLIMO_YEARS   = range(1991, 2021)        # 1991-2020 (30 yr) for ERA5 Tmax climo

LAT_MIN, LAT_MAX =  0.0, 50.0
LON_MIN, LON_MAX = 55.0, 105.0

WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21),
         4: (22, 28), 5: (29, 35), 6: (36, 42)}
MAX_LEAD = 42

# ── OPEN SPIRE ────────────────────────────────────────────────────────────────
print("Opening Spire arraylake store …")
client  = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")
ds_anom = xr.open_zarr(session.store, group="anomalies")
ds_mean = xr.open_zarr(session.store, group="mean_stddev")

def crop_spire(ds):
    return (ds.isel(latitude=slice(None, None, -1))
              .sel(latitude=slice(LAT_MIN, LAT_MAX),
                   longitude=slice(LON_MIN, LON_MAX)))

ds_anom = crop_spire(ds_anom)
ds_mean = crop_spire(ds_mean)

spire_lat  = ds_mean["latitude"].values
spire_lon  = ds_mean["longitude"].values
n_inits    = ds_mean.sizes["reference_time"]
init_times = pd.DatetimeIndex(ds_mean["reference_time"].values)
print(f"Spire: {n_inits} inits  |  grid {len(spire_lat)} × {len(spire_lon)}")

# ── VALID-DATE WINDOW ─────────────────────────────────────────────────────────
window_start = init_times[0]  + pd.Timedelta(1, "D")
window_end   = init_times[-1] + pd.Timedelta(MAX_LEAD, "D")
all_dates    = pd.date_range(window_start, window_end, freq="D")
date_to_idx  = {d: i for i, d in enumerate(all_dates)}
print(f"ERA5 valid-date window: {window_start.date()} → {window_end.date()} ({len(all_dates)} days)")

# ── ARCO-ERA5 ─────────────────────────────────────────────────────────────────
print("Opening ARCO-ERA5 …")
ds_era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ERA5 Tmax climatology (daily max of hourly T2m, DOY mean over 1991-2020)
#          Cached to disk. This is the expensive part.
# ══════════════════════════════════════════════════════════════════════════════
doys_needed = sorted(set(all_dates.day_of_year.values.tolist()))

if os.path.exists(TMAX_CLIMO_NC):
    print(f"Loading cached ERA5 Tmax climo ← {TMAX_CLIMO_NC}")
    tmax_climo_ds = xr.open_dataset(TMAX_CLIMO_NC)
    era5_tmax_climo_by_doy = {
        int(d): tmax_climo_ds["tmax_climo"].sel(doy=int(d)).values
        for d in doys_needed
    }
else:
    print(f"Computing ERA5 Tmax climatology for {len(doys_needed)} DOYs "
          f"over {CLIMO_YEARS.start}-{CLIMO_YEARS.stop-1} (heavy, one-time) …")
    da_t2m_full = ds_era5["2m_temperature"].sel(
        latitude=slice(LAT_MAX + 1, LAT_MIN - 1),
        longitude=slice(LON_MIN - 1, LON_MAX + 1),
    )
    # EFFICIENT: fetch each climo year's full DOY-window ONCE, daily-max via
    # resample, then accumulate per-DOY. Far fewer, larger reads than DOY×year.
    doy_min, doy_max = min(doys_needed), max(doys_needed)
    accum = {int(d): np.zeros((len(spire_lat), len(spire_lon)), np.float64)
             for d in doys_needed}
    count = {int(d): 0 for d in doys_needed}
    for yi, yr in enumerate(CLIMO_YEARS):
        d0 = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(doy_min - 1, "D")
        d1 = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(doy_max - 1, "D")
        hourly = da_t2m_full.sel(time=slice(f"{d0.date()}T00:00",
                                            f"{d1.date()}T23:00"))
        daily_max = (hourly.resample(time="1D").max("time").compute() - 273.15)
        daily_max = daily_max.interp(latitude=spire_lat, longitude=spire_lon,
                                     method="linear")
        dm_doy = pd.DatetimeIndex(daily_max["time"].values).day_of_year
        for k, doy in enumerate(dm_doy):
            doy = int(doy)
            if doy in accum:
                accum[doy] += daily_max.isel(time=k).values
                count[doy] += 1
        print(f"  year {yr} done ({yi+1}/{len(list(CLIMO_YEARS))})", flush=True)
    era5_tmax_climo_by_doy = {
        d: (accum[d] / max(count[d], 1)).astype(np.float32) for d in accum
    }

    # Save cache
    doy_arr = np.array(sorted(era5_tmax_climo_by_doy.keys()))
    stack   = np.stack([era5_tmax_climo_by_doy[int(d)] for d in doy_arr])
    xr.Dataset(
        {"tmax_climo": (["doy", "latitude", "longitude"], stack)},
        coords={"doy": doy_arr, "latitude": spire_lat, "longitude": spire_lon},
        attrs={"description": f"ERA5 daily-Tmax DOY climatology "
                              f"{CLIMO_YEARS.start}-{CLIMO_YEARS.stop-1}, India 0.5°"},
    ).to_netcdf(TMAX_CLIMO_NC)
    print(f"  cached → {TMAX_CLIMO_NC}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WB2 mean-T2m / Z500 / precip climatology (DOY), cached in memory
# ══════════════════════════════════════════════════════════════════════════════
print("Fetching WB2 climatology (mean T2m, Z500, precip) …")
wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr", verbose=False)
wb2_times = [pd.Timestamp("2001-01-01") + pd.Timedelta(int(doy) - 1, "D")
             for doy in doys_needed]
da_wb2 = wb2(wb2_times, ["t2m", "z500", "tp24"])
da_wb2 = (da_wb2
          .sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
          .isel(lat=slice(None, None, -1))
          .interp(lat=spire_lat, lon=spire_lon, method="linear"))

wb2_t2m_by_doy, wb2_z500_by_doy, wb2_tp_by_doy = {}, {}, {}
for k, doy in enumerate(doys_needed):
    wb2_t2m_by_doy[doy]  = da_wb2.isel(time=k).sel(variable="t2m").values  - 273.15
    wb2_z500_by_doy[doy] = da_wb2.isel(time=k).sel(variable="z500").values / 9.80665
    wb2_tp_by_doy[doy]   = da_wb2.isel(time=k).sel(variable="tp24").values
print(f"WB2 climo cached for {len(doys_needed)} DOYs.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ERA5 daily fields (mean T2m, max T2m, precip, Z500) for the window
# ══════════════════════════════════════════════════════════════════════════════
def fetch_era5_daily(var, level=None, agg="mean"):
    t0 = f"{window_start.date()}T00:00"; t1 = f"{window_end.date()}T23:00"
    da = ds_era5[var].sel(
        latitude=slice(LAT_MAX + 1, LAT_MIN - 1),
        longitude=slice(LON_MIN - 1, LON_MAX + 1),
        time=slice(t0, t1))
    if level is not None:
        da = da.sel(level=level)
    print(f"  fetching ERA5 '{var}'" + (f"@{level}" if level else "") +
          f" agg={agg} ({da.sizes['time']} steps) …", flush=True)
    if agg == "sum":
        daily = da.resample(time="1D").sum("time").compute() * 1000.0
    elif agg == "max":
        daily = da.resample(time="1D").max("time").compute()
    else:
        daily = da.resample(time="1D").mean("time").compute()
    if var == "2m_temperature":
        daily = daily - 273.15
    elif var == "geopotential":
        daily = daily / 9.80665
    daily = daily.interp(latitude=spire_lat, longitude=spire_lon, method="linear")
    daily = daily.reindex(time=all_dates)
    return daily.values.astype(np.float32)

print("Pre-fetching ERA5 daily fields …")
era5_t2m_mean_daily = fetch_era5_daily("2m_temperature", agg="mean")
era5_t2m_max_daily  = fetch_era5_daily("2m_temperature", agg="max")
era5_tp_daily       = fetch_era5_daily("total_precipitation", agg="sum")
era5_z500_daily     = fetch_era5_daily("geopotential", level=500, agg="mean")
print("ERA5 daily fields cached.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Spire weekly means (raw mean & raw max from mean_stddev; anomaly grp for pr/z5)
# ══════════════════════════════════════════════════════════════════════════════
print("Pre-computing Spire weekly means …")
sp_t2m_mean, sp_t2m_max, sp_precip, sp_z500 = {}, {}, {}, {}
for wk, (d0, d1) in WEEKS.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    sp_t2m_mean[wk] = ds_mean["air_temperature"].sel(step=steps).mean("step").compute() - 273.15
    sp_t2m_max[wk]  = ds_mean["air_temperature_max"].sel(step=steps).mean("step").compute() - 273.15
    sp_precip[wk]   = ds_anom["precipitation_amount"].sel(step=steps).mean("step").compute()
    sp_z500[wk]     = ds_anom["geopotential_height"].sel(isobar=50000.0, step=steps).mean("step").compute()
    print(f"  W{wk} done")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def era5_win(daily, dates):
    return daily[[date_to_idx[d] for d in dates]].mean(axis=0)
def climo_win(by_doy, dates):
    return np.mean([by_doy[int(d.day_of_year)] for d in dates], axis=0)

# ── ALLOCATE ──────────────────────────────────────────────────────────────────
shape = (n_inits, len(WEEKS), len(spire_lat), len(spire_lon))
keys = ["spire_t2m_mean_anom", "era5_t2m_mean_anom",
        "spire_t2m_max_anom",  "era5_t2m_max_anom",
        "spire_precip_anom",   "era5_precip_anom",
        "spire_z500_anom",     "era5_z500_anom"]
out = {k: np.full(shape, np.nan, dtype=np.float32) for k in keys}

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print(f"\nProcessing {n_inits} inits × {len(WEEKS)} weeks …")
for i, init_date in enumerate(init_times):
    for wk_idx, (wk, (d0, d1)) in enumerate(WEEKS.items()):
        vdates = pd.date_range(init_date + pd.Timedelta(d0, "D"),
                               init_date + pd.Timedelta(d1, "D"))

        # MEAN variant — Spire mean vs WB2 mean climo; ERA5 mean vs WB2 mean climo
        wb2_mean = climo_win(wb2_t2m_by_doy, vdates)
        out["spire_t2m_mean_anom"][i, wk_idx] = sp_t2m_mean[wk].isel(reference_time=i).values - wb2_mean
        out["era5_t2m_mean_anom"][i, wk_idx]  = era5_win(era5_t2m_mean_daily, vdates) - wb2_mean

        # MAX variant — Spire max vs ERA5 Tmax climo; ERA5 max vs ERA5 Tmax climo
        tmax_clim = climo_win(era5_tmax_climo_by_doy, vdates)
        out["spire_t2m_max_anom"][i, wk_idx] = sp_t2m_max[wk].isel(reference_time=i).values - tmax_clim
        out["era5_t2m_max_anom"][i, wk_idx]  = era5_win(era5_t2m_max_daily, vdates) - tmax_clim

        # Precip & Z500 (unchanged)
        out["spire_precip_anom"][i, wk_idx] = sp_precip[wk].isel(reference_time=i).values
        out["spire_z500_anom"][i, wk_idx]   = sp_z500[wk].isel(reference_time=i).values
        out["era5_precip_anom"][i, wk_idx]  = era5_win(era5_tp_daily, vdates) - climo_win(wb2_tp_by_doy, vdates)
        out["era5_z500_anom"][i, wk_idx]    = era5_win(era5_z500_daily, vdates) - climo_win(wb2_z500_by_doy, vdates)

    if (i + 1) % 10 == 0 or i == n_inits - 1:
        print(f"  [{i+1:02d}/{n_inits}] {init_date.date()} done")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\nPacking dataset …")
week_vals = list(WEEKS.keys())
def mk(arr, ln, u):
    return xr.DataArray(arr, dims=["init_time", "week", "latitude", "longitude"],
        coords={"init_time": init_times, "week": week_vals,
                "latitude": spire_lat, "longitude": spire_lon},
        attrs={"long_name": ln, "units": u})

ds_out = xr.Dataset({
    "spire_t2m_mean_anom": mk(out["spire_t2m_mean_anom"], "Spire mean-T2m anomaly (vs WB2)", "K"),
    "era5_t2m_mean_anom":  mk(out["era5_t2m_mean_anom"],  "ERA5 mean-T2m anomaly (vs WB2)",  "K"),
    "spire_t2m_max_anom":  mk(out["spire_t2m_max_anom"],  "Spire max-T2m anomaly (vs ERA5 Tmax climo)", "K"),
    "era5_t2m_max_anom":   mk(out["era5_t2m_max_anom"],   "ERA5 max-T2m anomaly (vs ERA5 Tmax climo)",  "K"),
    "spire_precip_anom":   mk(out["spire_precip_anom"],   "Spire precip anomaly", "mm/day"),
    "era5_precip_anom":    mk(out["era5_precip_anom"],    "ERA5 precip anomaly",  "mm/day"),
    "spire_z500_anom":     mk(out["spire_z500_anom"],     "Spire Z500 anomaly",   "gpm"),
    "era5_z500_anom":      mk(out["era5_z500_anom"],      "ERA5 Z500 anomaly",    "gpm"),
}, attrs={
    "description": "Weekly-mean anomalies v2 — consistent baselines, Spire JFM 2026 vs ERA5",
    "mean_variant": "Spire air_temperature & ERA5 mean-T2m both vs WB2 1990-2019 mean climo",
    "max_variant":  f"Spire air_temperature_max & ERA5 max-T2m both vs ERA5 {CLIMO_YEARS.start}-{CLIMO_YEARS.stop-1} Tmax DOY climo",
    "domain": f"lat {LAT_MIN}-{LAT_MAX}N lon {LON_MIN}-{LON_MAX}E 0.5deg",
})
ds_out.to_netcdf(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}")

# Quick sanity print
for v in ["spire_t2m_mean_anom", "era5_t2m_mean_anom", "spire_t2m_max_anom", "era5_t2m_max_anom"]:
    m = ds_out[v].mean("init_time")
    print(f"  {v}: W1 mean={float(m.sel(week=1).mean()):+.2f}  W6 mean={float(m.sel(week=6).mean()):+.2f}")
