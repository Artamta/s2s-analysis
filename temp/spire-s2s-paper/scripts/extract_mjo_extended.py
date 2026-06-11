"""
extract_mjo_extended.py  (v3 — Spire-only, no ERA5 fetch)
===========================================================
Extracts OLR (ttr), U850, U200 from Spire ArrayLake ONLY.
ERA5 reference for these MJO variables is NOT fetched here
(too slow from GCS for pressure-level + surface radiation data).

For ACC/comparison, we use init-mean anomaly — each init's deviation
from the ensemble mean. This allows within-model skill assessment.

Produces:
  data/weekly_anomalies_extended.nc  — India domain
  data/equatorial_hovmoller.nc       — Equatorial band daily data
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_INDIA = os.path.join(OUT_DIR, "weekly_anomalies_extended.nc")
OUT_HOV   = os.path.join(OUT_DIR, "equatorial_hovmoller.nc")

LAT_MIN, LAT_MAX = 0.0, 50.0
LON_MIN, LON_MAX = 55.0, 105.0
EQ_LAT_MIN, EQ_LAT_MAX = -15.0, 15.0

WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21),
         4: (22, 28), 5: (29, 35), 6: (36, 42)}
os.makedirs(OUT_DIR, exist_ok=True)

# ── OPEN SPIRE ────────────────────────────────────────────────────────────────
print("Opening Spire arraylake …")
client  = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")
ds_mean = xr.open_zarr(session.store, group="mean_stddev")

n_inits    = ds_mean.sizes["reference_time"]
init_times = pd.DatetimeIndex(ds_mean["reference_time"].values)
print(f"Spire: {n_inits} inits, steps 1-46, grid {ds_mean.sizes['latitude']}×{ds_mean.sizes['longitude']}")

def crop_india(ds):
    return (ds.isel(latitude=slice(None, None, -1))
              .sel(latitude=slice(LAT_MIN, LAT_MAX),
                   longitude=slice(LON_MIN, LON_MAX)))

def crop_eq(ds):
    return (ds.isel(latitude=slice(None, None, -1))
              .sel(latitude=slice(EQ_LAT_MIN, EQ_LAT_MAX)))

india_lat = crop_india(ds_mean)["latitude"].values
india_lon = crop_india(ds_mean)["longitude"].values

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Spire India-domain weekly means
# ══════════════════════════════════════════════════════════════════════════════
print("\n═ STEP 1: Spire weekly means (India) ═")
ds_india = crop_india(ds_mean)

sp_olr, sp_u850, sp_u200 = {}, {}, {}
for wk, (d0, d1) in WEEKS.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    sp_olr[wk]  = -ds_india["ttr"].sel(step=steps).mean("step").compute() / 86400.0
    sp_u850[wk] = ds_india["eastward_wind_at_isobaric_levels"].sel(
        isobar=85000.0, step=steps).mean("step").compute()
    sp_u200[wk] = ds_india["eastward_wind_at_isobaric_levels"].sel(
        isobar=20000.0, step=steps).mean("step").compute()
    print(f"  W{wk}: OLR [{float(sp_olr[wk].min()):.0f},{float(sp_olr[wk].max()):.0f}] "
          f"U850 [{float(sp_u850[wk].min()):.1f},{float(sp_u850[wk].max()):.1f}] "
          f"U200 [{float(sp_u200[wk].min()):.1f},{float(sp_u200[wk].max()):.1f}]")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute anomalies (init-mean baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═ STEP 2: Computing anomalies ═")

shape = (n_inits, len(WEEKS), len(india_lat), len(india_lon))

olr_all  = np.stack([sp_olr[wk].values  for wk in WEEKS], axis=1)  # (90,6,lat,lon)
u850_all = np.stack([sp_u850[wk].values for wk in WEEKS], axis=1)
u200_all = np.stack([sp_u200[wk].values for wk in WEEKS], axis=1)

# Anomaly = deviation from init-averaged field per week
olr_wkmean  = olr_all.mean(axis=0)
u850_wkmean = u850_all.mean(axis=0)
u200_wkmean = u200_all.mean(axis=0)

spire_olr_anom  = olr_all  - olr_wkmean[np.newaxis, :]
spire_u850_anom = u850_all - u850_wkmean[np.newaxis, :]
spire_u200_anom = u200_all - u200_wkmean[np.newaxis, :]

print(f"  OLR anom range: [{spire_olr_anom.min():.1f}, {spire_olr_anom.max():.1f}] W/m²")
print(f"  U850 anom range: [{spire_u850_anom.min():.2f}, {spire_u850_anom.max():.2f}] m/s")
print(f"  U200 anom range: [{spire_u200_anom.min():.2f}, {spire_u200_anom.max():.2f}] m/s")

# Save
print("\nSaving India-domain extended data …")
week_vals = list(WEEKS.keys())
def mk(arr, ln, u):
    return xr.DataArray(arr, dims=["init_time", "week", "latitude", "longitude"],
        coords={"init_time": init_times, "week": week_vals,
                "latitude": india_lat, "longitude": india_lon},
        attrs={"long_name": ln, "units": u})

ds_ext = xr.Dataset({
    "spire_olr_anom":   mk(spire_olr_anom,  "Spire OLR anomaly",  "W m-2"),
    "spire_u850_anom":  mk(spire_u850_anom, "Spire U850 anomaly", "m s-1"),
    "spire_u200_anom":  mk(spire_u200_anom, "Spire U200 anomaly", "m s-1"),
    "spire_olr_raw":    mk(olr_all,         "Spire OLR (TTR)",    "W m-2"),
    "spire_u850_raw":   mk(u850_all,        "Spire U850",         "m s-1"),
    "spire_u200_raw":   mk(u200_all,        "Spire U200",         "m s-1"),
    "olr_wk_climatology":  xr.DataArray(olr_wkmean,  dims=["week","latitude","longitude"],
                            coords={"week": week_vals, "latitude": india_lat, "longitude": india_lon}),
    "u850_wk_climatology": xr.DataArray(u850_wkmean, dims=["week","latitude","longitude"],
                            coords={"week": week_vals, "latitude": india_lat, "longitude": india_lon}),
    "u200_wk_climatology": xr.DataArray(u200_wkmean, dims=["week","latitude","longitude"],
                            coords={"week": week_vals, "latitude": india_lat, "longitude": india_lon}),
}, attrs={
    "description": "Extended Spire MJO variables: OLR/U850/U200 anomalies + raw + climatology",
    "anomaly_baseline": "Init-mean: anomaly = value - mean(all 90 inits for that week)",
})
ds_ext.to_netcdf(OUT_INDIA)
print(f"Saved → {OUT_INDIA} ({os.path.getsize(OUT_INDIA)/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Equatorial Hovmöller (daily, 15S-15N, all longitudes)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═ STEP 3: Equatorial Hovmöller ═")
ds_eq = crop_eq(ds_mean)
eq_lat = ds_eq["latitude"].values
eq_lon = ds_eq["longitude"].values
n_steps = 42

sp_olr_eq  = np.full((n_inits, n_steps, len(eq_lon)), np.nan, np.float32)
sp_u850_eq = np.full((n_inits, n_steps, len(eq_lon)), np.nan, np.float32)
sp_u200_eq = np.full((n_inits, n_steps, len(eq_lon)), np.nan, np.float32)

for s in range(n_steps):
    step_val = np.timedelta64(s + 1, "D")
    olr_s  = -ds_eq["ttr"].sel(step=step_val).compute().mean("latitude") / 86400.0
    u850_s = ds_eq["eastward_wind_at_isobaric_levels"].sel(
        isobar=85000.0, step=step_val).compute().mean("latitude")
    u200_s = ds_eq["eastward_wind_at_isobaric_levels"].sel(
        isobar=20000.0, step=step_val).compute().mean("latitude")
    
    sp_olr_eq[:, s, :]  = olr_s.values
    sp_u850_eq[:, s, :] = u850_s.values
    sp_u200_eq[:, s, :] = u200_s.values
    
    if (s + 1) % 7 == 0:
        print(f"  Step {s+1}/42 done")

step_coords = np.arange(1, n_steps + 1)
def mk_eq(arr, ln, u):
    return xr.DataArray(arr, dims=["init_time", "step", "longitude"],
        coords={"init_time": init_times, "step": step_coords, "longitude": eq_lon},
        attrs={"long_name": ln, "units": u})

ds_hov = xr.Dataset({
    "spire_olr_eq":  mk_eq(sp_olr_eq,  "Spire OLR eq-mean (15S-15N)", "W m-2"),
    "spire_u850_eq": mk_eq(sp_u850_eq, "Spire U850 eq-mean (15S-15N)", "m s-1"),
    "spire_u200_eq": mk_eq(sp_u200_eq, "Spire U200 eq-mean (15S-15N)", "m s-1"),
}, attrs={
    "description": "Equatorial Hovmöller data — Spire AI-S2S JFM 2026",
    "lat_band": f"{EQ_LAT_MIN} to {EQ_LAT_MAX}",
})
ds_hov.to_netcdf(OUT_HOV)
print(f"Saved → {OUT_HOV} ({os.path.getsize(OUT_HOV)/1e6:.1f} MB)")

print("\n" + "="*65)
print("DONE ✓")
print("="*65)
