"""
Download ERA5 Dec 25-30 2025 from ARCO-ERA5 (Google Cloud, anonymous).

WHY:  ERA5 GRIB files start 2025-12-31, but persistence for init=2026-01-01
      requires the observed week Dec 25-31 2025. This script fills that gap.

WHAT: Downloads TP (mm/day), Z500 (gpm), T2M (K) for Dec 25-30 2025
      over the India domain (lat 3-40N, lon 63-102E).

OUTPUT: /storage/raj.ayush/s2s-forecast-data/era5/data/era5_dec2025_persistence.nc
        Variables: tp (mm/day), z500 (gpm), t2m (K)
        Dims: time (6 days), latitude, longitude
"""
import warnings, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xarray as xr

# ─── Config ───────────────────────────────────────────────────────────────────
OUT    = '/storage/raj.ayush/s2s-forecast-data/era5/data/era5_dec2025_persistence.nc'
G      = 9.80665
DATES  = pd.date_range('2025-12-25', '2025-12-31', freq='D')
LAT    = slice(40, 3)     # India domain (ARCO uses descending lat)
LON    = slice(63, 102)

# ─── Open ARCO-ERA5 ───────────────────────────────────────────────────────────
print("Opening ARCO-ERA5 (anon)...", flush=True)
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    consolidated=True, storage_options={"token": "anon"}
)

# ─── Helper: daily mean/sum over India ────────────────────────────────────────
def daily_values(var_da, dates, agg='mean'):
    days = []
    for d in dates:
        day_str = d.strftime('%Y-%m-%d')
        data = var_da.sel(time=day_str)
        val  = data.sum('time') if agg == 'sum' else data.mean('time')
        days.append(val.load())
        print(f"    {day_str}: {agg}={float(val.mean()):.4f}", flush=True)
    return xr.concat(days, dim='time').assign_coords(time=DATES)

# ─── 1. Total Precipitation (mm/day) ─────────────────────────────────────────
print("\n[1/3] Downloading TP (mm/day)...", flush=True)
tp_raw = ds['total_precipitation'].sel(latitude=LAT, longitude=LON)
tp_out = daily_values(tp_raw, DATES, agg='sum') * 1000.0   # m → mm/day
tp_out.attrs.update({'units': 'mm/day', 'long_name': 'Total precipitation 24h sum (ARCO-ERA5)'})

# ─── 2. Z500 Geopotential Height (gpm) ───────────────────────────────────────
print("\n[2/3] Downloading Z500 (gpm)...", flush=True)
z_raw  = ds['geopotential'].sel(level=500, latitude=LAT, longitude=LON)
z_out  = daily_values(z_raw, DATES, agg='mean') / G         # m²/s² → gpm
z_out.attrs.update({'units': 'gpm', 'long_name': 'Geopotential height at 500 hPa (ARCO-ERA5)'})

# ─── 3. 2m Temperature (K) ───────────────────────────────────────────────────
print("\n[3/3] Downloading T2M (K)...", flush=True)
t_raw  = ds['2m_temperature'].sel(latitude=LAT, longitude=LON)
t_out  = daily_values(t_raw, DATES, agg='mean')
t_out.attrs.update({'units': 'K', 'long_name': '2m temperature daily mean (ARCO-ERA5)'})

# ─── Save ─────────────────────────────────────────────────────────────────────
print("\nSaving...", flush=True)
out_ds = xr.Dataset({'tp': tp_out, 'z500': z_out, 't2m': t_out})
out_ds.to_netcdf(OUT)

print(f"\n✓ WROTE {OUT}")
print(f"  TP    range: {float(tp_out.min()):.3f} – {float(tp_out.max()):.3f} mm/day")
print(f"  Z500  range: {float(z_out.min()):.1f} – {float(z_out.max()):.1f} gpm")
print(f"  T2M   range: {float(t_out.min()):.2f} – {float(t_out.max()):.2f} K  "
      f"({float(t_out.min())-273.15:.1f}–{float(t_out.max())-273.15:.1f} °C)")
print("DOWNLOAD_DONE", flush=True)
