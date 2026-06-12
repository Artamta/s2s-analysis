"""
01_era5_jfm_climatology.py

Fetch ERA5 1990-2019 JFM day-of-year climatology for the India domain using
earth2studio's WB2Climatology data source (WeatherBench2 pre-computed climo).

Variables (earth2studio names):
  - t2m    : 2m temperature (K)
  - z500   : geopotential at 500 hPa (m²/s²  → divide by 9.80665 → gpm)
  - tp24   : total precipitation 24-hr (m  → *1000 → mm/day)

Output: era5_jfm_climatology_1990_2019.nc
  Dimensions: (dayofyear: 90, latitude: 201, longitude: 201)
              dayofyear = day-of-year values for Jan 1 – Mar 31 (DOY 1–90)
              lat/lon on native WB2 0.25° grid, sub-selected to India domain

Strategy: WB2Climatology returns data keyed by a datetime (it looks up the
corresponding DOY + 6h slot).  We pass one representative time per day of year
(using 2001 as a dummy year so DOY 1-90 = Jan 1 – Mar 31, non-leap year) at
00:00 UTC and take the daily mean of the four 6-hourly slots (00/06/12/18).
"""

import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import WB2Climatology

# ── parameters ──────────────────────────────────────────────────────────────
CLIMO_STORE = "1990-2019_6h_1440x721.zarr"   # 0.25°, 1990-2019
VARIABLES   = ["t2m", "z500", "tp24"]         # earth2studio names
# India domain (WB2 lat is ascending: -90 → 90)
LAT_MIN, LAT_MAX =  0.0, 50.0
LON_MIN, LON_MAX = 55.0, 105.0
OUTPUT_FILE = "era5_jfm_climatology_1990_2019.nc"

# ── build time list: 4 × 6-hourly slots for each of DOY 1–90 ────────────────
# use 2001 (non-leap) as dummy year; WB2Climatology ignores the year
jfm_days = pd.date_range("2001-01-01", "2001-03-31", freq="D")   # 90 days
times_6h  = pd.date_range("2001-01-01", "2001-04-01", freq="6h")[:-4]  # 360 steps

print(f"Fetching {len(times_6h)} 6-hourly slots for DOY 1-90 …")

# ── fetch from WB2Climatology ────────────────────────────────────────────────
climo_src = WB2Climatology(climatology_zarr_store=CLIMO_STORE, verbose=True)

# fetch() returns xr.DataArray with dims (time, variable, lat, lon)
da = climo_src(times_6h.to_list(), VARIABLES)
# da coords: time=360, variable=3, lat=721, lon=1440

# ── crop to India domain (WB2 lat is DESCENDING: 90 → -90) ──────────────────
da_india = da.sel(
    lat=slice(LAT_MAX, LAT_MIN),   # descending → slice(50, 0)
    lon=slice(LON_MIN, LON_MAX),
)
# flip lat to ascending so output is consistent
da_india = da_india.isel(lat=slice(None, None, -1))
print(f"India domain shape after crop: {da_india.shape}")

# ── daily mean (average four 6-hourly steps per day) ────────────────────────
da_daily = da_india.resample(time="1D").mean("time")   # (90, 3, lat, lon)

# ── build output dataset with unit conversions ───────────────────────────────
t2m_climo  = da_daily.sel(variable="t2m")           # K
z500_climo = da_daily.sel(variable="z500") / 9.80665 # m²/s² → gpm
tp_climo   = da_daily.sel(variable="tp24") * 1000.0  # m → mm/day

# label dayofyear coordinate
doy_vals = da_daily.time.dt.dayofyear.values  # [1, 2, ..., 90]

ds_out = xr.Dataset(
    {
        "t2m":  xr.DataArray(t2m_climo.values,  dims=["dayofyear", "latitude", "longitude"],
                             attrs={"units": "K",      "long_name": "2m temperature climatology"}),
        "z500": xr.DataArray(z500_climo.values, dims=["dayofyear", "latitude", "longitude"],
                             attrs={"units": "gpm",    "long_name": "Z500 geopotential height climatology"}),
        "tp":   xr.DataArray(tp_climo.values,   dims=["dayofyear", "latitude", "longitude"],
                             attrs={"units": "mm/day", "long_name": "Daily precip climatology"}),
    },
    coords={
        "dayofyear": doy_vals,
        "latitude":  t2m_climo.lat.values,
        "longitude": t2m_climo.lon.values,
    },
    attrs={
        "description":        "ERA5 JFM day-of-year climatology (DOY 1-90) for India domain",
        "source":             f"WeatherBench2 {CLIMO_STORE} via earth2studio WB2Climatology",
        "reference_period":   "1990-2019",
        "spatial_resolution": "0.25 deg x 0.25 deg",
    },
)

print(ds_out)
ds_out.to_netcdf(OUTPUT_FILE)
print(f"\nSaved → {OUTPUT_FILE}")
