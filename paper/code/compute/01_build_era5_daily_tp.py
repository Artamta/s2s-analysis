"""
Build a TRUE 24-hour daily-total ERA5 precipitation field from ARCO-ERA5 hourly
(gs://gcp-public-data-arco-era5), over the India box and the JFM-2026 verification
period, to replace the 6-hour-window ERA5 used previously.

Output: analysis/era5_daily_tp.nc   (tp in mm/day; dims time,lat,lon; 0.25 deg) --> moved to /storage
"""
import warnings, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xarray as xr

OUT = '/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_tp.nc'
print("opening ARCO-ERA5 (anon)...", flush=True)
ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                  consolidated=True, storage_options={"token": "anon"})
tp = ds['total_precipitation'].sel(latitude=slice(40, 3), longitude=slice(63, 102))

dates = pd.date_range('2026-01-01', '2026-05-10', freq='D')
daily_list, kept = [], []
for i, d in enumerate(dates):
    ds_str = d.strftime('%Y-%m-%d')
    try:
        day = tp.sel(time=ds_str)
        if day.sizes.get('time', 0) == 0:
            print(f"[{ds_str}] no data -> stop", flush=True); break
        val = (day.sum('time') * 1000.0).load()  # m → mm/day true daily total
        daily_list.append(val); kept.append(d)
        if i % 10 == 0:
            print(f"[{ds_str}] ok, domain-mean={float(val.mean()):.3f} mm/day  ({i+1}/{len(dates)})", flush=True)
    except Exception as e:
        print(f"[{ds_str}] FAIL {e} -> stop", flush=True); break

out = xr.concat(daily_list, dim='time').assign_coords(time=kept).rename('tp')
# explicitly set correct units after ×1000 m→mm conversion
out.attrs['units'] = 'mm/day'
out.attrs['long_name'] = 'Total precipitation (true 24-h daily total, ARCO-ERA5 hourly sum ×1000)'
out.to_netcdf(OUT)
print(f"\nWROTE {OUT}  days={len(kept)}  range {kept[0].date()}..{kept[-1].date()}", flush=True)
print("mean daily tp (mm/day):", round(float(out.mean()), 3), flush=True)
print("BUILD_DONE", flush=True)
