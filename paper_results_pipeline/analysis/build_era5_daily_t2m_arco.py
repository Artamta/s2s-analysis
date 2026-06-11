"""True daily-mean 2-m temperature from ARCO-ERA5 hourly. Output: analysis/era5_daily_t2m.nc (K)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xarray as xr
OUT='/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/era5_daily_t2m.nc'
print("opening ARCO (anon)...", flush=True)
ds=xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
               consolidated=True, storage_options={"token":"anon"})
t2=ds['2m_temperature'].sel(latitude=slice(40,3), longitude=slice(63,102))
dates=pd.date_range('2026-01-01','2026-05-10',freq='D'); out=[]; kept=[]
for i,d in enumerate(dates):
    s=d.strftime('%Y-%m-%d')
    try:
        day=t2.sel(time=s)
        if day.sizes.get('time',0)==0: print(f"[{s}] none -> stop",flush=True); break
        out.append(day.mean('time').load()); kept.append(d)
        if i%20==0: print(f"[{s}] mean={float(out[-1].mean()):.1f}K ({i+1}/{len(dates)})",flush=True)
    except Exception as e: print(f"[{s}] FAIL {e}",flush=True); break
xr.concat(out,dim='time').assign_coords(time=kept).rename('t2m').to_netcdf(OUT)
print(f"WROTE {OUT} days={len(kept)}",flush=True); print("BUILD_DONE",flush=True)
