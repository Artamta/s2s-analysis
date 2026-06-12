import xarray as xr
import os
import numpy as np

path = '/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr'

print(f"Checking {path}...")
print(f"Size on disk: {sum(f.stat().st_size for f in os.scandir(path) if f.is_file())} bytes (root only, let's just check xarray)")

try:
    ds_ms = xr.open_zarr(path, group="mean_stddev")
    print("\n=== MEAN_STDDEV GROUP ===")
    print(ds_ms)
    
    # Check a specific variable
    if 'air_temperature' in ds_ms:
        print("\nChecking 'air_temperature'...")
        arr = ds_ms['air_temperature'].values
        print(f"Shape: {arr.shape}")
        print(f"Contains NaNs: {np.isnan(arr).any()}")
        print(f"Min: {np.nanmin(arr):.2f}, Max: {np.nanmax(arr):.2f}")
    
    ds_pctl = xr.open_zarr(path, group="percentiles")
    print("\n=== PERCENTILES GROUP ===")
    print(ds_pctl)
    
except Exception as e:
    print(f"Error reading Zarr: {e}")
