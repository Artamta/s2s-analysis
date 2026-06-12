import os
import xarray as xr
import pandas as pd
import gc

OUT_DIR = '/storage/raj.ayush/s2s-forecast-data/era5/data'
os.makedirs(OUT_DIR, exist_ok=True)

print("Connecting to Google Cloud ARCO-ERA5 dataset...")
# Open the dataset without loading it into RAM
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    consolidated=True
)

# Subset domain lazily
ds_domain = ds.sel(latitude=slice(50, 0), longitude=slice(55, 105))

# Generate daily dates
dates = pd.date_range(start='2026-01-01', end='2026-05-15', freq='D')

for date in dates:
    date_str = date.strftime('%Y%m%d')
    out_file = os.path.join(OUT_DIR, f'era5_surface_z500_{date_str}.nc')
    
    if os.path.exists(out_file):
        print(f"[{date_str}] Already exists, skipping!")
        continue
        
    print(f"[{date_str}] Downloading high-res ARCO-ERA5 data...")
    
    # Slice exactly 24 hours for this day
    day_slice = ds_domain.sel(time=slice(date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')))
    
    try:
        z500 = day_slice['geopotential'].sel(level=500)
        t2m = day_slice['2m_temperature']
        tp = day_slice['total_precipitation']
        
        out_ds = xr.Dataset({'z500': z500, 't2m': t2m, 'tp': tp})
        
        # Load and write to disk
        out_ds.to_netcdf(out_file)
        print(f"[{date_str}] Success! Saved to {out_file}")
    except Exception as e:
        print(f"[{date_str}] Error: {e}")
        
    # Free memory explicitly
    del day_slice
    gc.collect()

print("High-Res ERA5 Ground truth completely downloaded!")
