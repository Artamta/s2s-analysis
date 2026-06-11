import xarray as xr
import os

out_path = '/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc'

print("1. Connecting to WeatherBench2 GCS Zarr...")
ds = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr', storage_options={'token': 'anon'})

print("2. Extracting 2m Temperature (t2m)...")
t2m = ds['2m_temperature'].mean('hour')

print("3. Extracting Total Precipitation (tp)...")
tp = ds['total_precipitation_24hr'].mean('hour')

print("4. Extracting Geopotential at 500hPa (z500)...")
z500 = ds['geopotential'].sel(level=500).mean('hour')

print("5. Merging and Standardizing Variables...")
ds_clim = xr.Dataset({
    't2m': t2m,
    'tp': tp,
    'z500': z500
})

print(f"6. Downloading from Google Cloud and Saving to {out_path}...")
print("This may take 3-5 minutes depending on network speed...")
ds_clim.to_netcdf(out_path)

print("SUCCESS: 30-Year Climatology Download Complete!")
