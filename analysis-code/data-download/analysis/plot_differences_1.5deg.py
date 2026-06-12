#!/usr/bin/env python
"""
Compare S2S forecast biases (Forecast minus ERA5 Ground Truth) for Spire, 
ECMWF, and NCEP at 1.5 degree resolution for Week 1 (daily maximum temperature, Tmax).
Spire data is loaded directly from the Arraylake API, and ERA5 data is
loaded directly from the public ARCO-ERA5 Google Cloud Storage dataset.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from arraylake import Client

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def get_grib_daily_max_week1(file_path):
    """Loads 6-hourly GRIB, computes daily maximum, and averages over Week 1 (days 1-7)."""
    ds = xr.open_dataset(file_path, engine='cfgrib')
    steps_hours = ds.step.values / np.timedelta64(1, 'h')
    day = (steps_hours - 0.1) // 24 + 1
    ds = ds.assign_coords(day=('step', day))
    
    # Calculate daily maximum from 6-hourly windows
    daily_max = ds['mx2t6'].groupby('day').max(dim='step')
    # Mean of daily max over Week 1 (forecast days 1 to 7)
    week1_mean = daily_max.sel(day=slice(1, 7)).mean(dim='day')
    return week1_mean - 273.15 # Kelvin to Celsius

def main():
    # Setup paths relative to script location
    analysis_dir = os.path.dirname(os.path.abspath(__file__)) # benchmark(jfm)/analysis
    benchmark_dir = os.path.dirname(analysis_dir)             # benchmark(jfm)
    
    # Load 6h GRIB files for daily max computation
    ecmwf_file = os.path.join(benchmark_dir, "ecmwf", "data", "sfc_cf_20260101_6h.grib")
    ncep_file = os.path.join(benchmark_dir, "ncep", "data", "sfc_cf_20260101_6h.grib")
    
    print("Loading benchmark forecasts from local GRIB files...")
    # 1. Open ECMWF
    ecmwf_ds = xr.open_dataset(ecmwf_file, engine='cfgrib')
    ecmwf_week1 = get_grib_daily_max_week1(ecmwf_file)
    
    # 2. Open NCEP
    ncep_week1 = get_grib_daily_max_week1(ncep_file)
    
    # Define regional bounds
    lat_min, lat_max = 0.0, 50.0
    lon_min, lon_max = 55.0, 105.0
    
    # 3. Load Spire directly from Arraylake API
    print("Connecting to Arraylake API for Spire data...")
    client = Client()
    repo = client.get_repo('artamta/s2s-research')
    session = repo.readonly_session('main')
    ds_spire = xr.open_zarr(session.store, group='mean_stddev')
    
    # Select Week 1 steps (days 1 to 7)
    steps = [np.timedelta64(d, 'D') for d in range(1, 8)]
    spire_week1_raw = ds_spire['air_temperature_max'].sel(
        reference_time='2026-01-01', 
        step=steps
    ).mean('step').compute() - 273.15
    
    # Flip latitude if needed to ensure ascending order
    if spire_week1_raw.latitude.values[0] > spire_week1_raw.latitude.values[-1]:
        spire_week1_raw = spire_week1_raw.isel(latitude=slice(None, None, -1))
    
    # Crop to regional domain
    spire_week1_raw = spire_week1_raw.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    
    # 4. Load ERA5 directly from ARCO-ERA5 GCP Zarr Store
    print("Connecting to ARCO-ERA5 GCS Zarr Store for Ground Truth...")
    ds_e = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', storage_options={'token': 'anon'})
    
    # Select Week 1 forecast dates (Jan 2 to Jan 8)
    da_era5 = ds_e['2m_temperature'].sel(
        latitude=slice(lat_max+1, lat_min-1),
        longitude=slice(lon_min-1, lon_max+1),
        time=slice('2026-01-02T00:00', '2026-01-08T23:00')
    )
    # Compute daily maximum and average over Week 1
    era5_week1_raw = da_era5.resample(time='1D').max('time').mean('time').compute() - 273.15
    
    # Downsample Spire and ERA5 to match ECMWF's grid (1.5° resolution)
    print("Interpolating Spire and ERA5 datasets to match ECMWF 1.5° grid...")
    spire_week1 = spire_week1_raw.interp(latitude=ecmwf_ds.latitude, longitude=ecmwf_ds.longitude, method='linear')
    era5_week1 = era5_week1_raw.interp(latitude=ecmwf_ds.latitude, longitude=ecmwf_ds.longitude, method='linear')
    
    # Calculate Differences (Model - ERA5 Ground Truth)
    print("Calculating forecast biases (Model - ERA5)...")
    spire_diff = spire_week1 - era5_week1
    ecmwf_diff = ecmwf_week1 - era5_week1
    ncep_diff = ncep_week1 - era5_week1
    
    # Plotting setup (1x3 horizontal grid)
    fig = plt.figure(figsize=(18, 5.5))
    
    # Determine symmetric limits for the divergent colormap
    max_diff = max(
        np.abs(spire_diff).max().item(),
        np.abs(ecmwf_diff).max().item(),
        np.abs(ncep_diff).max().item()
    )
    limit = np.ceil(max_diff)
    limit = min(limit, 8.0) # Cap at +/- 8 degC for clean visual contrast
    vmin, vmax = -limit, limit
    
    biases = [
        ("Spire Bias (Spire - ERA5)", spire_diff, 1),
        ("ECMWF Bias (ECMWF - ERA5)", ecmwf_diff, 2),
        ("NCEP Bias (NCEP - ERA5)", ncep_diff, 3)
    ]
    
    lon, lat = ecmwf_ds.longitude.values, ecmwf_ds.latitude.values
    
    for name, data, idx in biases:
        if HAS_CARTOPY:
            projection = ccrs.PlateCarree()
            ax = fig.add_subplot(1, 3, idx, projection=projection)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            
            im = ax.pcolormesh(lon, lat, data.values, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        else:
            ax = fig.add_subplot(1, 3, idx)
            im = ax.pcolormesh(lon, lat, data.values, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            
        ax.set_title(name, fontsize=12, fontweight='bold')
        
    # Add common colorbar at the bottom
    fig.subplots_adjust(bottom=0.25)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weekly Mean Daily-Max Temperature Bias (°C)', fontsize=11)
    
    fig.suptitle('S2S Forecast Bias vs ERA5 Ground Truth (Week 1, Init: 2026-01-01) - Tmax Comparison', fontsize=14, fontweight='bold', y=0.98)
    
    plot_path = os.path.join(analysis_dir, "figures", "forecast_bias_vs_era5_1.5deg.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nBias comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
