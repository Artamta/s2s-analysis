#!/usr/bin/env python
"""
Compare S2S forecasts (ECMWF, NCEP, and Spire) against ERA5 Ground Truth
at 1.5 degree resolution for Week 1 (daily maximum temperature, Tmax).
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
    
    print("\n==========================================")
    print(" Grid Information after Downsampling")
    print("==========================================")
    print(f"ECMWF shape: {ecmwf_week1.shape}")
    print(f"NCEP shape:  {ncep_week1.shape}")
    print(f"Spire shape: {spire_week1.shape}")
    print(f"ERA5 shape:  {era5_week1.shape}")
    print(f"Latitude grid:  min={ecmwf_week1.latitude.values.min()}, max={ecmwf_week1.latitude.values.max()}, spacing=1.5°")
    print(f"Longitude grid: min={ecmwf_week1.longitude.values.min()}, max={ecmwf_week1.longitude.values.max()}, spacing=1.5°")
    
    # Plotting setup (2x2 grid)
    fig = plt.figure(figsize=(14, 11))
    
    # Common range for colorbar based on all four datasets
    vmin = min(ecmwf_week1.min().item(), ncep_week1.min().item(), spire_week1.min().item(), era5_week1.min().item())
    vmax = max(ecmwf_week1.max().item(), ncep_week1.max().item(), spire_week1.max().item(), era5_week1.max().item())
    vmin = np.floor(vmin)
    vmax = np.ceil(vmax)
    
    panels = [
        ("ERA5 Ground Truth (Tmax, 1.5°)", era5_week1, 1),
        ("Spire Forecast (Tmax, 1.5°)", spire_week1, 2),
        ("ECMWF Control Forecast (Tmax, 1.5°)", ecmwf_week1, 3),
        ("NCEP Control Forecast (Tmax, 1.5°)", ncep_week1, 4)
    ]
    
    lon, lat = ecmwf_ds.longitude.values, ecmwf_ds.latitude.values
    
    for name, data, idx in panels:
        if HAS_CARTOPY:
            projection = ccrs.PlateCarree()
            ax = fig.add_subplot(2, 2, idx, projection=projection)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            
            im = ax.pcolormesh(lon, lat, data.values, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        else:
            ax = fig.add_subplot(2, 2, idx)
            im = ax.pcolormesh(lon, lat, data.values, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            
        ax.set_title(name, fontsize=12, fontweight='bold')
    
    # Add common colorbar at the bottom
    fig.subplots_adjust(bottom=0.15, hspace=0.15)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weekly Mean Daily-Max Temperature (°C)', fontsize=12)
    
    fig.suptitle('S2S Forecast vs ERA5 Ground Truth (Week 1, Init: 2026-01-01) - Tmax Comparison', fontsize=15, fontweight='bold', y=0.95)
    
    plot_path = os.path.join(analysis_dir, "figures", "ecmwf_ncep_spire_era5_comparison_1.5deg.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n4-panel comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
