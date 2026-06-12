#!/usr/bin/env python
"""
Compare S2S forecasts (ECMWF, NCEP, and Spire) at 1.5 degree resolution
for Week 1. Spire dataset is downsampled using bilinear interpolation.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def main():
    # Setup paths relative to script location
    analysis_dir = os.path.dirname(os.path.abspath(__file__)) # benchmark(jfm)/analysis
    benchmark_dir = os.path.dirname(analysis_dir)             # benchmark(jfm)
    base_dir = os.path.dirname(benchmark_dir)                 # s2s_anlysis
    
    ecmwf_file = os.path.join(benchmark_dir, "ecmwf", "data", "sfc_cf_20260101.grib")
    ncep_file = os.path.join(benchmark_dir, "ncep", "data", "sfc_cf_20260101.grib")
    spire_file = os.path.join(base_dir, "spire_era5", "s2s_verification", "weekly_absolute_v2.nc")
    
    print("Loading datasets...")
    # 1. Open ECMWF
    ecmwf_ds = xr.open_dataset(ecmwf_file, engine='cfgrib')
    # Calculate Week 1 mean of mx2t6 (forecast days 1 to 7)
    ecmwf_week1_raw = ecmwf_ds['mx2t6'].sel(step=slice(np.timedelta64(1, 'D'), np.timedelta64(7, 'D'))).mean(dim='step')
    ecmwf_week1 = ecmwf_week1_raw - 273.15 # Kelvin to Celsius
    
    # 2. Open NCEP
    ncep_ds = xr.open_dataset(ncep_file, engine='cfgrib')
    # Calculate Week 1 mean of mx2t6 (forecast days 1 to 7)
    ncep_week1_raw = ncep_ds['mx2t6'].sel(step=slice(np.timedelta64(1, 'D'), np.timedelta64(7, 'D'))).mean(dim='step')
    ncep_week1 = ncep_week1_raw - 273.15 # Kelvin to Celsius
    
    # 3. Open Spire
    spire_ds = xr.open_dataset(spire_file)
    # Select init_time 2026-01-01 and week 1
    spire_week1_raw = spire_ds['spire_tmax'].sel(init_time='2026-01-01', week=1)
    
    # Downsample Spire to match ECMWF's grid (1.5° resolution)
    print("Downsampling Spire dataset to match ECMWF 1.5° grid...")
    spire_week1 = spire_week1_raw.interp(latitude=ecmwf_ds.latitude, longitude=ecmwf_ds.longitude, method='linear')
    
    print("\n==========================================")
    print(" Grid Information after Downsampling")
    print("==========================================")
    print(f"ECMWF shape: {ecmwf_week1.shape}")
    print(f"NCEP shape:  {ncep_week1.shape}")
    print(f"Spire shape: {spire_week1.shape}")
    print(f"Latitude grid:  min={ecmwf_week1.latitude.values.min()}, max={ecmwf_week1.latitude.values.max()}, spacing=1.5°")
    print(f"Longitude grid: min={ecmwf_week1.longitude.values.min()}, max={ecmwf_week1.longitude.values.max()}, spacing=1.5°")
    
    # Plotting setup
    fig = plt.figure(figsize=(18, 5.5))
    
    # Common range for colorbar based on all three datasets
    vmin = min(ecmwf_week1.min().item(), ncep_week1.min().item(), spire_week1.min().item())
    vmax = max(ecmwf_week1.max().item(), ncep_week1.max().item(), spire_week1.max().item())
    vmin = np.floor(vmin)
    vmax = np.ceil(vmax)
    
    datasets = [
        ("ECMWF Control (1.5°)", ecmwf_week1),
        ("NCEP Control (1.5°)", ncep_week1),
        ("Spire (Downsampled to 1.5°)", spire_week1)
    ]
    
    lon, lat = ecmwf_ds.longitude.values, ecmwf_ds.latitude.values
    
    for i, (name, data) in enumerate(datasets, 1):
        if HAS_CARTOPY:
            projection = ccrs.PlateCarree()
            ax = fig.add_subplot(1, 3, i, projection=projection)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            
            im = ax.pcolormesh(lon, lat, data.values, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        else:
            ax = fig.add_subplot(1, 3, i)
            im = ax.pcolormesh(lon, lat, data.values, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            
        ax.set_title(name, fontsize=12, fontweight='bold')
    
    # Add common colorbar
    fig.subplots_adjust(bottom=0.25)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Weekly Mean Max Temperature (°C)', fontsize=11)
    
    fig.suptitle('S2S Forecast Comparison (Week 1 Forecast, Init: 2026-01-01) - 1.5° Resolution', fontsize=14, fontweight='bold', y=0.98)
    
    plot_path = os.path.join(analysis_dir, "figures", "ecmwf_ncep_spire_comparison_1.5deg.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n3-panel comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
