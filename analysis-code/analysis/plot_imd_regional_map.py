import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr

# Load ERA5 as a background map
era5_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': '2t'})
era5_t2m = era5_ds['t2m'] - 273.15
era5_t2m = era5_t2m.rename({'latitude': 'lat', 'longitude': 'lon'})
era5_bg = era5_t2m.sel(time='2026-01-08', method='nearest')

# Slice to India Bounding Box
era5_bg = era5_bg.sel(lat=slice(38.0, 5.0), lon=slice(65.0, 100.0))

# Load Metrics
df = pd.read_csv('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/regional_metrics_week2.csv')

# Bounding Boxes
imd_regions_bounds = {
    'Northwest India':      (22.0, 38.0, 68.0, 82.0),
    'Central India':        (18.0, 28.0, 72.0, 89.0),
    'South Peninsula':      (8.0,  20.0, 72.0, 85.0),
    'East Northeast India': (20.0, 30.0, 85.0, 98.0)
}

fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Plot Background
im = ax.pcolormesh(era5_bg.lon, era5_bg.lat, era5_bg, cmap='Greys', shading='auto', alpha=0.3)

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']

for i, (idx, row) in enumerate(df.iterrows()):
    region = row['IMD Region']
    min_lat, max_lat, min_lon, max_lon = imd_regions_bounds[region]
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    # Draw Rectangle
    rect = patches.Rectangle((min_lon, min_lat), width, height, linewidth=3, edgecolor=colors[i], facecolor='none', alpha=0.8, linestyle='--')
    ax.add_patch(rect)
    
    # Annotate Text
    text_str = f"{region}\n" \
               f"FuXi Bias: {row['FuXi Mean Bias (°C)']}°C | RMSE: {row['FuXi RMSE (°C)']}°C\n" \
               f"Spire Bias: {row['Spire Mean Bias (°C)']}°C | RMSE: {row['Spire RMSE (°C)']}°C"
               
    # Determine text placement based on region to avoid overlap
    if region == 'Northwest India':
        xy = (min_lon + 1, max_lat - 2)
    elif region == 'Central India':
        xy = (min_lon + 1, min_lat + 1)
    elif region == 'South Peninsula':
        xy = (min_lon + 1, max_lat - 3)
    else:
        xy = (min_lon + 1, max_lat - 2)
        
    ax.text(xy[0], xy[1], text_str, fontsize=10, fontweight='bold', color='black',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors[i], boxstyle='round,pad=0.5'))

ax.set_title("S2S T2M Spatial Metrics over IMD Homogeneous Regions (Week 2)", fontsize=16, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle=':', alpha=0.6)

out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/imd_regional_metrics_map.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Map saved to {out_path}")
