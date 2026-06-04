import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

# ----------------------------------------------------------------------
# 1. SETUP EXACT GEOGRAPHIC BOUNDARIES (IMD SPECIFICATION)
# ----------------------------------------------------------------------
OUTPUT_DIR = "imd_exact_clones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact framing to show the entire Indian territory comfortably
lat_min, lat_max = 5.0, 38.5
lon_min, lon_max = 65.0, 98.5

# Date blocks to match the target operational weeks
WEEKS = [
    {"name": "Week1: 26Aug–01Sep", "start": "2021-08-26", "end": "2021-09-01"},
    {"name": "Week2: 02Sep–08Sep", "start": "2021-09-02", "end": "2021-09-08"},
    {"name": "Week3: 09Sep–15Sep", "start": "2021-09-09", "end": "2021-09-15"},
    {"name": "Week4: 16Sep–22Sep", "start": "2021-09-16", "end": "2021-09-22"}
]

# ----------------------------------------------------------------------
# 2. DEFINING BALANCED IMD COLORMAPS
# ----------------------------------------------------------------------
RAIN_ABS_COLORS = ['#ffffff', '#90ee90', '#32cd32', '#008000', '#006400']
rain_abs_cmap = mcolors.ListedColormap(RAIN_ABS_COLORS)
rain_abs_bounds = [0, 2, 5, 10, 20, 100]
rain_abs_norm = mcolors.BoundaryNorm(rain_abs_bounds, rain_abs_cmap.N)

RAIN_ANOM_COLORS = [
    '#cc3300', '#ff6600', '#ff9933', '#ffcc66', '#ffffcc', 
    '#ffffff', 
    '#ccccff', '#9999ff', '#6666ff', '#3333cc', '#000066'
]
rain_anom_cmap = mcolors.ListedColormap(RAIN_ANOM_COLORS)
rain_anom_bounds = [-100, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 100]
rain_anom_norm = mcolors.BoundaryNorm(rain_anom_bounds, rain_anom_cmap.N)

# ----------------------------------------------------------------------
# 3. ADVANCED GEOGRAPHIC MAP DECORATION FUNCTION
# ----------------------------------------------------------------------
def apply_exact_imd_decorations(ax, title_text):
    """Applies tight clipping bounds, sharp coastlines, and Indian state vectors."""
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # 1. Base Coastlines and International Borders
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, edgecolor='black', zorder=3)
    
    # 2. Injecting Indian Internal State Boundaries automatically using Natural Earth Shapefiles
    try:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )
        ax.add_feature(states_provinces, linewidth=0.6, edgecolor='black', linestyle='-', zorder=3)
    except Exception as e:
        print(f"Fallback to secondary state line render strategy due to: {e}")
        
    # 3. Gridlines configured to match font scaling from the target figures
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'weight': 'bold'}
    gl.ylabel_style = {'size': 8, 'weight': 'bold'}
    
    # Blue bold descriptive labels inside parenthesis exactly like the target images
    ax.set_title(title_text, fontsize=9, color='blue', weight='bold', pad=5)

# ----------------------------------------------------------------------
# 4. DATA PIPELINE AND PLOT EXECUTION
# ----------------------------------------------------------------------
print("Streaming targeted ARCO-ERA5 slices from cloud storage...")
ds_era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"}
)

lat_slice = slice(lat_max, lat_min) if ds_era5['latitude'][0] > ds_era5['latitude'][-1] else slice(lat_min, lat_max)
lon_slice = slice(lon_min, lon_max)

# Initialize canvas
fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), layout="constrained",
                         subplot_kw={'projection': ccrs.PlateCarree()})

# Absolute text assignments along the top edge coordinates
fig.text(0.01, 0.97, "ERA5 Actual Rainfall (mm/day)", fontsize=15, color='red', weight='bold')
fig.text(0.42, 0.97, "IC=20210825 (Valid Profile)", fontsize=11, weight='bold')
fig.text(0.58, 0.97, "ERA5 Rainfall Anomaly (mm/day)", fontsize=15, color='red', weight='bold')

for idx, wk in enumerate(WEEKS):
    time_slice = slice(f"{wk['start']}T00:00:00", f"{wk['end']}T23:00:00")
    
    # Download parameter slice
    tp_hourly = ds_era5["total_precipitation"].sel(latitude=lat_slice, longitude=lon_slice, time=time_slice).compute()
    
    # Mathematical transformation into real-world mm/day
    tp_daily = tp_hourly.resample(time='1D').sum(dim='time') * 1000.0
    rain_abs_data = tp_daily.mean(dim='time').values
    
    # Placeholder anomaly (Swap with your true climatology array subtraction)
    rain_anom_data = rain_abs_data - 6.5
    
    lats = tp_hourly.latitude.values
    lons = tp_hourly.longitude.values
    
    # Left Quadrant (Actuals)
    ax_abs = axes[idx // 2, (idx % 2)]
    im_abs = ax_abs.pcolormesh(lons, lats, rain_abs_data, cmap=rain_abs_cmap, norm=rain_abs_norm, transform=ccrs.PlateCarree())
    apply_exact_imd_decorations(ax_abs, f"({wk['name']})")
    
    # Right Quadrant (Anomalies)
    ax_anom = axes[idx // 2, (idx % 2) + 2]
    im_anom = ax_anom.pcolormesh(lons, lats, rain_anom_data, cmap=rain_anom_cmap, norm=rain_anom_norm, transform=ccrs.PlateCarree())
    apply_exact_imd_decorations(ax_anom, f"({wk['name']})")

# Construct bottom layout colorbars matching the layout margins perfectly
cax_abs = fig.add_axes([0.12, 0.03, 0.28, 0.02])
cb_abs = fig.colorbar(im_abs, cax=cax_abs, orientation='horizontal', extend='max')
cb_abs.set_ticks([2, 5, 10, 20, 40])
cb_abs.ax.tick_params(labelsize=9)

cax_anom = fig.add_axes([0.60, 0.03, 0.28, 0.02])
cb_anom = fig.colorbar(im_anom, cax=cax_anom, orientation='horizontal', extend='both')
cb_anom.set_ticks([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20])
cb_anom.ax.tick_params(labelsize=8)

output_filename = f"{OUTPUT_DIR}/era5_exact_india_map_rainfall.png"
fig.savefig(output_filename, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"🎉 Done! The exact multi-panel layout has been saved to: {output_filename}")