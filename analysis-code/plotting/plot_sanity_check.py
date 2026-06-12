import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')

print("Starting Sanity Check Visualization...")

# Target settings
INIT_DATE = '2026-01-01'
INIT_STR = '20260101'
LEAD_DAY = 7
TARGET_DATE = '2026-01-08'

# Indian Subcontinent Bounds
lat_slice = slice(38, 5)
lon_slice = slice(65, 100)

def standardize_grid(da):
    """Regrid to 1.5 deg and slice to India to ensure identical shapes."""
    if 'latitude' in da.dims: da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    # Sort latitudes to ensure descending order for slice(38, 5)
    if da.lat[0] < da.lat[-1]:
        da = da.sortby('lat', ascending=False)
        
    target_lat = np.arange(38, 5, -1.5)
    target_lon = np.arange(65, 100, 1.5)
    return da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

# 1. Load ERA5 (Ground Truth) for Target Date
print("Loading ERA5...")
era_nc = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface_z500_{TARGET_DATE.replace("-","")}.nc')
# Daily Max T2M from hourly
era_t2m = standardize_grid(era_nc['t2m'].max('time'))
# Daily TP (accumulation)
era_tp = standardize_grid(era_nc['tp'].sum('time') * 1000) # to mm
# Z500 Daily Mean
era_z500 = standardize_grid(era_nc['z500'].mean('time') / 9.80665) # to gpm

# 2. Load ECMWF
print("Loading ECMWF...")
ec_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_cf_{INIT_STR}.grib', engine='cfgrib', filter_by_keys={'shortName': 'mx2t6'})
ec_t2m = standardize_grid(ec_ds['mx2t6'].isel(step=LEAD_DAY-1))

ec_ds_tp = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/sfc_new_cf_{INIT_STR}.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
# ECMWF TP is accumulated from day 0. We need (Day 7 - Day 6)
if LEAD_DAY == 1:
    ec_tp = ec_ds_tp['tp'].isel(step=LEAD_DAY-1)
else:
    ec_tp = ec_ds_tp['tp'].isel(step=LEAD_DAY-1) - ec_ds_tp['tp'].isel(step=LEAD_DAY-2)
ec_tp = standardize_grid(ec_tp)

ec_ds_z = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/pl_cf_{INIT_STR}.grib', engine='cfgrib')
ec_z500 = standardize_grid(ec_ds_z['gh'].isel(step=LEAD_DAY-1))

# 3. Load NCEP
print("Loading NCEP...")
nc_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_cf_{INIT_STR}.grib', engine='cfgrib', filter_by_keys={'shortName': 'mx2t6'})
nc_t2m = standardize_grid(nc_ds['mx2t6'].isel(step=LEAD_DAY-1))

nc_ds_tp = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/sfc_cf_{INIT_STR}.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
if LEAD_DAY == 1:
    nc_tp = nc_ds_tp['tp'].isel(step=LEAD_DAY-1)
else:
    nc_tp = nc_ds_tp['tp'].isel(step=LEAD_DAY-1) - nc_ds_tp['tp'].isel(step=LEAD_DAY-2)
nc_tp = standardize_grid(nc_tp)

nc_ds_z = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/pl_cf_{INIT_STR}.grib', engine='cfgrib')
nc_z500 = standardize_grid(nc_ds_z['gh'].isel(step=LEAD_DAY-1))

# 4. Load FuXi
print("Loading FuXi...")
fx_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/fuxi/output/{INIT_STR}/member/00/{LEAD_DAY:02d}.nc')['__xarray_dataarray_variable__'].squeeze()
fx_t2m = standardize_grid(fx_ds.sel(channel='t2m'))
fx_tp = standardize_grid(fx_ds.sel(channel='tp'))
fx_z500 = standardize_grid(fx_ds.sel(channel='z500') / 9.80665) # to gpm

# 5. Load Spire
print("Loading Spire...")
sp_ds = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')
sp_ds = sp_ds.sel(reference_time=INIT_DATE).isel(step=LEAD_DAY-1)
sp_t2m = standardize_grid(sp_ds['air_temperature'])
sp_z500 = standardize_grid(sp_ds['geopotential_height_at_isobaric_levels'].sel(isobar=50000)) # 500 hPa is 50000 Pa

# Spire TP is already a daily rate in kg/m^2 (mm), no need to subtract
sp_tp = standardize_grid(sp_ds['precipitation_amount'])

print("Plotting...")
# --- PLOTTING ---
fig, axes = plt.subplots(3, 5, figsize=(25, 15), subplot_kw={'projection': ccrs.PlateCarree()})

models = ['ERA5 (Truth)', 'ECMWF', 'NCEP', 'FuXi', 'SPIRE']
vars_data = [
    [era_t2m, ec_t2m, nc_t2m, fx_t2m, sp_t2m], # T2M
    [era_tp, ec_tp, nc_tp, fx_tp, sp_tp],      # TP
    [era_z500, ec_z500, nc_z500, fx_z500, sp_z500] # Z500
]
var_names = ['Temp (K)', 'Precip (mm/day)', 'Z500 (gpm)']
cmaps = ['coolwarm', 'Blues', 'viridis']

vmins = [240, 0, 5200]
vmaxs = [310, 30, 5900]

for row in range(3):
    vmin = vmins[row]
    vmax = vmaxs[row]
    
    for col in range(5):
        ax = axes[row, col]
        ax.set_extent([65, 100, 5, 38], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='black')
        
        data = vars_data[row][col]
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmaps[row], vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        if row == 0:
            ax.set_title(models[col], fontsize=18, fontweight='bold', pad=15)
        if col == 0:
            ax.text(-0.25, 0.5, var_names[row], va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontsize=18, fontweight='bold')
            
    # Add a single colorbar for the entire row
    cbar_ax = fig.add_axes([0.92, 0.68 - row*0.27, 0.015, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
output_path = '/home/raj.ayush/.gemini/antigravity-cli/brain/3948ed00-797e-4f08-bcda-cf28d16936ed/sanity_check_maps.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved plot to {output_path}")

with open('/home/raj.ayush/.gemini/antigravity-cli/brain/3948ed00-797e-4f08-bcda-cf28d16936ed/sanity_check_maps.md', 'w') as f:
    f.write("# Model Outputs Sanity Check\n\n")
    f.write("Showing Lead Day 7 (Initialized Jan 1, 2026. Target Jan 8, 2026)\n\n")
    f.write("![Sanity Check Maps](sanity_check_maps.png)")
