import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pandas as pd

fuxi_base = "/storage/raj.ayush/s2s-forecast-data/fuxi/output/20260101/member"
era5_base = "/storage/raj.ayush/s2s-forecast-data/era5/data"
out_path = "/home/raj.ayush/s2s/s2s_anlysis/analysis-code/verification/fuxi_weekly_t2m_20260101.png"

print("1. Loading FuXi Ensemble...")
members = [f"{i:02d}" for i in range(11)]
fuxi_list = []
for m in members:
    paths = [f"{fuxi_base}/{m}/{day:02d}.nc" for day in range(1, 43)]
    ds = xr.open_mfdataset(paths, combine='nested', concat_dim='lead_time')
    fuxi_list.append(ds)

fuxi = xr.concat(fuxi_list, dim='member').mean('member')
# Extract t2m (index 65 in FuXi channel list)
fuxi_t2m = fuxi['__xarray_dataarray_variable__'].isel(channel=65)

print("2. Loading ERA5 Ground Truth...")
dates = pd.date_range('2026-01-01', periods=42).strftime('%Y%m%d')
era5_files = [f"{era5_base}/era5_surface_z500_{d}.nc" for d in dates]
era5 = xr.open_mfdataset(era5_files, combine='nested', concat_dim='lead_time')
era5_t2m = era5['t2m'].mean('time') if 'time' in era5.dims and era5.sizes['time'] > 1 else era5['t2m']

print("3. Aligning Grids...")
fuxi_t2m = fuxi_t2m.rename({'lat': 'latitude', 'lon': 'longitude'})
# FuXi is 90 to -90, ERA5 might be 90 to -90. Just regrid to ERA5
fuxi_t2m = fuxi_t2m.interp(latitude=era5_t2m.latitude, longitude=era5_t2m.longitude, method='linear')

print("4. Calculating Weekly Averages...")
weeks_fuxi = []
weeks_era5 = []
weeks_bias = []

for w in range(6):
    start, end = w * 7, (w + 1) * 7
    # Mean over the 7 days of the week
    f_week = fuxi_t2m.isel(lead_time=slice(start, end)).mean('lead_time')
    e_week = era5_t2m.isel(lead_time=slice(start, end)).mean('lead_time')
    bias = f_week - e_week
    
    weeks_fuxi.append(f_week)
    weeks_era5.append(e_week)
    weeks_bias.append(bias)

print("5. Plotting 18-Panel Grid...")
fig, axes = plt.subplots(3, 6, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Kelvin to Celsius for better readability
def k_to_c(da): return da - 273.15

for w in range(6):
    ax_f = axes[0, w]
    ax_e = axes[1, w]
    ax_b = axes[2, w]
    
    # FuXi
    im1 = ax_f.pcolormesh(fuxi_t2m.longitude, fuxi_t2m.latitude, k_to_c(weeks_fuxi[w]), 
                          cmap='coolwarm', vmin=-30, vmax=40, transform=ccrs.PlateCarree())
    ax_f.coastlines(color='black', linewidth=0.5)
    ax_f.set_title(f"FuXi Week {w+1}")
    
    # ERA5
    im2 = ax_e.pcolormesh(era5_t2m.longitude, era5_t2m.latitude, k_to_c(weeks_era5[w]), 
                          cmap='coolwarm', vmin=-30, vmax=40, transform=ccrs.PlateCarree())
    ax_e.coastlines(color='black', linewidth=0.5)
    ax_e.set_title(f"ERA5 Week {w+1}")
    
    # Bias
    im3 = ax_b.pcolormesh(fuxi_t2m.longitude, fuxi_t2m.latitude, weeks_bias[w], 
                          cmap='bwr', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
    ax_b.coastlines(color='black', linewidth=0.5)
    ax_b.set_title(f"Bias (FuXi - ERA5)")

# Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.45, 0.015, 0.4])
fig.colorbar(im1, cax=cbar_ax1, label='Temperature (°C)')

cbar_ax2 = fig.add_axes([0.92, 0.1, 0.015, 0.25])
fig.colorbar(im3, cax=cbar_ax2, label='Bias (°C)')

plt.suptitle("FuXi vs ERA5: Weekly Global Surface Temperature Evolution (Init: 2026-01-01)", fontsize=20, y=0.98)
plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)

plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot perfectly saved to {out_path}!")
