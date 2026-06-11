import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box

# ERA5 Ground Truth (Using the 135-day dataset we just downloaded)
era5_ds = xr.open_dataset('/storage/raj.ayush/benchmark(jfm)/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': '2t'})
era5_t2m = era5_ds['t2m'] - 273.15 # Convert to Celsius
era5_t2m = apply_indian_subcontinent_bounding_box(era5_t2m).rename({'latitude': 'lat', 'longitude': 'lon'})
# Week 2 Target: Jan 8 to Jan 14
era5_week2 = era5_t2m.sel(time=slice('2026-01-08', '2026-01-14')).mean('time')

print("Loaded ERA5")

# FuXi
init_date = '20260101'
fuxi_base = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_date}/member"
fuxi_list = [xr.open_mfdataset([f"{fuxi_base}/{m:02d}/{day:02d}.nc" for day in range(8, 15)], combine='nested', concat_dim='lead_time') for m in range(11)]
fuxi = xr.concat(fuxi_list, dim='member').mean('member')
fuxi_t2m = fuxi['__xarray_dataarray_variable__'].isel(channel=66) - 273.15
fuxi_t2m = fuxi_t2m.rename({'lat': 'latitude', 'lon': 'longitude'})
fuxi_t2m = apply_indian_subcontinent_bounding_box(fuxi_t2m.mean('lead_time')).rename({'latitude': 'lat', 'longitude': 'lon'})

print("Loaded FuXi")

# Spire (Using air_temperature at 1000hPa surface)
spire = xr.open_zarr('/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr', group='mean_stddev')
spire_date = spire.sel(reference_time='2026-01-01')
spire_t2m = spire_date['air_temperature'].isel(step=slice(7, 14)) - 273.15
spire_t2m = apply_indian_subcontinent_bounding_box(spire_t2m.mean('step')).rename({'latitude': 'lat', 'longitude': 'lon'})

print("Loaded Spire")

# Regrid everything to a common 1.5 deg grid
target_lat = np.arange(40, 5, -1.5)
target_lon = np.arange(60, 100, 1.5)

fuxi_regrid = fuxi_t2m.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
spire_regrid = spire_t2m.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
era5_regrid = era5_week2.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

f_bias = fuxi_regrid - era5_regrid
s_bias = spire_regrid - era5_regrid

print("Regridded and Calculated Bias")

# 2x3 Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

models = [(fuxi_regrid, f_bias, "FuXi S2S"), (spire_regrid, s_bias, "Spire S2S")]

for row, (fcst, bias, name) in enumerate(models):
    ax1, ax2, ax3 = axes[row]
    
    im1 = ax1.pcolormesh(fcst.lon, fcst.lat, fcst, cmap='RdYlBu_r', shading='auto', vmin=-10, vmax=35)
    ax1.set_title(f"{name} Week 2 Forecast (T2M)", fontsize=10, fontweight='bold')
    
    im2 = ax2.pcolormesh(era5_regrid.lon, era5_regrid.lat, era5_regrid, cmap='RdYlBu_r', shading='auto', vmin=-10, vmax=35)
    ax2.set_title(f"Copernicus ERA5 Ground Truth", fontsize=10, fontweight='bold')
    
    vmax_bias = 5.0
    im3 = ax3.pcolormesh(bias.lon, bias.lat, bias, cmap='RdBu_r', shading='auto', vmin=-vmax_bias, vmax=vmax_bias)
    ax3.set_title(f"True Bias ({name} - ERA5)", fontsize=10, fontweight='bold')

    # Add Colorbars on the right of each row
    fig.colorbar(im1, ax=[ax1, ax2], label='2m Temperature (°C)', fraction=0.02)
    fig.colorbar(im3, ax=ax3, label='Bias Error (°C)', fraction=0.04)

plt.suptitle("Week 2 (Jan 8-14) Target: 2m Temperature Spatial Validation vs Copernicus ERA5", fontsize=16, fontweight='bold')
out_path = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/figures/verification/t2m_fuxi_spire_era5_week2.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Image saved to {out_path}")
