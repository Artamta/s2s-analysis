import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box

init_date = '20260101'
out_path = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/verification/4model_z500_bias.png'

print("1. Loading ARCO-ERA5 Ground Truth (Z500)...")
ds_era5 = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', storage_options={'token': 'anon'})
era5_z500 = ds_era5['geopotential'].sel(level=500) / 9.80665

era5_daily = era5_z500.sel(time=slice('2026-01-01', '2026-02-11')).resample(time='1D').mean('time').compute()

print("2. Loading Forecast Models...")
# FuXi
fuxi_base = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_date}/member"
fuxi_list = [xr.open_mfdataset([f"{fuxi_base}/{m:02d}/{day:02d}.nc" for day in range(1, 43)], combine='nested', concat_dim='lead_time') for m in range(11)]
fuxi = xr.concat(fuxi_list, dim='member').mean('member')
fuxi_z500 = fuxi['__xarray_dataarray_variable__'].isel(channel=5) / 9.80665
fuxi_z500 = fuxi_z500.rename({'lat': 'latitude', 'lon': 'longitude'})

# Spire
spire = xr.open_zarr('/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr', group='mean_stddev')
spire_date = spire.sel(reference_time='2026-01-01')
spire_z500 = spire_date['geopotential_height_at_isobaric_levels'].sel(isobar=50000).isel(step=slice(0, 42))
spire_z500 = spire_z500.rename({'step': 'lead_time'})

# ECMWF
ecmwf_cf = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ecmwf/data/pl_cf_{init_date}.grib', engine='cfgrib')
ecmwf_z500 = ecmwf_cf['gh'].isel(step=slice(0, 42))
ecmwf_z500 = ecmwf_z500.rename({'step': 'lead_time'})

# NCEP
ncep_cf = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ncep/data/pl_cf_{init_date}.grib', engine='cfgrib')
ncep_z500 = ncep_cf['gh'].isel(step=slice(0, 42))
ncep_z500 = ncep_z500.rename({'step': 'lead_time'})

print("3. Regridding to 1.5 deg...")
target_lat = np.arange(90, -90.1, -1.5)
target_lon = np.arange(0, 360, 1.5)

fuxi_z500 = fuxi_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
spire_z500 = spire_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
ecmwf_z500 = ecmwf_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
ncep_z500 = ncep_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
era5_daily = era5_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')

print("4. Applying Indian Bounding Box...")
fuxi_ind = apply_indian_subcontinent_bounding_box(fuxi_z500)
spire_ind = apply_indian_subcontinent_bounding_box(spire_z500)
ecmwf_ind = apply_indian_subcontinent_bounding_box(ecmwf_z500)
ncep_ind = apply_indian_subcontinent_bounding_box(ncep_z500)
era5_ind = apply_indian_subcontinent_bounding_box(era5_daily)

def get_weekly_bias(model_da, week_idx):
    start, end = week_idx * 7, (week_idx + 1) * 7
    m_week = model_da.isel(lead_time=slice(start, end)).mean('lead_time')
    e_week = era5_ind.isel(time=slice(start, end)).mean('time')
    return (m_week - e_week).squeeze()

models = {'FuXi': fuxi_ind, 'SPIRE': spire_ind, 'ECMWF': ecmwf_ind, 'NCEP': ncep_ind}

print("5. Plotting...")
fig, axes = plt.subplots(4, 6, figsize=(24, 14), subplot_kw={'projection': ccrs.PlateCarree()})

for row, (name, da) in enumerate(models.items()):
    for w in range(6):
        ax = axes[row, w]
        bias = get_weekly_bias(da, w)
        im = ax.pcolormesh(bias.longitude, bias.latitude, bias, cmap='RdBu_r', vmin=-100, vmax=100, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        
        if row == 0:
            ax.set_title(f"Week {w+1}", fontsize=14)
        if w == 0:
            ax.text(-0.15, 0.5, name, va='center', ha='center', rotation='vertical', 
                    transform=ax.transAxes, fontsize=16, fontweight='bold')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Z500 Forecast Bias (Model - ERA5 Ground Truth) (m)')
plt.suptitle("Multi-Model Z500 Forecast Bias vs ARCO-ERA5 Ground Truth (Init: Jan 1, 2026)", fontsize=22, y=0.95)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"SUCCESS! Figure saved to: {out_path}")
