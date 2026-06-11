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
out_path = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/verification/4model_z500_anomalies.png'

print("1. Loading ERA5 Climatology Baseline (Z500)...")
clim = xr.open_dataset('/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc')
clim_z500 = clim['z500']
# ERA5 climatology is Geopotential (m^2/s^2). Convert to Geopotential Height (m):
clim_z500 = clim_z500 / 9.80665

print("2. Loading Forecast Models...")
# FuXi
print(" - FuXi...")
fuxi_base = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_date}/member"
fuxi_list = []
for m in [f"{i:02d}" for i in range(11)]:
    paths = [f"{fuxi_base}/{m}/{day:02d}.nc" for day in range(1, 43)]
    fuxi_list.append(xr.open_mfdataset(paths, combine='nested', concat_dim='lead_time'))
fuxi = xr.concat(fuxi_list, dim='member').mean('member')
# FuXi channel 5 is Z500. It is in Geopotential (m^2/s^2). Convert to Height.
fuxi_z500 = fuxi['__xarray_dataarray_variable__'].isel(channel=5) / 9.80665
fuxi_z500 = fuxi_z500.rename({'lat': 'latitude', 'lon': 'longitude'})

# Spire
print(" - SPIRE...")
spire = xr.open_zarr('/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr', group='mean_stddev')
spire_date = spire.sel(reference_time='2026-01-01')
# Spire is already in Geopotential Height (m)
spire_z500 = spire_date['geopotential_height_at_isobaric_levels'].sel(isobar=50000).isel(step=slice(0, 42))
spire_z500 = spire_z500.rename({'step': 'lead_time'})

# ECMWF
print(" - ECMWF...")
ecmwf_cf = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ecmwf/data/pl_cf_{init_date}.grib', engine='cfgrib')
# ECMWF 'gh' is Geopotential Height (m)
ecmwf_z500 = ecmwf_cf['gh'].isel(step=slice(0, 42))
ecmwf_z500 = ecmwf_z500.rename({'step': 'lead_time'})

# NCEP
print(" - NCEP...")
ncep_cf = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ncep/data/pl_cf_{init_date}.grib', engine='cfgrib')
ncep_z500 = ncep_cf['gh'].isel(step=slice(0, 42))
ncep_z500 = ncep_z500.rename({'step': 'lead_time'})

print("3. Regridding all models to 1.5 deg Global Standard...")
target_lat = np.arange(90, -90.1, -1.5)
target_lon = np.arange(0, 360, 1.5)

fuxi_z500 = fuxi_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
spire_z500 = spire_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
ecmwf_z500 = ecmwf_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
ncep_z500 = ncep_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')
clim_z500 = clim_z500.interp(latitude=target_lat, longitude=target_lon, method='linear')

print("4. Applying Safe Indian Bounding Box...")
fuxi_ind = apply_indian_subcontinent_bounding_box(fuxi_z500)
spire_ind = apply_indian_subcontinent_bounding_box(spire_z500)
ecmwf_ind = apply_indian_subcontinent_bounding_box(ecmwf_z500)
ncep_ind = apply_indian_subcontinent_bounding_box(ncep_z500)
clim_ind = apply_indian_subcontinent_bounding_box(clim_z500)

# Jan 1st is dayofyear 1, Feb 11 is dayofyear 42
clim_sliced = clim_ind.sel(dayofyear=slice(1, 42))

def get_weekly_anom(model_da, week_idx):
    start, end = week_idx * 7, (week_idx + 1) * 7
    m_week = model_da.isel(lead_time=slice(start, end)).mean('lead_time')
    c_week = clim_sliced.isel(dayofyear=slice(start, end)).mean('dayofyear')
    return m_week - c_week

models = {'FuXi': fuxi_ind, 'SPIRE': spire_ind, 'ECMWF': ecmwf_ind, 'NCEP': ncep_ind}

print("5. Plotting 24-Panel Figure...")
fig, axes = plt.subplots(4, 6, figsize=(24, 14), subplot_kw={'projection': ccrs.PlateCarree()})

for row, (name, da) in enumerate(models.items()):
    for w in range(6):
        ax = axes[row, w]
        anom = get_weekly_anom(da, w).squeeze()
        im = ax.pcolormesh(anom.longitude, anom.latitude, anom, cmap='PuOr_r', vmin=-100, vmax=100, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        
        if row == 0:
            ax.set_title(f"Week {w+1}", fontsize=14)
        if w == 0:
            ax.text(-0.15, 0.5, name, va='center', ha='center', rotation='vertical', 
                    transform=ax.transAxes, fontsize=16, fontweight='bold')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Z500 Anomaly (m)')

plt.suptitle("Multi-Model Z500 Anomalies (Init: Jan 1, 2026)", fontsize=22, y=0.95)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"SUCCESS! Figure saved to: {out_path}")
