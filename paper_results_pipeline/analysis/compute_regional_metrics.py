import numpy as np
import xarray as xr
import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.spatial_masking import extract_imd_homogeneous_region

print("Starting Regional Metrics Calculation...")

# 1. Load ERA5 Ground Truth
era5_ds = xr.open_dataset('/storage/raj.ayush/benchmark(jfm)/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': '2t'})
era5_t2m = era5_ds['t2m'] - 273.15
era5_t2m = era5_t2m.rename({'latitude': 'lat', 'longitude': 'lon'})
era5_week2 = era5_t2m.sel(time=slice('2026-01-08', '2026-01-14')).mean('time')

# 2. Load FuXi
init_date = '20260101'
fuxi_base = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_date}/member"
fuxi_list = [xr.open_mfdataset([f"{fuxi_base}/{m:02d}/{day:02d}.nc" for day in range(8, 15)], combine='nested', concat_dim='lead_time') for m in range(11)]
fuxi = xr.concat(fuxi_list, dim='member').mean('member')
fuxi_t2m = fuxi['__xarray_dataarray_variable__'].isel(channel=66) - 273.15
fuxi_t2m = fuxi_t2m.rename({'lat': 'latitude', 'lon': 'longitude'})
fuxi_week2 = fuxi_t2m.mean('lead_time').rename({'latitude': 'lat', 'longitude': 'lon'})

# 3. Load Spire
spire = xr.open_zarr('/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr', group='mean_stddev')
spire_date = spire.sel(reference_time='2026-01-01')
spire_t2m = spire_date['air_temperature'].isel(step=slice(7, 14)) - 273.15
spire_week2 = spire_t2m.mean('step').rename({'latitude': 'lat', 'longitude': 'lon'})

# 4. Regrid to Common Grid (1.5 deg)
target_lat = np.arange(40, 5, -1.5)
target_lon = np.arange(60, 100, 1.5)

fuxi_regrid = fuxi_week2.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
spire_regrid = spire_week2.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
era5_regrid = era5_week2.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

# 5. Calculate Metrics for IMD Regions
regions = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
results = []

def calc_metrics(fcst, obs):
    bias = (fcst - obs).mean().values.item()
    rmse = np.sqrt(((fcst - obs)**2).mean()).values.item()
    return bias, rmse

for region in regions:
    # Rename lat/lon to latitude/longitude for the spatial_masking function
    f_reg = extract_imd_homogeneous_region(fuxi_regrid.rename({'lat': 'latitude', 'lon': 'longitude'}), region)
    s_reg = extract_imd_homogeneous_region(spire_regrid.rename({'lat': 'latitude', 'lon': 'longitude'}), region)
    e_reg = extract_imd_homogeneous_region(era5_regrid.rename({'lat': 'latitude', 'lon': 'longitude'}), region)
    
    # Rename back to compute metrics
    f_reg = f_reg.rename({'latitude': 'lat', 'longitude': 'lon'})
    s_reg = s_reg.rename({'latitude': 'lat', 'longitude': 'lon'})
    e_reg = e_reg.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    f_bias, f_rmse = calc_metrics(f_reg, e_reg)
    s_bias, s_rmse = calc_metrics(s_reg, e_reg)
    
    results.append({
        'IMD Region': region.replace('_', ' ').title(),
        'FuXi Mean Bias (°C)': round(f_bias, 3),
        'FuXi RMSE (°C)': round(f_rmse, 3),
        'Spire Mean Bias (°C)': round(s_bias, 3),
        'Spire RMSE (°C)': round(s_rmse, 3)
    })

df = pd.DataFrame(results)
print("\n" + "="*80)
print("WEEK 2 (JAN 8-14) T2M SPATIAL METRICS BY IMD REGION")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Save to CSV
out_csv = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/regional_metrics_week2.csv'
df.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}")
