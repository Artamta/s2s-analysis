import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box

print("Starting DAILY S2S Skill Horizon Computation for Z500...")

# 1. Load ERA5 (Convert z to geopotential height in meters)
era_z_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_pressure_500hpa.grib', engine='cfgrib')
era_z = era_z_ds['z'] / 9.80665 # m2/s2 -> m
era_z = apply_indian_subcontinent_bounding_box(era_z).rename({'latitude': 'lat', 'longitude': 'lon'})

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29', '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26', '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']

spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')

# Preload Operational Datasets to avoid repeatedly opening GRIB files
ecmwf_dict = {}
ncep_dict = {}
for init_date in init_dates:
    init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
    try:
        ec_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ecmwf_dict[init_date] = apply_indian_subcontinent_bounding_box(ec_ds['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except Exception as e:
        pass
    try:
        nc_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ncep_dict[init_date] = apply_indian_subcontinent_bounding_box(nc_ds['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except Exception as e:
        pass

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)

results = []

def calc_acc(f, o):
    f_anom = f - f.mean()
    o_anom = o - o.mean()
    num = (f_anom * o_anom).mean()
    den = np.sqrt((f_anom**2).mean() * (o_anom**2).mean())
    return float((num / den).values)

def calc_rmse(f, o):
    return float(np.sqrt(((f - o)**2).mean()).values)

for ld in range(1, 43): # Days 1 to 42
    print(f"Processing Lead Day {ld}...")
    f_acc, f_rmse = [], []
    s_acc, s_rmse = [], []
    e_acc, e_rmse = [], []
    n_acc, n_rmse = [], []
    
    for init_date in init_dates:
        target_date = pd.to_datetime(init_date) + pd.Timedelta(days=ld-1)
        target_str = target_date.strftime('%Y-%m-%d')
        if target_str > '2026-05-15':
            continue
            
        try:
            e_z_target = era_z.sel(time=target_str).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except:
            continue
        
        # Spire
        try:
            s_d = spire.sel(reference_time=init_date)['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0).isel(step=ld-1)
            s_d = apply_indian_subcontinent_bounding_box(s_d).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            s_acc.append(calc_acc(s_d, e_z_target))
            s_rmse.append(calc_rmse(s_d, e_z_target))
        except: pass
        
        # FuXi (member 00)
        init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
        try:
            f_path = f"/storage/raj.ayush/s2s-forecast-data/fuxi/output/{init_str}/member/00/{ld:02d}.nc"
            if os.path.exists(f_path):
                f_d = xr.open_dataset(f_path)['__xarray_dataarray_variable__'].isel(channel=5) / 9.80665
                f_d = apply_indian_subcontinent_bounding_box(f_d.rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                f_acc.append(calc_acc(f_d, e_z_target))
                f_rmse.append(calc_rmse(f_d, e_z_target))
        except: pass

        # ECMWF
        if init_date in ecmwf_dict:
            try:
                ec_d = ecmwf_dict[init_date].isel(step=ld-1).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                e_acc.append(calc_acc(ec_d, e_z_target))
                e_rmse.append(calc_rmse(ec_d, e_z_target))
            except: pass
            
        # NCEP
        if init_date in ncep_dict:
            try:
                nc_d = ncep_dict[init_date].isel(step=ld-1).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                n_acc.append(calc_acc(nc_d, e_z_target))
                n_rmse.append(calc_rmse(nc_d, e_z_target))
            except: pass

    results.append({
        'Lead Day': ld,
        'FuXi_ACC': np.mean(f_acc) if f_acc else np.nan,
        'FuXi_RMSE': np.mean(f_rmse) if f_rmse else np.nan,
        'Spire_ACC': np.mean(s_acc) if s_acc else np.nan,
        'Spire_RMSE': np.mean(s_rmse) if s_rmse else np.nan,
        'ECMWF_ACC': np.mean(e_acc) if e_acc else np.nan,
        'ECMWF_RMSE': np.mean(e_rmse) if e_rmse else np.nan,
        'NCEP_ACC': np.mean(n_acc) if n_acc else np.nan,
        'NCEP_RMSE': np.mean(n_rmse) if n_rmse else np.nan
    })

df = pd.DataFrame(results)
df.to_csv('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/daily_skill_horizon_z500.csv', index=False)

# PLOTTING
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

x = df['Lead Day']

# ACC
ax1.plot(x, df['FuXi_ACC'], label='FuXi-S2S', color='#0072B2', linewidth=3)
ax1.plot(x, df['Spire_ACC'], label='SPIRE', color='#D55E00', linewidth=3)
ax1.plot(x, df['ECMWF_ACC'], label='ECMWF (Operational)', color='black', linewidth=2, linestyle='--')
ax1.plot(x, df['NCEP_ACC'], label='NCEP (Operational)', color='gray', linewidth=2, linestyle='--')

ax1.axhline(0.5, color='red', linestyle=':', linewidth=2, label='Predictability Limit (ACC=0.5)')
ax1.set_ylabel('Pattern Correlation Coefficient (ACC)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Lead Time (Days)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Z500 Spatial ACC Horizon', fontsize=14, fontweight='bold')
ax1.set_ylim(0.0, 1.0)
ax1.set_xlim(1, 42)
ax1.legend()
ax1.grid(True, alpha=0.5)

# RMSE
ax2.plot(x, df['FuXi_RMSE'], label='FuXi-S2S', color='#0072B2', linewidth=3)
ax2.plot(x, df['Spire_RMSE'], label='SPIRE', color='#D55E00', linewidth=3)
ax2.plot(x, df['ECMWF_RMSE'], label='ECMWF (Operational)', color='black', linewidth=2, linestyle='--')
ax2.plot(x, df['NCEP_RMSE'], label='NCEP (Operational)', color='gray', linewidth=2, linestyle='--')

ax2.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Lead Time (Days)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Z500 Spatial RMSE Growth', fontsize=14, fontweight='bold')
ax2.set_xlim(1, 42)
ax2.legend()
ax2.grid(True, alpha=0.5)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/skill_horizon_z500_daily.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Daily Horizon saved to {out_path}")
