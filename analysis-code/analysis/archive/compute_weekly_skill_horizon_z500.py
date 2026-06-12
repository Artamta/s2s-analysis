import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os
import scipy.stats as stats

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box

print("Starting WEEKLY S2S Skill Horizon Computation for Z500...")

# 1. Load ERA5
era_z_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_pressure_500hpa.grib', engine='cfgrib')
era_z = era_z_ds['z'] / 9.80665 # m2/s2 -> m
era_z = apply_indian_subcontinent_bounding_box(era_z).rename({'latitude': 'lat', 'longitude': 'lon'})

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29', '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26', '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']

spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')

# Preload Operational
ecmwf_dict = {}
ncep_dict = {}
for init_date in init_dates:
    init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
    try:
        ec_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ecmwf_dict[init_date] = apply_indian_subcontinent_bounding_box(ec_ds['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass
    try:
        nc_ds = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ncep_dict[init_date] = apply_indian_subcontinent_bounding_box(nc_ds['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)

weeks = [
    ('Week 1', 1, 7),
    ('Week 2', 8, 14),
    ('Week 3', 15, 21),
    ('Week 4', 22, 28),
    ('Week 5', 29, 35),
    ('Week 6', 36, 42)
]

era_z_clim = era_z.mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

def calc_acc(f, o, clim):
    f_anom = f - clim
    f_anom = f_anom - f_anom.mean()
    o_anom = o - clim
    o_anom = o_anom - o_anom.mean()
    num = (f_anom * o_anom).mean()
    den = np.sqrt((f_anom**2).mean() * (o_anom**2).mean())
    return float((num / den).values)

def calc_rmse(f, o):
    return float(np.sqrt(((f - o)**2).mean()).values)

results = []

for week_idx, (week_name, day_start, day_end) in enumerate(weeks):
    print(f"Processing {week_name}...")
    f_acc, f_rmse = [], []
    s_acc, s_rmse = [], []
    e_acc, e_rmse = [], []
    n_acc, n_rmse = [], []
    
    for init_date in init_dates:
        dates = pd.date_range(start=init_date, periods=42)[day_start-1:day_end]
        valid_dates = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if len(valid_dates) == 0:
            continue
            
        try:
            e_z_week = era_z.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except:
            continue
            
        # Spire
        try:
            s_d = spire.sel(reference_time=init_date)['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0).isel(step=slice(day_start-1, day_end)).mean('step')
            s_d = apply_indian_subcontinent_bounding_box(s_d).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            s_acc.append(calc_acc(s_d, e_z_week, era_z_clim))
            s_rmse.append(calc_rmse(s_d, e_z_week))
        except: pass
        
        # FuXi
        init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
        f_week_data = []
        for d in range(day_start, day_end + 1):
            f_path = f"/storage/raj.ayush/s2s-forecast-data/fuxi/output/{init_str}/member/00/{d:02d}.nc"
            if os.path.exists(f_path):
                try:
                    ds = xr.open_dataset(f_path)['__xarray_dataarray_variable__'].isel(channel=5) / 9.80665
                    f_week_data.append(ds)
                except: pass
        if f_week_data:
            f_d = xr.concat(f_week_data, dim='time').mean('time')
            f_d = apply_indian_subcontinent_bounding_box(f_d.rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            f_acc.append(calc_acc(f_d, e_z_week, era_z_clim))
            f_rmse.append(calc_rmse(f_d, e_z_week))
            
        # ECMWF
        if init_date in ecmwf_dict:
            try:
                ec_d = ecmwf_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                e_acc.append(calc_acc(ec_d, e_z_week, era_z_clim))
                e_rmse.append(calc_rmse(ec_d, e_z_week))
            except: pass
            
        # NCEP
        if init_date in ncep_dict:
            try:
                nc_d = ncep_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                n_acc.append(calc_acc(nc_d, e_z_week, era_z_clim))
                n_rmse.append(calc_rmse(nc_d, e_z_week))
            except: pass

    def get_stats(arr):
        if len(arr) == 0: return np.nan, np.nan
        return np.mean(arr), 1.96 * np.std(arr) / np.sqrt(len(arr))

    f_m_acc, f_ci_acc = get_stats(f_acc)
    f_m_rmse, f_ci_rmse = get_stats(f_rmse)
    
    s_m_acc, s_ci_acc = get_stats(s_acc)
    s_m_rmse, s_ci_rmse = get_stats(s_rmse)
    
    e_m_acc, e_ci_acc = get_stats(e_acc)
    e_m_rmse, e_ci_rmse = get_stats(e_rmse)
    
    n_m_acc, n_ci_acc = get_stats(n_acc)
    n_m_rmse, n_ci_rmse = get_stats(n_rmse)

    results.append({
        'Week': week_idx + 1,
        'FuXi_ACC': f_m_acc, 'FuXi_ACC_CI': f_ci_acc,
        'FuXi_RMSE': f_m_rmse, 'FuXi_RMSE_CI': f_ci_rmse,
        'Spire_ACC': s_m_acc, 'Spire_ACC_CI': s_ci_acc,
        'Spire_RMSE': s_m_rmse, 'Spire_RMSE_CI': s_ci_rmse,
        'ECMWF_ACC': e_m_acc, 'ECMWF_ACC_CI': e_ci_acc,
        'ECMWF_RMSE': e_m_rmse, 'ECMWF_RMSE_CI': e_ci_rmse,
        'NCEP_ACC': n_m_acc, 'NCEP_ACC_CI': n_ci_acc,
        'NCEP_RMSE': n_m_rmse, 'NCEP_RMSE_CI': n_ci_rmse
    })

df = pd.DataFrame(results)

# PLOTTING
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

x = df['Week']

colors = {'FuXi': '#0072B2', 'Spire': '#D55E00', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}

for ax in [ax1, ax2]:
    # Vertical shading for weeks
    for i in range(1, 7):
        if i % 2 == 0:
            ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.1)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 7)], fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

# ACC
ax1.plot(x, df['FuXi_ACC'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o', markersize=8)
ax1.fill_between(x, df['FuXi_ACC'] - df['FuXi_ACC_CI'], df['FuXi_ACC'] + df['FuXi_ACC_CI'], color=colors['FuXi'], alpha=0.2)

ax1.plot(x, df['Spire_ACC'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s', markersize=8)
ax1.fill_between(x, df['Spire_ACC'] - df['Spire_ACC_CI'], df['Spire_ACC'] + df['Spire_ACC_CI'], color=colors['Spire'], alpha=0.2)

ax1.plot(x, df['ECMWF_ACC'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^', markersize=8)
ax1.fill_between(x, df['ECMWF_ACC'] - df['ECMWF_ACC_CI'], df['ECMWF_ACC'] + df['ECMWF_ACC_CI'], color=colors['ECMWF'], alpha=0.1)

ax1.plot(x, df['NCEP_ACC'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d', markersize=8)
ax1.fill_between(x, df['NCEP_ACC'] - df['NCEP_ACC_CI'], df['NCEP_ACC'] + df['NCEP_ACC_CI'], color=colors['NCEP'], alpha=0.1)

ax1.axhline(0.6, color='black', linestyle='-', linewidth=2, label='Predictability Limit (ACC=0.6)')
ax1.set_ylabel('Anomaly Correlation Coefficient (ACC)', fontsize=13, fontweight='bold')
ax1.set_title('(a) Z500 Weekly Skill Horizon', fontsize=16, fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.legend(loc='lower left', fontsize=11)

# RMSE
ax2.plot(x, df['FuXi_RMSE'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o', markersize=8)
ax2.fill_between(x, df['FuXi_RMSE'] - df['FuXi_RMSE_CI'], df['FuXi_RMSE'] + df['FuXi_RMSE_CI'], color=colors['FuXi'], alpha=0.2)

ax2.plot(x, df['Spire_RMSE'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s', markersize=8)
ax2.fill_between(x, df['Spire_RMSE'] - df['Spire_RMSE_CI'], df['Spire_RMSE'] + df['Spire_RMSE_CI'], color=colors['Spire'], alpha=0.2)

ax2.plot(x, df['ECMWF_RMSE'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^', markersize=8)
ax2.fill_between(x, df['ECMWF_RMSE'] - df['ECMWF_RMSE_CI'], df['ECMWF_RMSE'] + df['ECMWF_RMSE_CI'], color=colors['ECMWF'], alpha=0.1)

ax2.plot(x, df['NCEP_RMSE'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d', markersize=8)
ax2.fill_between(x, df['NCEP_RMSE'] - df['NCEP_RMSE_CI'], df['NCEP_RMSE'] + df['NCEP_RMSE_CI'], color=colors['NCEP'], alpha=0.1)

ax2.set_ylabel('Root Mean Square Error (m)', fontsize=13, fontweight='bold')
ax2.set_title('(b) Z500 Weekly RMSE Growth', fontsize=16, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/skill_horizon_z500_blueprint.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Blueprint Horizon saved to {out_path}")
