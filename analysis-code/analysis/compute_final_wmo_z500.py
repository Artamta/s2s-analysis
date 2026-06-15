import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os
from scipy.stats import norm

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box, extract_imd_homogeneous_region
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias

print("Starting FINAL WMO-Compliant WEEKLY S2S Skill Horizon for Z500 (with CRPS and Regions)...")

def crps_gauss(mu, sig, y):
    """closed-form CRPS of a Gaussian(mu,sig) forecast vs scalar/array obs y."""
    sig = np.maximum(sig, 1e-6)
    w = (y - mu) / sig
    return sig * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))

def area_weighted_mean(da, w_da):
    """Calculate area-weighted mean over lat/lon dimensions."""
    return (da * w_da).mean(dim=['lat', 'lon']).values.item()

# 1. Load ERA5
era_z_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_pressure_500hpa.grib', engine='cfgrib')
era_z = era_z_ds['z'] / 9.80665 # m2/s2 -> m
era_z = apply_indian_subcontinent_bounding_box(era_z).rename({'latitude': 'lat', 'longitude': 'lon'})

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29', '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26', '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']

spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')

# Preload Operational
ecmwf_cf_dict = {}
ecmwf_pf_dict = {}
ncep_cf_dict = {}
ncep_pf_dict = {}

for init_date in init_dates:
    init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
    try:
        ec_cf = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ecmwf_cf_dict[init_date] = apply_indian_subcontinent_bounding_box(ec_cf['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass
    try:
        ec_pf = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ecmwf/data/pl_pf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ecmwf_pf_dict[init_date] = apply_indian_subcontinent_bounding_box(ec_pf['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass
    
    try:
        nc_cf = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/pl_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ncep_cf_dict[init_date] = apply_indian_subcontinent_bounding_box(nc_cf['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass
    try:
        nc_pf = xr.open_dataset(f'/storage/raj.ayush/s2s-forecast-data/ncep/data/pl_pf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'gh'})
        ncep_pf_dict[init_date] = apply_indian_subcontinent_bounding_box(nc_pf['gh']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)

regions = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21), ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]

era_z_clim = era_z.mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

results = []
regional_results = []

for week_idx, (week_name, day_start, day_end) in enumerate(weeks):
    print(f"Processing {week_name}...")
    f_mets = {'acc': [], 'rmse': [], 'crps': []}
    s_mets = {'acc': [], 'rmse': [], 'crps': []}
    e_mets = {'acc': [], 'rmse': [], 'crps': []}
    n_mets = {'acc': [], 'rmse': [], 'crps': []}
    
    for init_date in init_dates:
        dates = pd.date_range(start=init_date, periods=42)[day_start-1:day_end]
        valid_dates = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if len(valid_dates) == 0: continue
            
        try:
            e_z_week = era_z.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except: continue
            
        s_mean = f_mean = ec_mean = nc_mean = None
        s_std = f_std = ec_std = nc_std = None
        
        # Spire
        try:
            s_mean = spire.sel(reference_time=init_date)['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0).isel(step=slice(day_start-1, day_end)).mean('step')
            s_std = spire.sel(reference_time=init_date)['geopotential_height_at_isobaric_levels_stddev'].sel(isobar=50000.0).isel(step=slice(day_start-1, day_end)).mean('step')
            s_mean = apply_indian_subcontinent_bounding_box(s_mean).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            s_std = apply_indian_subcontinent_bounding_box(s_std).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except: pass
        
        # FuXi
        init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
        f_week_data = []
        for mem in range(11):
            mem_data = []
            for d in range(day_start, day_end + 1):
                f_path = f"/storage/raj.ayush/s2s-forecast-data/fuxi/output/{init_str}/member/{mem:02d}/{d:02d}.nc"
                if os.path.exists(f_path):
                    try:
                        ds = xr.open_dataset(f_path)['__xarray_dataarray_variable__'].isel(channel=5) / 9.80665
                        mem_data.append(ds)
                    except: pass
            if mem_data:
                mem_week_mean = xr.concat(mem_data, dim='time').mean('time')
                if 'lead_time' in mem_week_mean.dims: mem_week_mean = mem_week_mean.mean('lead_time')
                f_week_data.append(mem_week_mean)
                
        if f_week_data:
            f_ens = xr.concat(f_week_data, dim='member')
            f_mean = apply_indian_subcontinent_bounding_box(f_ens.mean('member').rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            f_std = apply_indian_subcontinent_bounding_box(f_ens.std('member', ddof=1).rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            
        # ECMWF
        if init_date in ecmwf_cf_dict and init_date in ecmwf_pf_dict:
            try:
                ec_mean = ecmwf_cf_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                ec_pf = ecmwf_pf_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step')
                ec_std = ec_pf.std('number', ddof=1).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            except: pass
            
        # NCEP
        if init_date in ncep_cf_dict and init_date in ncep_pf_dict:
            try:
                nc_mean = ncep_cf_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
                nc_pf = ncep_pf_dict[init_date].isel(step=slice(day_start-1, day_end)).mean('step')
                nc_std = nc_pf.std('number', ddof=1).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
            except: pass

        # Compute Metrics across all Regions
        for region_name in regions:
            if region_name != 'All India':
                e_z_eval = extract_imd_homogeneous_region(e_z_week.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'})
                clim_eval = extract_imd_homogeneous_region(era_z_clim.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'})
                
                s_eval = extract_imd_homogeneous_region(s_mean.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if s_mean is not None else None
                s_std_eval = extract_imd_homogeneous_region(s_std.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if s_std is not None else None
                
                f_eval = extract_imd_homogeneous_region(f_mean.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if f_mean is not None else None
                f_std_eval = extract_imd_homogeneous_region(f_std.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if f_std is not None else None
                
                ec_eval = extract_imd_homogeneous_region(ec_mean.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if ec_mean is not None else None
                ec_std_eval = extract_imd_homogeneous_region(ec_std.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if ec_std is not None else None
                
                nc_eval = extract_imd_homogeneous_region(nc_mean.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if nc_mean is not None else None
                nc_std_eval = extract_imd_homogeneous_region(nc_std.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if nc_std is not None else None
                
                w_da = get_cosine_latitude_weights(e_z_eval.lat.values)
            else:
                e_z_eval = e_z_week
                clim_eval = era_z_clim
                
                s_eval, s_std_eval = s_mean, s_std
                f_eval, f_std_eval = f_mean, f_std
                ec_eval, ec_std_eval = ec_mean, ec_std
                nc_eval, nc_std_eval = nc_mean, nc_std
                
                w_da = get_cosine_latitude_weights(target_lat)

            def track_mets(model_name, m_eval, m_std_eval, o_eval, clim):
                if m_eval is None: return
                try:
                    acc = calc_wmo_acc(m_eval, o_eval, clim, w_da)
                    rmse = calc_wmo_rmse(m_eval, o_eval, w_da)
                    bias = calc_wmo_bias(m_eval, o_eval, w_da)
                    
                    crps_val = np.nan
                    if m_std_eval is not None:
                        crps_da = crps_gauss(m_eval, m_std_eval, o_eval)
                        crps_val = area_weighted_mean(crps_da, w_da)
                        
                    regional_results.append({
                        'Region': region_name, 'Week': week_name, 'Init_Date': init_date, 'Model': model_name,
                        'ACC': acc, 'RMSE': rmse, 'Bias': bias, 'CRPS': crps_val
                    })
                    if region_name == 'All India':
                        if model_name == 'FuXi': f_mets['acc'].append(acc); f_mets['rmse'].append(rmse); f_mets['crps'].append(crps_val)
                        if model_name == 'Spire': s_mets['acc'].append(acc); s_mets['rmse'].append(rmse); s_mets['crps'].append(crps_val)
                        if model_name == 'ECMWF': e_mets['acc'].append(acc); e_mets['rmse'].append(rmse); e_mets['crps'].append(crps_val)
                        if model_name == 'NCEP': n_mets['acc'].append(acc); n_mets['rmse'].append(rmse); n_mets['crps'].append(crps_val)
                except Exception as e:
                    pass

            track_mets('FuXi', f_eval, f_std_eval, e_z_eval, clim_eval)
            track_mets('Spire', s_eval, s_std_eval, e_z_eval, clim_eval)
            track_mets('ECMWF', ec_eval, ec_std_eval, e_z_eval, clim_eval)
            track_mets('NCEP', nc_eval, nc_std_eval, e_z_eval, clim_eval)

    def get_stats(arr):
        arr = [a for a in arr if not np.isnan(a)]
        if len(arr) == 0: return np.nan, np.nan
        return np.mean(arr), 1.96 * np.std(arr) / np.sqrt(len(arr))

    results.append({
        'Week': week_idx + 1,
        'FuXi_ACC': get_stats(f_mets['acc'])[0], 'FuXi_ACC_CI': get_stats(f_mets['acc'])[1],
        'FuXi_RMSE': get_stats(f_mets['rmse'])[0], 'FuXi_RMSE_CI': get_stats(f_mets['rmse'])[1],
        'FuXi_CRPS': get_stats(f_mets['crps'])[0], 'FuXi_CRPS_CI': get_stats(f_mets['crps'])[1],
        
        'Spire_ACC': get_stats(s_mets['acc'])[0], 'Spire_ACC_CI': get_stats(s_mets['acc'])[1],
        'Spire_RMSE': get_stats(s_mets['rmse'])[0], 'Spire_RMSE_CI': get_stats(s_mets['rmse'])[1],
        'Spire_CRPS': get_stats(s_mets['crps'])[0], 'Spire_CRPS_CI': get_stats(s_mets['crps'])[1],
        
        'ECMWF_ACC': get_stats(e_mets['acc'])[0], 'ECMWF_ACC_CI': get_stats(e_mets['acc'])[1],
        'ECMWF_RMSE': get_stats(e_mets['rmse'])[0], 'ECMWF_RMSE_CI': get_stats(e_mets['rmse'])[1],
        'ECMWF_CRPS': get_stats(e_mets['crps'])[0], 'ECMWF_CRPS_CI': get_stats(e_mets['crps'])[1],
        
        'NCEP_ACC': get_stats(n_mets['acc'])[0], 'NCEP_ACC_CI': get_stats(n_mets['acc'])[1],
        'NCEP_RMSE': get_stats(n_mets['rmse'])[0], 'NCEP_RMSE_CI': get_stats(n_mets['rmse'])[1],
        'NCEP_CRPS': get_stats(n_mets['crps'])[0], 'NCEP_CRPS_CI': get_stats(n_mets['crps'])[1]
    })

pd.DataFrame(regional_results).to_csv('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/z500_regional_wmo_with_crps.csv', index=False)
df = pd.DataFrame(results)

# PLOTTING FULL DOMAIN HORIZON
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), dpi=300)
x = df['Week']
colors = {'FuXi': '#0072B2', 'Spire': '#D55E00', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}

for ax in [ax1, ax2, ax3]:
    for i in range(1, 7):
        if i % 2 == 0: ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.1)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 7)], fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

ax1.plot(x, df['FuXi_ACC'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o')
ax1.plot(x, df['Spire_ACC'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s')
ax1.plot(x, df['ECMWF_ACC'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^')
ax1.plot(x, df['NCEP_ACC'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d')
ax1.axhline(0.6, color='black', linestyle='-', linewidth=2, label='Predictability Limit')
ax1.set_ylabel('Anomaly Pattern Correlation (APCC)', fontsize=13, fontweight='bold')
ax1.set_title('(a) Z500 ACC', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left', fontsize=11)

ax2.plot(x, df['FuXi_RMSE'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o')
ax2.plot(x, df['Spire_RMSE'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s')
ax2.plot(x, df['ECMWF_RMSE'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^')
ax2.plot(x, df['NCEP_RMSE'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d')
ax2.set_ylabel('Root Mean Square Error (m)', fontsize=13, fontweight='bold')
ax2.set_title('(b) Z500 RMSE', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)

ax3.plot(x, df['FuXi_CRPS'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o')
ax3.plot(x, df['Spire_CRPS'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s')
ax3.plot(x, df['ECMWF_CRPS'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^')
ax3.plot(x, df['NCEP_CRPS'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d')
ax3.set_ylabel('Continuous Ranked Probability Score', fontsize=13, fontweight='bold')
ax3.set_title('(c) Z500 CRPS (Lower is Better)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=11)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/skill_horizon_z500_wmo_with_crps.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Z500 WMO Horizon with CRPS saved to {out_path}")
