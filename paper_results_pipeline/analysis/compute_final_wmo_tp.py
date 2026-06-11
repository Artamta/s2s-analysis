import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box, extract_imd_homogeneous_region
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias

print("Starting FINAL WMO-Compliant WEEKLY S2S Skill Horizon for TOTAL PRECIPITATION (TP)...")

# 1. Load ERA5 TP
era_tp_ds = xr.open_dataset('/storage/raj.ayush/benchmark(jfm)/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
era_tp = era_tp_ds['tp'] * 1000.0 # meters -> mm
era_tp = apply_indian_subcontinent_bounding_box(era_tp).rename({'latitude': 'lat', 'longitude': 'lon'})

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29', '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26', '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']

spire = xr.open_zarr('/storage/raj.ayush/spire-hindecast-weekely-initialized/spire_hindcast_jfm.zarr', group='mean_stddev')

# Preload Operational Datasets
ecmwf_dict = {}
ncep_dict = {}
for init_date in init_dates:
    init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
    try:
        ec_ds = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ecmwf/data/sfc_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
        ecmwf_dict[init_date] = apply_indian_subcontinent_bounding_box(ec_ds['tp']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass
    try:
        nc_ds = xr.open_dataset(f'/storage/raj.ayush/benchmark(jfm)/ncep/data/sfc_cf_{init_str}.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
        ncep_dict[init_date] = apply_indian_subcontinent_bounding_box(nc_ds['tp']).rename({'latitude': 'lat', 'longitude': 'lon'})
    except: pass

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)

regions = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']

weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21), ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]

# Compute Climatology for Anomaly Pattern Correlation
era_tp_clim = era_tp.mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

results = []
regional_results = []

for week_idx, (week_name, day_start, day_end) in enumerate(weeks):
    print(f"Processing {week_name}...")
    
    # Dictionaries to hold 13-init values for full domain
    f_mets = {'acc': [], 'rmse': []}
    s_mets = {'acc': [], 'rmse': []}
    e_mets = {'acc': [], 'rmse': []}
    n_mets = {'acc': [], 'rmse': []}
    
    for init_date in init_dates:
        dates = pd.date_range(start=init_date, periods=42)[day_start-1:day_end]
        valid_dates = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if len(valid_dates) == 0: continue
            
        try:
            # Daily ERA5 is averaged over the week
            e_tp_week = era_tp.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except: continue
        
        # Helper to compute daily accumulation and weekly mean
        def get_weekly_mean_from_cumulative(cum_da, d_start, d_end):
            # cum_da is accumulated from day 0. We need (cum(day_end) - cum(day_start-1)) / days
            if d_start == 1:
                val = cum_da.isel(step=d_end-1) / (d_end - d_start + 1)
            else:
                val = (cum_da.isel(step=d_end-1) - cum_da.isel(step=d_start-2)) / (d_end - d_start + 1)
            return val.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()

        s_d = f_d = ec_d = nc_d = None
        
        # Spire
        try:
            s_tp = spire.sel(reference_time=init_date)['precipitation_amount']
            s_tp = apply_indian_subcontinent_bounding_box(s_tp).rename({'latitude': 'lat', 'longitude': 'lon'})
            s_d = get_weekly_mean_from_cumulative(s_tp, day_start, day_end)
        except: pass
        
        # FuXi
        init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
        try:
            path_end = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_str}/member/00/{day_end:02d}.nc"
            path_start = f"/storage/raj.ayush/fuxi-init-jfm-weekely/output/{init_str}/member/00/{day_start-1:02d}.nc"
            if os.path.exists(path_end):
                d_end_val = xr.open_dataset(path_end)['__xarray_dataarray_variable__'].sel(channel='tp')
                if 'lead_time' in d_end_val.dims: d_end_val = d_end_val.mean('lead_time')
                if 'member' in d_end_val.dims: d_end_val = d_end_val.mean('member')
                
                if day_start == 1:
                    f_val = d_end_val / (day_end - day_start + 1)
                elif os.path.exists(path_start):
                    d_start_val = xr.open_dataset(path_start)['__xarray_dataarray_variable__'].sel(channel='tp')
                    if 'lead_time' in d_start_val.dims: d_start_val = d_start_val.mean('lead_time')
                    if 'member' in d_start_val.dims: d_start_val = d_start_val.mean('member')
                    f_val = (d_end_val - d_start_val) / (day_end - day_start + 1)
                else:
                    f_val = None
                    
                if f_val is not None:
                    f_d = apply_indian_subcontinent_bounding_box(f_val.rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except: pass
            
        # ECMWF
        if init_date in ecmwf_dict:
            try:
                ec_tp = ecmwf_dict[init_date]
                ec_d = get_weekly_mean_from_cumulative(ec_tp, day_start, day_end)
            except: pass
            
        # NCEP
        if init_date in ncep_dict:
            try:
                nc_tp = ncep_dict[init_date]
                nc_d = get_weekly_mean_from_cumulative(nc_tp, day_start, day_end)
            except: pass

        # Compute Metrics across all Regions
        for region_name in regions:
            if region_name != 'All India':
                e_z_eval = extract_imd_homogeneous_region(e_tp_week.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'})
                clim_eval = extract_imd_homogeneous_region(era_tp_clim.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'})
                s_eval = extract_imd_homogeneous_region(s_d.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if s_d is not None else None
                f_eval = extract_imd_homogeneous_region(f_d.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if f_d is not None else None
                ec_eval = extract_imd_homogeneous_region(ec_d.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if ec_d is not None else None
                nc_eval = extract_imd_homogeneous_region(nc_d.rename({'lat':'latitude','lon':'longitude'}), region_name).rename({'latitude':'lat','longitude':'lon'}) if nc_d is not None else None
                w_da = get_cosine_latitude_weights(e_z_eval.lat.values)
            else:
                e_z_eval = e_tp_week
                clim_eval = era_tp_clim
                s_eval, f_eval, ec_eval, nc_eval = s_d, f_d, ec_d, nc_d
                w_da = get_cosine_latitude_weights(target_lat)

            def track_mets(model_name, m_eval, o_eval, clim):
                if m_eval is None: return
                try:
                    acc = calc_wmo_acc(m_eval, o_eval, clim, w_da)
                    rmse = calc_wmo_rmse(m_eval, o_eval, w_da)
                    bias = calc_wmo_bias(m_eval, o_eval, w_da)
                    regional_results.append({
                        'Region': region_name, 'Week': week_name, 'Init_Date': init_date, 'Model': model_name,
                        'ACC': acc, 'RMSE': rmse, 'Bias': bias
                    })
                    if region_name == 'All India':
                        if model_name == 'FuXi': f_mets['acc'].append(acc); f_mets['rmse'].append(rmse)
                        if model_name == 'Spire': s_mets['acc'].append(acc); s_mets['rmse'].append(rmse)
                        if model_name == 'ECMWF': e_mets['acc'].append(acc); e_mets['rmse'].append(rmse)
                        if model_name == 'NCEP': n_mets['acc'].append(acc); n_mets['rmse'].append(rmse)
                except Exception as e:
                    pass

            track_mets('FuXi', f_eval, e_z_eval, clim_eval)
            track_mets('Spire', s_eval, e_z_eval, clim_eval)
            track_mets('ECMWF', ec_eval, e_z_eval, clim_eval)
            track_mets('NCEP', nc_eval, e_z_eval, clim_eval)

    # Full domain aggregated stats
    def get_stats(arr):
        if len(arr) == 0: return np.nan, np.nan
        return np.mean(arr), 1.96 * np.std(arr) / np.sqrt(len(arr))

    results.append({
        'Week': week_idx + 1,
        'FuXi_ACC': get_stats(f_mets['acc'])[0], 'FuXi_ACC_CI': get_stats(f_mets['acc'])[1],
        'FuXi_RMSE': get_stats(f_mets['rmse'])[0], 'FuXi_RMSE_CI': get_stats(f_mets['rmse'])[1],
        'Spire_ACC': get_stats(s_mets['acc'])[0], 'Spire_ACC_CI': get_stats(s_mets['acc'])[1],
        'Spire_RMSE': get_stats(s_mets['rmse'])[0], 'Spire_RMSE_CI': get_stats(s_mets['rmse'])[1],
        'ECMWF_ACC': get_stats(e_mets['acc'])[0], 'ECMWF_ACC_CI': get_stats(e_mets['acc'])[1],
        'ECMWF_RMSE': get_stats(e_mets['rmse'])[0], 'ECMWF_RMSE_CI': get_stats(e_mets['rmse'])[1],
        'NCEP_ACC': get_stats(n_mets['acc'])[0], 'NCEP_ACC_CI': get_stats(n_mets['acc'])[1],
        'NCEP_RMSE': get_stats(n_mets['rmse'])[0], 'NCEP_RMSE_CI': get_stats(n_mets['rmse'])[1]
    })

# Save regional results for later
pd.DataFrame(regional_results).to_csv('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/tp_regional_wmo.csv', index=False)

df = pd.DataFrame(results)

# PLOTTING FULL DOMAIN HORIZON
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
x = df['Week']
colors = {'FuXi': '#0072B2', 'Spire': '#D55E00', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}

for ax in [ax1, ax2]:
    for i in range(1, 7):
        if i % 2 == 0: ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.1)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 7)], fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

ax1.plot(x, df['FuXi_ACC'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o')
ax1.fill_between(x, df['FuXi_ACC'] - df['FuXi_ACC_CI'], df['FuXi_ACC'] + df['FuXi_ACC_CI'], color=colors['FuXi'], alpha=0.2)
ax1.plot(x, df['Spire_ACC'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s')
ax1.fill_between(x, df['Spire_ACC'] - df['Spire_ACC_CI'], df['Spire_ACC'] + df['Spire_ACC_CI'], color=colors['Spire'], alpha=0.2)
ax1.plot(x, df['ECMWF_ACC'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^')
ax1.fill_between(x, df['ECMWF_ACC'] - df['ECMWF_ACC_CI'], df['ECMWF_ACC'] + df['ECMWF_ACC_CI'], color=colors['ECMWF'], alpha=0.1)
ax1.plot(x, df['NCEP_ACC'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d')
ax1.fill_between(x, df['NCEP_ACC'] - df['NCEP_ACC_CI'], df['NCEP_ACC'] + df['NCEP_ACC_CI'], color=colors['NCEP'], alpha=0.1)

ax1.axhline(0.6, color='black', linestyle='-', linewidth=2, label='Predictability Limit (ACC=0.6)')
ax1.set_ylabel('Anomaly Pattern Correlation (APCC)', fontsize=13, fontweight='bold')
ax1.set_title('(a) S2S Weekly Horizon: Total Precipitation (TP) ACC\n[WMO-Compliant Cosine Latitude Weighted APCC]', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower left', fontsize=11)

ax2.plot(x, df['FuXi_RMSE'], label='FuXi-S2S', color=colors['FuXi'], linewidth=3, marker='o')
ax2.fill_between(x, df['FuXi_RMSE'] - df['FuXi_RMSE_CI'], df['FuXi_RMSE'] + df['FuXi_RMSE_CI'], color=colors['FuXi'], alpha=0.2)
ax2.plot(x, df['Spire_RMSE'], label='SPIRE', color=colors['Spire'], linewidth=3, marker='s')
ax2.fill_between(x, df['Spire_RMSE'] - df['Spire_RMSE_CI'], df['Spire_RMSE'] + df['Spire_RMSE_CI'], color=colors['Spire'], alpha=0.2)
ax2.plot(x, df['ECMWF_RMSE'], label='ECMWF (Op)', color=colors['ECMWF'], linewidth=2, linestyle='--', marker='^')
ax2.fill_between(x, df['ECMWF_RMSE'] - df['ECMWF_RMSE_CI'], df['ECMWF_RMSE'] + df['ECMWF_RMSE_CI'], color=colors['ECMWF'], alpha=0.1)
ax2.plot(x, df['NCEP_RMSE'], label='NCEP (Op)', color=colors['NCEP'], linewidth=2, linestyle='-.', marker='d')
ax2.fill_between(x, df['NCEP_RMSE'] - df['NCEP_RMSE_CI'], df['NCEP_RMSE'] + df['NCEP_RMSE_CI'], color=colors['NCEP'], alpha=0.1)

ax2.set_ylabel('Root Mean Square Error (mm/day)', fontsize=13, fontweight='bold')
ax2.set_title('(b) S2S Weekly Horizon: Total Precipitation (TP) RMSE\n[WMO-Compliant Area-Weighted RMSE]', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/figures/verification/skill_horizon_tp_wmo.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! TP WMO Horizon saved to {out_path}")
