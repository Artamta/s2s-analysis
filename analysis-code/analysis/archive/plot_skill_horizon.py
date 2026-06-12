import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.spatial_masking import apply_indian_subcontinent_bounding_box

print("Starting S2S Skill Horizon Computation (Z500 and TP)...")

# 1. Load ERA5
era_z_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_pressure_500hpa.grib', engine='cfgrib')
era_z = apply_indian_subcontinent_bounding_box(era_z_ds['z']).rename({'latitude': 'lat', 'longitude': 'lon'})
era_tp_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/data/era5_surface.grib', engine='cfgrib', filter_by_keys={'shortName': 'tp'})
era_tp = apply_indian_subcontinent_bounding_box(era_tp_ds['tp']).rename({'latitude': 'lat', 'longitude': 'lon'})

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22'] # Testing on first 4 dates for speed

weeks = [
    ('Week 1', 1, 7),
    ('Week 2', 8, 14),
    ('Week 3', 15, 21),
    ('Week 4', 22, 28)
]

spire = xr.open_zarr('/storage/raj.ayush/s2s-forecast-data/spire/spire_hindcast_jfm.zarr', group='mean_stddev')

results = []

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)

for week_name, day_start, day_end in weeks:
    print(f"Processing {week_name}...")
    fuxi_acc_z, fuxi_rmse_z = [], []
    spire_acc_z, spire_rmse_z = [], []
    
    for init_date in init_dates:
        # Get exact dates for the week
        dates = pd.date_range(start=init_date, periods=42)[day_start-1:day_end]
        valid_dates = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if len(valid_dates) < (day_end - day_start + 1):
            continue
            
        # ERA5
        e_z = era_z.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        e_tp = era_tp.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time').interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        
        # Spire
        s_date = spire.sel(reference_time=init_date)
        s_z = s_date['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0).isel(step=slice(day_start-1, day_end)).mean('step')
        s_z = apply_indian_subcontinent_bounding_box(s_z).rename({'latitude': 'lat', 'longitude': 'lon'}).interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        
        # FuXi
        init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
        try:
            fuxi_base = f"/storage/raj.ayush/s2s-forecast-data/fuxi/output/{init_str}/member"
            f_list = [xr.open_mfdataset([f"{fuxi_base}/{m:02d}/{day:02d}.nc" for day in range(day_start, day_end+1)], combine='nested', concat_dim='lead_time') for m in range(2)] # Just 2 members for speed
            fuxi_concat = xr.concat(f_list, dim='member').mean('member').mean('lead_time')
            f_z = fuxi_concat['__xarray_dataarray_variable__'].isel(channel=5)
            f_z = apply_indian_subcontinent_bounding_box(f_z.rename({'lat': 'latitude', 'lon': 'longitude'})).rename({'latitude': 'lat', 'longitude': 'lon'})
            f_z = f_z.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
        except:
            continue
            
        # Metrics
        def calc_acc(f, o):
            f_anom = f - f.mean()
            o_anom = o - o.mean()
            num = (f_anom * o_anom).mean()
            den = np.sqrt((f_anom**2).mean() * (o_anom**2).mean())
            return (num / den).values.item()
            
        def calc_rmse(f, o):
            return np.sqrt(((f - o)**2).mean()).values.item()
            
        fuxi_acc_z.append(calc_acc(f_z, e_z))
        fuxi_rmse_z.append(calc_rmse(f_z, e_z))
        spire_acc_z.append(calc_acc(s_z, e_z))
        spire_rmse_z.append(calc_rmse(s_z, e_z))

    results.append({
        'Week': week_name,
        'FuXi_Z500_ACC': np.mean(fuxi_acc_z),
        'FuXi_Z500_RMSE': np.mean(fuxi_rmse_z),
        'Spire_Z500_ACC': np.mean(spire_acc_z),
        'Spire_Z500_RMSE': np.mean(spire_rmse_z)
    })

df = pd.DataFrame(results)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

weeks = df['Week']

# ACC Plot
ax1.plot(weeks, df['FuXi_Z500_ACC'], marker='o', linewidth=3, label='FuXi-S2S', color='#0072B2')
ax1.plot(weeks, df['Spire_Z500_ACC'], marker='s', linewidth=3, label='Spire S2S', color='#D55E00')
ax1.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Predictability Limit (ACC=0.5)')
ax1.set_ylabel('Pattern Correlation (ACC)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Z500 Anomaly Correlation Horizon', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(0, 1.0)

# RMSE Plot
ax2.plot(weeks, df['FuXi_Z500_RMSE'], marker='o', linewidth=3, label='FuXi-S2S', color='#0072B2')
ax2.plot(weeks, df['Spire_Z500_RMSE'], marker='s', linewidth=3, label='Spire S2S', color='#D55E00')
ax2.set_ylabel('RMSE (m2/s2)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Z500 RMSE Growth', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/skill_horizon_z500.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Map saved to {out_path}")
