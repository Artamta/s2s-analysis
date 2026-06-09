#!/usr/bin/env python
"""
Map-wise plots for Tmax:
1. Actual Weekly Means (ERA5, Spire, ECMWF, NCEP)
2. Weekly Biases (Spire - ERA5, ECMWF - ERA5, NCEP - ERA5)
"""
import os
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from arraylake import Client

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 12,
})

def load_grib(path, var):
    return xr.open_dataset(path, engine='cfgrib',
                           backend_kwargs={'filter_by_keys': {'shortName': var}})[var]

def to_daily(da, agg):
    hrs  = da.step.values / np.timedelta64(1, 'h')
    days = (hrs - 0.1) // 24 + 1
    da   = da.assign_coords(day=('step', days))
    return getattr(da.groupby('day'), agg)('step')

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(base, 'figures', 'maps')
    os.makedirs(out, exist_ok=True)

    print('Connecting …')
    client  = Client()
    repo    = client.get_repo('artamta/s2s-research')
    session = repo.readonly_session('main')
    ds_sp   = xr.open_zarr(session.store, group='mean_stddev')
    ds_era  = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                           storage_options={'token': 'anon'})

    # --- Load Tmax ---
    print('Loading ECMWF/NCEP …')
    e_raw = load_grib(os.path.join(os.path.dirname(base), 'ecmwf/data/sfc_cf_20260101_6h.grib'), 'mx2t6')
    n_raw = load_grib(os.path.join(os.path.dirname(base), 'ncep/data/sfc_cf_20260101_6h.grib'),  'mx2t6')
    e_d = to_daily(e_raw, 'max') - 273.15
    n_d = to_daily(n_raw, 'max') - 273.15
    nd  = min(42, int(e_d.day.max()), int(n_d.day.max()))
    e_d = e_d.sel(day=slice(1, nd))
    n_d = n_d.sel(day=slice(1, nd))

    print('Loading Spire …')
    steps  = [np.timedelta64(d, 'D') for d in range(1, nd+1)]
    sp_raw = ds_sp['air_temperature_max'].sel(
                 reference_time='2026-01-01', step=steps).compute() - 273.15
    if sp_raw.latitude.values[0] > sp_raw.latitude.values[-1]:
        sp_raw = sp_raw.isel(latitude=slice(None, None, -1))
    # Approximate regional crop for Spire to save memory before interp
    sp_raw = sp_raw.sel(latitude=slice(0, 50), longitude=slice(55, 105))

    print('Loading ERA5 …')
    end = pd.Timestamp('2026-01-01') + pd.Timedelta(days=nd)
    era = ds_era['2m_temperature'].sel(
        latitude=slice(51, -1), longitude=slice(54, 106),
        time=slice('2026-01-02', end.strftime('%Y-%m-%dT23:00'))).compute() - 273.15
    era_d = era.resample(time='1D').max('time').isel(time=slice(0, nd))

    sp_d  = sp_raw.rename({'step': 'day'}).assign_coords(day=np.arange(1, nd+1))
    era_d = era_d.assign_coords(time=np.arange(1, nd+1)).rename({'time': 'day'})
    
    print('Interpolating to ECMWF 1.5 deg grid ...')
    sp_d  = sp_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)
    era_d = era_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)
    n_d   = n_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)

    # Weeks definition
    weeks = [(1,7), (8,14), (15,21), (22,28), (29,35), (36,42)]
    
    # ---------------------------------------------------------
    # 1. Plot Tmax Weekly Means (Ground Truth + 3 Models)
    # ---------------------------------------------------------
    print('Plotting Tmax Weekly Means Maps ...')
    models = {'ERA5 (Ground Truth)': era_d, 'Spire': sp_d, 'ECMWF': e_d, 'NCEP': n_d}
    fig, axes = plt.subplots(6, 4, figsize=(16, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap = 'RdYlBu_r'
    vmin, vmax = -10, 40

    for i, (w_start, w_end) in enumerate(weeks):
        for j, (m_name, m_da) in enumerate(models.items()):
            ax = axes[i, j]
            ax.coastlines(linewidth=0.5, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
            
            w_data = m_da.sel(day=slice(w_start, w_end)).mean('day')
            im = ax.pcolormesh(w_data.longitude, w_data.latitude, w_data, 
                               transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
            
            if i == 0:
                ax.set_title(m_name, fontsize=14, fontweight='bold', pad=10)
            if j == 0:
                ax.text(-0.15, 0.5, f'Week {i+1}\n(Days {w_start}-{w_end})', 
                        va='center', ha='center', rotation='vertical', 
                        transform=ax.transAxes, fontsize=14, fontweight='bold')

    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Tmax (°C)', fontsize=14)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.05, hspace=0.1)
    fig.savefig(os.path.join(out, 'tmax_weekly_means_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ tmax_weekly_means_map.png')

    # ---------------------------------------------------------
    # 2. Plot Tmax Weekly Biases (3 Models vs ERA5)
    # ---------------------------------------------------------
    print('Plotting Tmax Weekly Biases Maps ...')
    bias_models = {'Spire Bias': sp_d, 'ECMWF Bias': e_d, 'NCEP Bias': n_d}
    fig, axes = plt.subplots(6, 3, figsize=(12, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_bias = 'coolwarm'
    vmin_b, vmax_b = -8, 8

    for i, (w_start, w_end) in enumerate(weeks):
        era_w = era_d.sel(day=slice(w_start, w_end)).mean('day')
        for j, (m_name, m_da) in enumerate(bias_models.items()):
            ax = axes[i, j]
            ax.coastlines(linewidth=0.5, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
            
            w_data = m_da.sel(day=slice(w_start, w_end)).mean('day')
            bias = w_data - era_w
            
            im = ax.pcolormesh(bias.longitude, bias.latitude, bias, 
                               transform=ccrs.PlateCarree(), cmap=cmap_bias, vmin=vmin_b, vmax=vmax_b)
            
            if i == 0:
                ax.set_title(m_name, fontsize=14, fontweight='bold', pad=10)
            if j == 0:
                ax.text(-0.15, 0.5, f'Week {i+1}\n(Days {w_start}-{w_end})', 
                        va='center', ha='center', rotation='vertical', 
                        transform=ax.transAxes, fontsize=14, fontweight='bold')

    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Tmax Bias (Model - ERA5) (°C)', fontsize=14)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.05, hspace=0.1)
    fig.savefig(os.path.join(out, 'tmax_weekly_biases_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ✓ tmax_weekly_biases_map.png')

if __name__ == '__main__':
    main()
