#!/usr/bin/env python
"""
Additional simple baseline plots:
1. Bias (mean error) vs lead time
2. RMSE with persistence baseline added
"""
import os, warnings
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from arraylake import Client
import pandas as pd

warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#e0e0e0', 'grid.linewidth': 0.5,
    'grid.linestyle': '--', 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.fontsize': 10, 'legend.frameon': True, 'legend.edgecolor': '#ccc',
    'legend.fancybox': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C  = {'Spire': '#1f77b4', 'ECMWF': '#d62728', 'NCEP': '#2ca02c', 'Persistence': '#999999'}
LS = {'Spire': '-', 'ECMWF': '--', 'NCEP': '-.', 'Persistence': ':'}
LW = {'Spire': 2.0, 'ECMWF': 1.6, 'NCEP': 1.6, 'Persistence': 1.4}

def load_grib(path, var, level=None):
    filt = {'shortName': var}
    if level: filt['level'] = level
    ds = xr.open_dataset(path, engine='cfgrib',
                         backend_kwargs={'filter_by_keys': filt})
    return ds[var]

def to_daily(da, agg):
    hrs  = da.step.values / np.timedelta64(1, 'h')
    days = (hrs - 0.1) // 24 + 1
    da   = da.assign_coords(day=('step', days))
    return getattr(da.groupby('day'), agg)('step')

def daily_rmse(f, o):
    return np.sqrt(((f - o)**2).mean(dim=['latitude', 'longitude']))

def daily_bias(f, o):
    return (f - o).mean(dim=['latitude', 'longitude'])

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    out  = os.path.join(here, 'figures')
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
    e_raw = load_grib(os.path.join(base, 'ecmwf/data/sfc_cf_20260101_6h.grib'), 'mx2t6')
    n_raw = load_grib(os.path.join(base, 'ncep/data/sfc_cf_20260101_6h.grib'),  'mx2t6')
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
    sp_raw = sp_raw.sel(latitude=slice(0, 50), longitude=slice(55, 105))

    print('Loading ERA5 …')
    end = pd.Timestamp('2026-01-01') + pd.Timedelta(days=nd)
    era = ds_era['2m_temperature'].sel(
        latitude=slice(51, -1), longitude=slice(54, 106),
        time=slice('2026-01-02', end.strftime('%Y-%m-%dT23:00'))).compute() - 273.15
    era_d = era.resample(time='1D').max('time').isel(time=slice(0, nd))

    # Align
    sp_d  = sp_raw.rename({'step': 'day'}).assign_coords(day=np.arange(1, nd+1))
    era_d = era_d.assign_coords(time=np.arange(1, nd+1)).rename({'time': 'day'})
    sp_d  = sp_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)
    era_d = era_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)

    # Persistence = day-1 ERA5 repeated for all days
    persist = era_d.sel(day=1).drop_vars('day')  # snapshot
    # broadcast to all days
    persist_all = persist.expand_dims(day=np.arange(1, nd+1))

    days = np.arange(1, nd+1)

    def smooth(arr, w=3):
        return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

    # --- Plot 1: Bias vs lead time ---
    print('Plotting bias …')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, da in [('Spire', sp_d), ('ECMWF', e_d), ('NCEP', n_d)]:
        bias = smooth(daily_bias(da, era_d).values)
        ax.plot(days, bias, color=C[name], ls=LS[name], lw=LW[name], label=name)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlim(1, nd)
    ax.set_xlabel('Forecast lead time (days)')
    ax.set_ylabel('Mean Bias (°C)')
    ax.set_title('Tmax — Mean Bias vs lead time (verified against ERA5)')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.legend()
    for s in ax.spines.values(): s.set_linewidth(0.8); s.set_color('#333')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'tmax_bias.png'))
    plt.close()
    print('  ✓ tmax_bias.png')

    # --- Plot 2: RMSE with persistence baseline ---
    print('Plotting RMSE + persistence …')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, da in [('Spire', sp_d), ('ECMWF', e_d), ('NCEP', n_d)]:
        r = smooth(daily_rmse(da, era_d).values)
        ax.plot(days, r, color=C[name], ls=LS[name], lw=LW[name], label=name)
    # persistence
    r_p = smooth(daily_rmse(persist_all, era_d).values)
    ax.plot(days, r_p, color=C['Persistence'], ls=LS['Persistence'],
            lw=LW['Persistence'], label='Persistence')
    ax.set_xlim(1, nd)
    ax.set_xlabel('Forecast lead time (days)')
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Tmax — RMSE vs lead time (with persistence baseline)')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.legend()
    for s in ax.spines.values(): s.set_linewidth(0.8); s.set_color('#333')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'tmax_rmse_persistence.png'))
    plt.close()
    print('  ✓ tmax_rmse_persistence.png')

    # --- Plot 3: RMSE Skill Score vs persistence ---
    print('Plotting skill score …')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rmse_p = daily_rmse(persist_all, era_d).values
    for name, da in [('Spire', sp_d), ('ECMWF', e_d), ('NCEP', n_d)]:
        rmse_m = daily_rmse(da, era_d).values
        ss = smooth(1 - rmse_m / rmse_p)
        ax.plot(days, ss, color=C[name], ls=LS[name], lw=LW[name], label=name)
    ax.axhline(0, color='#999', ls=':', lw=0.8)
    ax.set_xlim(1, nd)
    ax.set_xlabel('Forecast lead time (days)')
    ax.set_ylabel('RMSE Skill Score')
    ax.set_title('Tmax — Skill Score vs Persistence (>0 = beats persistence)')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.legend()
    for s in ax.spines.values(): s.set_linewidth(0.8); s.set_color('#333')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'tmax_skill_score.png'))
    plt.close()
    print('  ✓ tmax_skill_score.png')

    print('\nDone. 3 extra plots saved.')

if __name__ == '__main__':
    main()
