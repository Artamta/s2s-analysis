#!/usr/bin/env python
"""
Publication-quality S2S benchmark plots.
- Daily lead time (days 1–42)
- Proper Anomaly Correlation (removes time-mean climatological field)
- Tmax and Z500
- 4 clean plots: Tmax RMSE, Tmax ACC, Z500 RMSE, Z500 ACC
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

# ── Publication style ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     '#333333',
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.color':         '#e0e0e0',
    'grid.linewidth':     0.5,
    'grid.linestyle':     '--',
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.labelsize':     12,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.minor.size':   2,
    'ytick.minor.size':   2,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize':    10,
    'legend.frameon':     True,
    'legend.edgecolor':   '#cccccc',
    'legend.fancybox':    False,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
})

C  = {'Spire': '#1f77b4', 'ECMWF': '#d62728', 'NCEP': '#2ca02c'}
LS = {'Spire': '-', 'ECMWF': '--', 'NCEP': '-.'}
LW = {'Spire': 2.0, 'ECMWF': 1.6, 'NCEP': 1.6}

# ── Metrics ──────────────────────────────────────────────────────────────────
def daily_rmse(fcst, obs):
    """RMSE per day, averaged over all grid points."""
    return np.sqrt(((fcst - obs)**2).mean(dim=['latitude', 'longitude']))

def daily_acc(fcst, obs):
    """
    Proper Anomaly Correlation Coefficient per day.
    Removes the time-mean field (pseudo-climatology) from both forecast
    and observation before computing spatial correlation.
    This ensures ACC measures actual forecast SKILL, not just the
    equator-to-pole gradient.
    """
    # Remove mean field across all lead times (pseudo-climatology)
    clim_f = fcst.mean(dim='day')
    clim_o = obs.mean(dim='day')
    f_anom = fcst - clim_f
    o_anom = obs  - clim_o

    # Spatial correlation of anomalies per day
    cov    = (f_anom * o_anom).mean(dim=['latitude', 'longitude'])
    std_f  = np.sqrt((f_anom**2).mean(dim=['latitude', 'longitude']))
    std_o  = np.sqrt((o_anom**2).mean(dim=['latitude', 'longitude']))
    return cov / (std_f * std_o + 1e-12)

# ── Data loading ─────────────────────────────────────────────────────────────
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

def load_var(cfg, base, ds_spire, ds_era5, max_days=42):
    """Load all 4 datasets for one variable, return aligned daily arrays."""
    e_raw = load_grib(os.path.join(base, 'ecmwf/data', cfg['grib_file']),
                      cfg['grib_var'], cfg.get('grib_level'))
    n_raw = load_grib(os.path.join(base, 'ncep/data', cfg['grib_file']),
                      cfg['grib_var'], cfg.get('grib_level'))

    e_d = to_daily(e_raw, cfg['agg']) * cfg.get('scale', 1) + cfg.get('offset', 0)
    n_d = to_daily(n_raw, cfg['agg']) * cfg.get('scale', 1) + cfg.get('offset', 0)
    nd  = min(max_days, int(e_d.day.max()), int(n_d.day.max()))
    e_d = e_d.sel(day=slice(1, nd))
    n_d = n_d.sel(day=slice(1, nd))

    # Spire
    steps  = [np.timedelta64(d, 'D') for d in range(1, nd+1)]
    sp_raw = ds_spire[cfg['spire_var']].sel(
                 reference_time='2026-01-01', step=steps).compute()
    if cfg.get('spire_level'):
        dim = 'isobar' if 'isobar' in sp_raw.dims else 'isobaricInhPa'
        sp_raw = sp_raw.sel({dim: cfg['spire_level']}, method='nearest')
    if sp_raw.latitude.values[0] > sp_raw.latitude.values[-1]:
        sp_raw = sp_raw.isel(latitude=slice(None, None, -1))
    sp_raw = sp_raw.sel(latitude=slice(0, 50), longitude=slice(55, 105))
    sp_raw = sp_raw * cfg.get('spire_scale', 1) + cfg.get('offset', 0)

    # ERA5
    end = pd.Timestamp('2026-01-01') + pd.Timedelta(days=nd)
    era = ds_era5[cfg['era5_var']].sel(
        latitude=slice(51, -1), longitude=slice(54, 106),
        time=slice('2026-01-02', end.strftime('%Y-%m-%dT23:00')))
    if cfg.get('era5_level'):
        era = era.sel(level=cfg['era5_level'])
    era_d = getattr(era.resample(time='1D'), cfg.get('era5_agg', cfg['agg']))('time').compute()
    era_d = era_d.isel(time=slice(0, nd))
    era_d = era_d * cfg.get('era5_scale', 1) + cfg.get('offset', 0)

    # Align coordinates
    sp_d  = sp_raw.rename({'step': 'day'}).assign_coords(day=np.arange(1, nd+1))
    era_d = era_d.assign_coords(time=np.arange(1, nd+1)).rename({'time': 'day'})
    sp_d  = sp_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)
    era_d = era_d.interp(latitude=e_d.latitude, longitude=e_d.longitude)

    return e_d, n_d, sp_d, era_d, nd

# ── Plotting ─────────────────────────────────────────────────────────────────
def make_plot(out_path, days, series, ylabel, title, threshold=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for name in ['ECMWF', 'NCEP', 'Spire']:
        y = series[name]
        ax.plot(days, y, color=C[name], ls=LS[name], lw=LW[name], label=name)

    if threshold is not None:
        ax.axhline(threshold, color='#999999', ls=':', lw=0.8)
        ax.text(days[-1]-1, threshold+0.02, f'{threshold}', color='#999',
                fontsize=9, ha='right', va='bottom')

    ax.set_xlim(1, days[-1])
    ax.set_xlabel('Forecast lead time (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.legend(loc='best')

    # Keep all 4 spines for publication look
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ {os.path.basename(out_path)}')

# ── Main ─────────────────────────────────────────────────────────────────────
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

    VARS = [
        dict(name='Tmax', unit='°C',
             grib_file='sfc_cf_20260101_6h.grib', grib_var='mx2t6',
             spire_var='air_temperature_max',
             era5_var='2m_temperature', era5_agg='max',
             offset=-273.15, agg='max'),
        dict(name='Z500', unit='m',
             grib_file='pl_cf_20260101.grib', grib_var='gh', grib_level=500,
             spire_var='geopotential_height_at_isobaric_levels', spire_level=50000,
             era5_var='geopotential', era5_level=500, era5_scale=1/9.80665,
             era5_agg='mean', agg='mean'),
    ]

    for cfg in VARS:
        vname = cfg['name']
        print(f'\n── {vname} ──')

        e_d, n_d, sp_d, era_d, nd = load_var(cfg, base, ds_sp, ds_era)
        days = np.arange(1, nd+1)

        # RMSE
        rmse_series = {
            'ECMWF': daily_rmse(e_d, era_d).values,
            'NCEP':  daily_rmse(n_d, era_d).values,
            'Spire': daily_rmse(sp_d, era_d).values,
        }
        make_plot(os.path.join(out, f'{vname.lower()}_rmse.png'),
                  days, rmse_series,
                  f'RMSE ({cfg["unit"]})',
                  f'{vname} — RMSE vs lead time (verified against ERA5)')

        # ACC (proper anomaly correlation, 3-day running mean for smoothness)
        def smooth(arr, w=3):
            return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

        acc_series = {
            'ECMWF': smooth(daily_acc(e_d, era_d).values),
            'NCEP':  smooth(daily_acc(n_d, era_d).values),
            'Spire': smooth(daily_acc(sp_d, era_d).values),
        }
        make_plot(os.path.join(out, f'{vname.lower()}_acc.png'),
                  days, acc_series,
                  'ACC',
                  f'{vname} — Anomaly Correlation vs lead time (verified against ERA5)',
                  threshold=0.5)

    print('\nDone. 4 plots saved to:', out)

if __name__ == '__main__':
    main()
