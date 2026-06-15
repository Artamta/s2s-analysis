"""
FAST week-wise scatter dump — TP + Z500 only (no T2M model reload).
T2M data comes from the existing pooled scatter_points.npz.
Keys: <VAR>_<MODEL>_W<1-6>_fcst / _obs   (+ pooled keys)
Output: paper/results/scatter_points_weekwise.npz
"""
import os, sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

DATA   = '/storage/raj.ayush/s2s-forecast-data'
ADIR   = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
OUTDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
OPEN   = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
LAND   = get_land_mask(target_lat, target_lon)
weeks  = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
          ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)


def add_pairs(store, key, fcst, obs):
    a = np.asarray(fcst).ravel(); b = np.asarray(obs).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    store.setdefault(key + '_fcst', []).append(a[m])
    store.setdefault(key + '_obs',  []).append(b[m])


store = {}

# ---------- TP + Z500 from weekly_anom_fields.nc ----------
era6   = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib',
                         filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
clim6  = to_grid(era6.mean('time'))
daily  = xr.open_dataset(f'{DATA}/era5/daily/era5_daily_tp.nc')['tp']
clim_tp = to_grid(daily.mean('time'))
fields = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
init_dates = [str(x) for x in fields['init'].values]


def era_daily_week(valid):
    try:
        return to_grid(daily.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


for ii, init in enumerate(init_dates):
    print(f'TP+Z500 init {ii+1}/{len(init_dates)}: {init}', flush=True)
    for wi, (wn, ds, de) in enumerate(weeks):
        wk = wi + 1
        dates = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid:
            continue
        o_tp = era_daily_week(valid)
        o_z  = fields['z_obs'].isel(init=ii, week=wi).values
        for m in MODELS:
            a_tp = fields['tp_fcst'].sel(model=m).isel(init=ii, week=wi)
            tp_fac = 24.0 if m == 'FuXi' else 1.0
            if o_tp is not None and not np.isnan(o_tp).all() and not np.isnan(a_tp).all():
                fcst_tp = ((a_tp + clim6) * tp_fac - clim_tp).values
                obs_tp  = (o_tp - clim_tp).values
                add_pairs(store, f'TP_{m}_W{wk}', fcst_tp, obs_tp)
                add_pairs(store, f'TP_{m}', fcst_tp, obs_tp)
            a_z = fields['z_fcst'].sel(model=m).isel(init=ii, week=wi).values
            if np.isfinite(a_z).any() and np.isfinite(o_z).any():
                add_pairs(store, f'Z500_{m}_W{wk}', a_z, o_z)
                add_pairs(store, f'Z500_{m}', a_z, o_z)
print('TP + Z500 done', flush=True)

# ---------- T2M: copy from existing pooled scatter ----------
old = np.load(f'{ADIR}/scatter_points.npz')
for m in MODELS:
    for suf in ('_fcst', '_obs'):
        k = f'T2M_{m}{suf}'
        if k in old.files:
            store[k] = [old[k]]
            print(f'Copied pooled {k} n={old[k].size}', flush=True)

# Concatenate and save
out = {k: np.concatenate(v) for k, v in store.items()}
np.savez_compressed(f'{OUTDIR}/scatter_points_weekwise.npz', **out)
print(f'WROTE scatter_points_weekwise.npz with {len(out)} arrays', flush=True)
for var in ['TP', 'Z500']:
    for m in MODELS:
        for wk in range(1, 7):
            k = f'{var}_{m}_W{wk}_fcst'
            if k in out:
                print(f'  {var:5s} {m:6s} W{wk} n={out[k].size}', flush=True)
print('SCATTER_WEEKWISE_DONE', flush=True)
