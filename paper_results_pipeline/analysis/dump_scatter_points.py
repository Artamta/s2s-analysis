"""
Pool grid-point (forecast, ERA5) pairs over all 13 inits x 6 lead-weeks x land
points, for the density scatter figure (reference: FuXi-S2S / TianXing-S2S Fig 10).
Three variables, four systems:
  * TP   : raw weekly-mean precip recovered as (stored anomaly + clim_6h),
           verified vs TRUE 24-h daily ERA5 (era5_daily_tp.nc). Units mm/day.
  * Z500 : weekly-mean geopotential-height ANOMALY from weekly_anom_fields.nc. Units m.
  * T2M  : weekly-mean 2-m temperature reloaded from the raw ensembles
           (SPIRE zarr, FuXi nc, ECMWF/NCEP grib). Units K.
All fields land-masked, India domain. TP & Z500 cost no model reload; only T2M reloads.
Output: analysis/scatter_points.npz  with keys "<VAR>_<MODEL>_fcst" / "_obs".
"""
import os, sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
target_lat = np.arange(38, 5, -1.5); target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)


def add_pairs(store, key, fcst, obs):
    """flatten a (lat,lon) fcst/obs pair, keep finite matches, append to store."""
    a = np.asarray(fcst).ravel(); b = np.asarray(obs).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    store.setdefault(key + '_fcst', []).append(a[m])
    store.setdefault(key + '_obs', []).append(b[m])


store = {}

# ---------- TP (corrected) + Z500 (anomaly), no model reload ----------
era6 = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
clim6 = to_grid(era6.mean('time'))
daily = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
fields = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
init_dates = [str(x) for x in fields['init'].values]


def era_daily_week(valid):
    try:
        return to_grid(daily.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


for ii, init in enumerate(init_dates):
    for wi, (wn, ds, de) in enumerate(weeks):
        dates = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid:
            continue
        # TP truth (true daily ERA5) and Z500 truth (anomaly)
        o_tp = era_daily_week(valid)
        o_z = fields['z_obs'].isel(init=ii, week=wi).values
        for m in MODELS:
            a_tp = fields['tp_fcst'].sel(model=m).isel(init=ii, week=wi)
            if o_tp is not None and not np.isnan(o_tp).all() and not np.isnan(a_tp).all():
                add_pairs(store, f'TP_{m}', (a_tp + clim6).values, o_tp.values)
            a_z = fields['z_fcst'].sel(model=m).isel(init=ii, week=wi).values
            if np.isfinite(a_z).any() and np.isfinite(o_z).any():
                add_pairs(store, f'Z500_{m}', a_z, o_z)
print('TP + Z500 pooled (no reload)', flush=True)

# ---------- T2M, reload raw ensembles ----------
dailyT = xr.open_dataset(f'{ADIR}/era5_daily_t2m.nc')['t2m']


def era_week_t(valid):
    try:
        return to_grid(dailyT.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


def fuxi_day(init_str, day):
    fs = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p): continue
        da = xr.open_dataset(p)['__xarray_dataarray_variable__'].sel(channel='t2m')
        for d in list(da.dims):
            if d not in ('lat', 'lon', 'latitude', 'longitude'): da = da.mean(d)
        fs.append(da)
    return None if not fs else xr.concat(fs, 'm').mean('m')


def load_op_t2m(model, init_str):
    base = f'{DATA}/{model}/data'
    try:
        mx = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'mx2t6'}, **OPEN)['mx2t6']
        mn = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'mn2t6'}, **OPEN)['mn2t6']
        t = (mx + mn) / 2.0
        return t.mean('number') if 'number' in t.dims else t
    except Exception as e:
        print(f"  {model} t2m fail {init_str}: {e}", flush=True); return None


for init in init_dates:
    init_str = pd.to_datetime(init).strftime('%Y%m%d'); print(f"=== T2M INIT {init} ===", flush=True)
    try:
        sp = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr", group='mean_stddev').sel(reference_time=init)['air_temperature']
    except Exception as e:
        sp = None; print("  SPIRE fail", e, flush=True)
    ec = load_op_t2m('ecmwf', init_str); nc = load_op_t2m('ncep', init_str)
    fx = {d: fuxi_day(init_str, d) for d in range(1, 43)}
    for wn, ds_, de in weeks:
        dts = pd.date_range(start=init, periods=42)[ds_ - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dts if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid: continue
        o = era_week_t(valid)
        if o is None or np.isnan(o).all(): continue
        f = {}
        if sp is not None: f['SPIRE'] = to_grid(sp.isel(step=slice(ds_ - 1, de)).mean('step'))
        fd = [fx[d] for d in range(ds_, de + 1) if fx.get(d) is not None]
        if fd: f['FuXi'] = to_grid(xr.concat(fd, 't').mean('t'))
        for nm, arr in [('ECMWF', ec), ('NCEP', nc)]:
            if arr is not None and arr.sizes.get('step', 0) >= de:
                f[nm] = to_grid(arr.isel(step=slice(ds_ - 1, de)).mean('step'))
        for m in MODELS:
            if m in f:
                add_pairs(store, f'T2M_{m}', f[m].values, o.values)
print('T2M pooled (reloaded)', flush=True)

out = {k: np.concatenate(v) for k, v in store.items()}
np.savez_compressed(f'{ADIR}/scatter_points.npz', **out)
print('WROTE scatter_points.npz with', len(out), 'arrays', flush=True)
for var in ['TP', 'Z500', 'T2M']:
    for m in MODELS:
        k = f'{var}_{m}_fcst'
        if k in out:
            print(f'  {var:5s} {m:6s} n={out[k].size}', flush=True)
print('SCATTER_DUMP_DONE', flush=True)
