"""
07c_dump_t2m_weekwise.py — Week-wise T2M scatter dump.

Loads raw T2M from all 4 models, computes weekly-mean anomalies vs ERA5
daily-mean T2M, and appends T2M_<MODEL>_W<1-6>_fcst/_obs keys to the
existing scatter_points_weekwise.npz.

Truth : ERA5 daily-mean t2m (era5_daily_t2m.nc, built from ARCO hourly)
SPIRE : air_temperature from Zarr ensemble mean
FuXi  : 't2m' channel, 11-member mean, daily snapshots averaged to weekly
ECMWF : (mx2t6 + mn2t6) / 2 PF-mean proxy
NCEP  : same as ECMWF

Output: paper/results/scatter_points_weekwise.npz  (UPDATED in-place)

NOTE: This is I/O-heavy (FuXi: 11 members × 42 days × 13 inits).
      Run on a dedicated CPU: ~15-30 min.
"""
import os, sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

DATA   = '/storage/raj.ayush/s2s-forecast-data'
OUTF   = '/home/raj.ayush/s2s/s2s_anlysis/paper/results/scatter_points_weekwise.npz'
OPEN   = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})

init_dates = [
    '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
    '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
    '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26',
]
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(
        da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(),
        LAND
    )


def add_pairs(store, key, fcst, obs):
    a = np.asarray(fcst).ravel()
    b = np.asarray(obs).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    store.setdefault(key + '_fcst', []).append(a[m])
    store.setdefault(key + '_obs',  []).append(b[m])


# ----- ERA5 truth -----
dailyT = xr.open_dataset(f'{DATA}/era5/daily/era5_daily_t2m.nc')['t2m']
clim_t2 = to_grid(dailyT.mean('time'))  # T2M climatology

def era_week_t(valid):
    try:
        return to_grid(dailyT.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


# ----- FuXi day loader -----
def fuxi_day(init_str, day):
    fs = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p):
            continue
        da = xr.open_dataset(p)['__xarray_dataarray_variable__'].sel(channel='t2m')
        for d in list(da.dims):
            if d not in ('lat', 'lon', 'latitude', 'longitude'):
                da = da.mean(d)
        fs.append(da)
    return None if not fs else xr.concat(fs, 'm').mean('m')


# ----- ECMWF/NCEP loader -----
def load_op_t2m(model, init_str):
    base = f'{DATA}/{model}/data'
    try:
        mx = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',
                             filter_by_keys={'shortName': 'mx2t6'}, **OPEN)['mx2t6']
        mn = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',
                             filter_by_keys={'shortName': 'mn2t6'}, **OPEN)['mn2t6']
        t = (mx + mn) / 2.0
        return t.mean('number') if 'number' in t.dims else t
    except Exception as e:
        print(f"  {model} t2m fail {init_str}: {e}", flush=True)
        return None


# ============== MAIN LOOP ==============
store = {}

for idx, init in enumerate(init_dates):
    init_str = pd.to_datetime(init).strftime('%Y%m%d')
    print(f"T2M init {idx+1}/{len(init_dates)}: {init}", flush=True)

    # Load models for this init
    try:
        sp = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr",
                          group='mean_stddev').sel(reference_time=init)['air_temperature']
    except Exception as e:
        sp = None; print(f"  SPIRE fail: {e}", flush=True)

    ec = load_op_t2m('ecmwf', init_str)
    nc = load_op_t2m('ncep', init_str)

    # FuXi: load all 42 days at once for this init
    fx = {}
    for d in range(1, 43):
        fx[d] = fuxi_day(init_str, d)

    for wi, (wn, ds_, de) in enumerate(weeks):
        wk = wi + 1
        dts = pd.date_range(start=init, periods=42)[ds_ - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dts if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid:
            continue
        o = era_week_t(valid)
        if o is None or np.isnan(o).all():
            continue

        f = {}
        # SPIRE
        if sp is not None:
            f['SPIRE'] = to_grid(sp.isel(step=slice(ds_ - 1, de)).mean('step'))
        # FuXi
        fd = [fx[d] for d in range(ds_, de + 1) if fx.get(d) is not None]
        if fd:
            f['FuXi'] = to_grid(xr.concat(fd, 't').mean('t'))
        # ECMWF / NCEP
        for nm, arr in [('ECMWF', ec), ('NCEP', nc)]:
            if arr is not None and arr.sizes.get('step', 0) >= de:
                f[nm] = to_grid(arr.isel(step=slice(ds_ - 1, de)).mean('step'))

        for m in MODELS:
            if m in f:
                # Anomaly = field - climatology
                fcst_anom = (f[m] - clim_t2).values
                obs_anom  = (o - clim_t2).values
                add_pairs(store, f'T2M_{m}_W{wk}', fcst_anom, obs_anom)
                add_pairs(store, f'T2M_{m}', fcst_anom, obs_anom)

print('T2M week-wise extraction done', flush=True)

# Concatenate
new_keys = {k: np.concatenate(v) for k, v in store.items()}

# Merge with existing npz (keep TP + Z500 keys)
existing = dict(np.load(OUTF))
# Remove old pooled T2M keys — we're replacing them with fresh ones
for k in list(existing.keys()):
    if k.startswith('T2M_'):
        del existing[k]

existing.update(new_keys)
np.savez_compressed(OUTF, **existing)

print(f'\nUPDATED {OUTF} with {len(new_keys)} new T2M arrays', flush=True)
print(f'Total arrays in file: {len(existing)}', flush=True)
for m in MODELS:
    for wk in range(1, 7):
        k = f'T2M_{m}_W{wk}_fcst'
        if k in new_keys:
            print(f'  T2M   {m:6s} W{wk} n={new_keys[k].size}', flush=True)
print('T2M_WEEKWISE_DONE', flush=True)
