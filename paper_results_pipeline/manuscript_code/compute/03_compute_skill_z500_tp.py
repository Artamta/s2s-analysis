"""
Enhanced verification pass for the JFM-2026 SPIRE benchmark.
Supersedes compute_skill_horizon_final.py and additionally produces, in ONE
heavy pass (so the node is loaded only once):

  * model='MME'         : equal-weight multi-model ensemble (mean of the 4 systems)
  * model='Persistence' : pre-initialization observed weekly anomaly, all leads
  * variance ratio      : cosine-weighted spatial std of forecast vs obs anomaly
  * saved anomaly fields : analysis/weekly_anom_fields.nc  (for spatial maps)

Outputs:
  analysis/skill_per_init_full.csv   (long format, incl. MME + Persistence)
  analysis/weekly_anom_fields.nc     (model,init,week,lat,lon anomaly fields)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias
from utils.verification_extra import get_land_mask, mask_land

G = 9.80665
DATA = '/storage/raj.ayush/s2s-forecast-data'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
              '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
              '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
REGION_BOUNDS = {
    'All India':            (5.0, 38.0, 65.0, 100.0),
    'northwest_india':      (22.0, 38.0, 68.0, 82.0),
    'central_india':        (18.0, 28.0, 72.0, 89.0),
    'south_peninsula':      (8.0, 20.0, 72.0, 85.0),
    'east_northeast_india': (20.0, 30.0, 85.0, 98.0),
}
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
LAND = get_land_mask(target_lat, target_lon)


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)


def weekly_mean_cumulative(cum, ds, de):
    days = de - ds + 1
    return (cum.isel(step=de - 1) / days) if ds == 1 else (cum.isel(step=de - 1) - cum.isel(step=ds - 2)) / days


print("Loading ERA5 ...", flush=True)
era_tp_raw = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
era_z_raw = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
clim_tp = to_grid(era_tp_raw.mean('time'))
clim_z = to_grid(era_z_raw.mean('time'))


def era_week(raw, valid):
    try:
        return to_grid(raw.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


def load_spire(init):
    s = xr.open_zarr(f'{DATA}/spire/spire_hindcast_jfm.zarr', group='mean_stddev').sel(reference_time=init)
    return s['precipitation_amount'], s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0)


def fuxi_day(init_str, day, ch):
    fs = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p):
            continue
        da = xr.open_dataset(p)['__xarray_dataarray_variable__']
        da = da.sel(channel='tp') if ch == 'tp' else (da.isel(channel=5) / G)
        for d in list(da.dims):
            if d not in ('lat', 'lon', 'latitude', 'longitude'):
                da = da.mean(d)
        fs.append(da)
    return None if not fs else xr.concat(fs, dim='m').mean('m')


def load_op(model, init_str):
    base = f'{DATA}/{model}/data'
    tp = gh = None
    try:
        d = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
        tp = d.mean('number') if 'number' in d.dims else d
    except Exception as e:
        print(f"  {model} tp fail {init_str}: {e}", flush=True)
    try:
        d = xr.open_dataset(f'{base}/pl_pf_{init_str}.grib', filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
        if 'isobaricInhPa' in d.dims: d = d.sel(isobaricInhPa=500)
        gh = d.mean('number') if 'number' in d.dims else d
    except Exception as e:
        print(f"  {model} gh fail {init_str}: {e}", flush=True)
    return tp, gh


def regional(field, rg):
    a, b, c, d = REGION_BOUNDS[rg]
    return field.sel(lat=slice(b, a), lon=slice(c, d))


def wstd(x, w):
    m = x.weighted(w).mean(['lat', 'lon'])
    return float(np.sqrt(((x - m) ** 2).weighted(w).mean(['lat', 'lon'])))


def metrics(model, var, rg, init, week, f, o, clim):
    if f is None or o is None:
        return None
    fr, orr, cr = regional(f, rg), regional(o, rg), regional(clim, rg)
    w = get_cosine_latitude_weights(fr.lat.values)
    try:
        return dict(variable=var, region=rg, week=week, init_date=init, model=model,
                    pcc=calc_wmo_acc(fr, orr, cr, w), rmse=calc_wmo_rmse(fr, orr, w),
                    bias=calc_wmo_bias(fr, orr, w),
                    fcst_std=wstd(fr - cr, w), obs_std=wstd(orr - cr, w),
                    fcst_mean=float(fr.weighted(w).mean(['lat', 'lon'])),
                    obs_mean=float(orr.weighted(w).mean(['lat', 'lon'])))
    except Exception as e:
        print(f"  metric fail {model}/{var}/{rg}/{week}/{init}: {e}", flush=True)
        return None


# anomaly-field store for spatial maps: [model, init, week, lat, lon]
ny, nx = len(target_lat), len(target_lon)
store = {v: np.full((len(MODELS), len(init_dates), len(weeks), ny, nx), np.nan) for v in ['tp', 'z']}
obs_store = {v: np.full((len(init_dates), len(weeks), ny, nx), np.nan) for v in ['tp', 'z']}

rows = []
for ii, init in enumerate(init_dates):
    init_str = pd.to_datetime(init).strftime('%Y%m%d')
    print(f"\n=== INIT {init} ===", flush=True)
    try:
        sp_tp, sp_z = load_spire(init)
    except Exception as e:
        sp_tp = sp_z = None; print(f"  SPIRE fail {e}", flush=True)
    ec_tp, ec_z = load_op('ecmwf', init_str)
    nc_tp, nc_z = load_op('ncep', init_str)
    fx_tp = {d: fuxi_day(init_str, d, 'tp') for d in range(1, 43)}
    fx_z = {d: fuxi_day(init_str, d, 5) for d in range(1, 43)}

    # persistence reference = observed week immediately before init
    pre = pd.date_range(end=pd.to_datetime(init) - pd.Timedelta(days=1), periods=7)
    pre_v = [d.strftime('%Y-%m-%d') for d in pre]
    pers_tp, pers_z = era_week(era_tp_raw, pre_v), era_week(era_z_raw, pre_v)

    for wi, (wn, ds, de) in enumerate(weeks):
        dates = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if not valid:
            continue
        o_tp, o_z = era_week(era_tp_raw, valid), era_week(era_z_raw, valid)
        if o_tp is None or o_z is None:
            continue

        f_tp, f_z = {}, {}
        if sp_tp is not None:
            f_tp['SPIRE'] = to_grid(sp_tp.isel(step=slice(ds - 1, de)).mean('step'))
            f_z['SPIRE'] = to_grid(sp_z.isel(step=slice(ds - 1, de)).mean('step'))
        td = [fx_tp[d] for d in range(ds, de + 1) if fx_tp.get(d) is not None]
        zd = [fx_z[d] for d in range(ds, de + 1) if fx_z.get(d) is not None]
        if td: f_tp['FuXi'] = to_grid(xr.concat(td, 't').mean('t'))
        if zd: f_z['FuXi'] = to_grid(xr.concat(zd, 't').mean('t'))
        for nm, tp, z in [('ECMWF', ec_tp, ec_z), ('NCEP', nc_tp, nc_z)]:
            if tp is not None and tp.sizes.get('step', 0) >= de:
                f_tp[nm] = to_grid(weekly_mean_cumulative(tp, ds, de))
            if z is not None and z.sizes.get('step', 0) >= de:
                f_z[nm] = to_grid(z.isel(step=slice(ds - 1, de)).mean('step'))
        # MME = equal-weight mean of the 4 systems (where available)
        base4_tp = [f_tp[m] for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP'] if m in f_tp]
        base4_z = [f_z[m] for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP'] if m in f_z]
        if len(base4_tp) >= 3: f_tp['MME'] = xr.concat(base4_tp, 'mm').mean('mm')
        if len(base4_z) >= 3: f_z['MME'] = xr.concat(base4_z, 'mm').mean('mm')
        if pers_tp is not None: f_tp['Persistence'] = pers_tp
        if pers_z is not None: f_z['Persistence'] = pers_z

        # store anomaly fields (All India grid) for spatial maps
        obs_store['tp'][ii, wi] = (o_tp - clim_tp).values
        obs_store['z'][ii, wi] = (o_z - clim_z).values
        for mi, m in enumerate(MODELS):
            if m in f_tp: store['tp'][mi, ii, wi] = (f_tp[m] - clim_tp).values
            if m in f_z: store['z'][mi, ii, wi] = (f_z[m] - clim_z).values

        for rg in REGION_BOUNDS:
            for m in MODELS:
                r = metrics(m, 'TP', rg, init, wn, f_tp.get(m), o_tp, clim_tp)
                if r: rows.append(r)
                r = metrics(m, 'Z500', rg, init, wn, f_z.get(m), o_z, clim_z)
                if r: rows.append(r)

pd.DataFrame(rows).to_csv(f'{ADIR}/skill_per_init_full.csv', index=False)
print(f"\nWROTE skill_per_init_full.csv ({len(rows)} rows)", flush=True)

# save anomaly fields
ds_out = xr.Dataset(
    {'tp_fcst': (('model', 'init', 'week', 'lat', 'lon'), store['tp']),
     'z_fcst': (('model', 'init', 'week', 'lat', 'lon'), store['z']),
     'tp_obs': (('init', 'week', 'lat', 'lon'), obs_store['tp']),
     'z_obs': (('init', 'week', 'lat', 'lon'), obs_store['z'])},
    coords={'model': MODELS, 'init': init_dates, 'week': [w[0] for w in weeks],
            'lat': target_lat, 'lon': target_lon})
ds_out.to_netcdf(f'{ADIR}/weekly_anom_fields.nc')
print("WROTE weekly_anom_fields.nc", flush=True)
print("COMPUTE_DONE", flush=True)
