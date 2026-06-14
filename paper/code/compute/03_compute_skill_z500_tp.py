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
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias
from utils.verification_extra import get_land_mask, mask_land

G         = 9.80665
DATA      = '/storage/raj.ayush/s2s-forecast-data'
OPEN      = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
ADIR      = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
CLIM_PATH = '/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc'  # 1990-2019 30yr WMO
DEC25_PATH = '/storage/raj.ayush/s2s-forecast-data/era5/data/era5_dec2025_persistence.nc'

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
              '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
              '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
# IMD 4 homogeneous regions — proper state-boundary masks (no overlaps)
_mask_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc')
REGION_MASKS = {k: _mask_ds[k].values.astype(bool) for k in _mask_ds.data_vars}
ALL_INDIA_MASK = np.zeros((len(np.arange(38,5,-1.5)), len(np.arange(65,100,1.5))), dtype=bool)
for _m in REGION_MASKS.values(): ALL_INDIA_MASK |= _m
REGIONS = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
LAND = get_land_mask(target_lat, target_lon)


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)

#This handles cumulative precipitation — ECMWF/NCEP store TP as a running total, not daily values:

def weekly_mean_cumulative(cum, ds, de):
    days = de - ds + 1
    return (cum.isel(step=de - 1) / days) if ds == 1 else (cum.isel(step=de - 1) - cum.isel(step=ds - 2)) / days


print("Loading ERA5 ...", flush=True)
era_tp_raw = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
era_z_raw  = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G

print("Loading 30-yr climatology (1990-2019) ...", flush=True)
clim_era = xr.open_dataset(CLIM_PATH)   # dims: dayofyear, latitude, longitude

# Dec 25-30 2025 patch for Jan 1 init persistence
dec25 = xr.open_dataset(DEC25_PATH) if os.path.exists(DEC25_PATH) else None
if dec25 is not None:
    print(f"Loaded Dec25 persistence patch: {DEC25_PATH}", flush=True)
else:
    print("WARNING: Dec25 persistence patch not found — Jan 1 init uses 1 day only", flush=True)


def clim_week(doys, var, scale=1.0):
    """Return weekly climatological mean on target grid for given day-of-year list."""
    c = clim_era[var].sel(dayofyear=doys).mean('dayofyear') * scale
    return to_grid(c)



def era_week(raw, valid):
    try:
        return to_grid(raw.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception as e:
        print(f"  ERA5 obs not found for {valid[0]} → {valid[-1]}: {e}", flush=True)
        return None


def era_week_pers(raw_tp, raw_z, valid):
    """Persistence: uses dec25 patch for Dec 25-31, GRIB for Jan 1+."""
    GRIB_START = '2026-01-01'
    before = [d for d in valid if d < GRIB_START]  # needs dec25
    after  = [d for d in valid if d >= GRIB_START] # in GRIB

    def _load_days(tp_pieces, z_pieces, days, source):
        for d in days:
            try:
                if source == 'dec25' and dec25 is not None:
                    tp_pieces.append(to_grid(dec25['tp'].sel(time=d)))
                    z_pieces.append(to_grid(dec25['z500'].sel(time=d)))
                else:
                    tp_pieces.append(to_grid(raw_tp.sel(time=d)))
                    z_pieces.append(to_grid(raw_z.sel(time=d)))
            except Exception as e:
                print(f"  pers load error {d}/{source}: {e}", flush=True)

    tp_p, z_p = [], []
    if before:  _load_days(tp_p, z_p, before, 'dec25')
    if after:   _load_days(tp_p, z_p, after,  'grib')

    if not tp_p:
        return None, None
    pers_tp = xr.concat(tp_p, 't').mean('t')
    pers_z  = xr.concat(z_p,  't').mean('t')
    print(f"  Persistence: {len(tp_p)}/7 days (dec25={len(before)}, grib={len(after)})", flush=True)
    return pers_tp, pers_z


def load_spire(init):
    s = xr.open_zarr(f'{DATA}/spire/spire_hindcast_jfm.zarr', group='mean_stddev').sel(reference_time=init)
    return s['precipitation_amount'], s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0)


def fuxi_day(init_str, day, ch):
    fs = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p):
            print(f"  FuXi missing: {init_str}/member/{mem:02d}/day{day:02d}", flush=True)
            continue
        try:
            da = xr.open_dataset(p)['__xarray_dataarray_variable__']
            da = da.sel(channel='tp') if ch == 'tp' else (da.sel(channel='z500') / G)
            da = da.squeeze()   # remove size-1 dims (time, lead_time)
            fs.append(da)
        except Exception as e:
            print(f"  FuXi load error: {init_str}/member/{mem:02d}/day{day:02d}: {e}", flush=True)
    if not fs:
        print(f"  FuXi: NO members found for {init_str} day {day:02d}", flush=True)
        return None
    return xr.concat(fs, dim='m').mean('m')


def load_op(model, init_str):
    base = f'{DATA}/{model}/data'
    tp = gh = None
    try:
        d = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
        tp = d.mean('number') if 'number' in d.dims else d
        print(f"  {model.upper()} TP loaded: {init_str} ({tp.sizes.get('step','?')} steps, {tp.sizes.get('number','det')} members)", flush=True)
    except Exception as e:
        print(f"  {model.upper()} TP NOT FOUND {init_str}: {e}", flush=True)
    try:
        d = xr.open_dataset(f'{base}/pl_pf_{init_str}.grib', filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
        if 'isobaricInhPa' in d.dims: d = d.sel(isobaricInhPa=500)
        gh = d.mean('number') if 'number' in d.dims else d
        print(f"  {model.upper()} Z500 loaded: {init_str} ({gh.sizes.get('step','?')} steps)", flush=True)
    except Exception as e:
        print(f"  {model.upper()} Z500 NOT FOUND {init_str}: {e}", flush=True)
    return tp, gh


def regional(field, rg):
    """Apply IMD state-boundary mask. Returns field with NaN outside region."""
    if rg == 'All India':
        mask = xr.DataArray(ALL_INDIA_MASK, dims=['lat', 'lon'],
                            coords={'lat': target_lat, 'lon': target_lon})
    else:
        mask = xr.DataArray(REGION_MASKS[rg], dims=['lat', 'lon'],
                            coords={'lat': target_lat, 'lon': target_lon})
    return field.where(mask)


def wstd(x, w):
    m = x.weighted(w).mean(['lat', 'lon'])
    return float(np.sqrt(((x - m) ** 2).weighted(w).mean(['lat', 'lon'])))


def metrics(model, var, rg, init, week, f, o, clim):
    if f is None or o is None:
        return None
    fr, orr, cr = regional(f, rg), regional(o, rg), regional(clim, rg)
    w = get_cosine_latitude_weights(target_lat)  # full grid; NaN pts excluded by weighted.mean
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
    fx_tp = {d: fuxi_day(init_str, d, 'tp')    for d in range(1, 43)}
    fx_z  = {d: fuxi_day(init_str, d, 'z500')  for d in range(1, 43)}
    n_fx_tp = sum(1 for v in fx_tp.values() if v is not None)
    n_fx_z  = sum(1 for v in fx_z.values()  if v is not None)
    print(f"  FuXi TP: {n_fx_tp}/42 days found | Z500: {n_fx_z}/42 days found", flush=True)

    # persistence = observed week immediately before init (uses dec25 patch for Jan 1)
    pre   = pd.date_range(end=pd.to_datetime(init) - pd.Timedelta(days=1), periods=7)
    pre_v = [d.strftime('%Y-%m-%d') for d in pre]
    pers_tp, pers_z = era_week_pers(era_tp_raw, era_z_raw, pre_v)


    for wi, (wn, ds, de) in enumerate(weeks):
        dates = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if not valid:
            continue
        o_tp, o_z = era_week(era_tp_raw, valid), era_week(era_z_raw, valid)
        if o_tp is None or o_z is None:
            print(f"  SKIP {wn}: ERA5 obs missing for {valid[0]}→{valid[-1]}", flush=True)
            continue

        # Week-specific 30-yr climatology (proper anomaly baseline)
        doys    = [pd.to_datetime(d).dayofyear for d in valid]
        clim_tp = clim_week(doys, 'tp',   scale=1000.0)   # m → mm
        clim_z  = clim_week(doys, 'z500', scale=1.0/G)    # geopotential → gpm

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

        for rg in REGIONS:
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
