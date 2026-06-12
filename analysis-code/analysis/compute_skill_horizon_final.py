"""
UNIFIED WMO-compliant weekly S2S skill horizon for the JFM-2026 SPIRE benchmark.
Supersedes compute_final_wmo_tp.py and compute_final_wmo_z500.py.

Tier-1 fixes implemented (2026-06-12):
  #1/#2  SPIRE & FuXi precipitation aggregated by MEAN over days (no false
         temporal de-accumulation); only ECMWF/NCEP are de-accumulated.
  #3     Per-model precipitation sanity gate printed for week 1.
  #4     Land mask (global_land_mask) -> land points only for TP and Z500.
  #5     ALL systems verified as ENSEMBLE MEANS
         (SPIRE pre-computed mean; FuXi 11-member mean; ECMWF/NCEP PF mean).
  #6     Bootstrap CIs over the 13 init dates (done at aggregation/plot stage
         from the per-init CSV written here).

Metrics by variable (per user decision 2026-06-12):
  TP    -> PCC only is publication-grade (ERA5 tp on disk is a 6-h window/day,
           so absolute mm/day differ ~15-20x across systems; PCC is scale-
           invariant). RMSE/Bias for TP are still written to CSV but flagged
           NOT-FOR-PUBLICATION and used only for the qualitative wet/dry panel.
  Z500  -> PCC + RMSE + Bias all publication-grade.

Output: analysis/skill_per_init.csv  (long format; drives every figure).
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias
from utils.verification_extra import get_land_mask, mask_land

G = 9.80665
DATA = '/storage/raj.ayush/s2s-forecast-data'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
OUT_CSV = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/skill_per_init.csv'

init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
              '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
              '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26']
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
REGION_BOUNDS = {  # (min_lat, max_lat, min_lon, max_lon); land mask applied on top
    'All India':            (5.0, 38.0, 65.0, 100.0),
    'northwest_india':      (22.0, 38.0, 68.0, 82.0),
    'central_india':        (18.0, 28.0, 72.0, 89.0),
    'south_peninsula':      (8.0, 20.0, 72.0, 85.0),
    'east_northeast_india': (20.0, 30.0, 85.0, 98.0),
}

LAND = get_land_mask(target_lat, target_lon)


def to_grid(da):
    """Rename to lat/lon if needed, interp to target grid, land-mask."""
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    da = da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze()
    return mask_land(da, LAND)


def weekly_mean_cumulative(cum_da, d_start, d_end):
    """ECMWF/NCEP: mean daily RATE over the week from a cumulative series (dim 'step')."""
    days = d_end - d_start + 1
    if d_start == 1:
        return cum_da.isel(step=d_end - 1) / days
    return (cum_da.isel(step=d_end - 1) - cum_da.isel(step=d_start - 2)) / days


# ----------------------------------------------------------------------
# ERA5 reference (continuous), + in-sample season climatology
# ----------------------------------------------------------------------
print("Loading ERA5 reference ...", flush=True)
era_tp_raw = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
era_z_raw = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
# keep on native grid; slice per week then to_grid
era_clim_tp = to_grid(era_tp_raw.mean('time'))
era_clim_z = to_grid(era_z_raw.mean('time'))


def era_week(raw, valid_dates):
    return to_grid(raw.sel(time=slice(valid_dates[0], valid_dates[-1])).mean('time'))


# ----------------------------------------------------------------------
# Ensemble-mean loaders -> native-grid weekly-mean field (pre to_grid)
# ----------------------------------------------------------------------
def load_spire(init_date):
    s = xr.open_zarr(f'{DATA}/spire/spire_hindcast_jfm.zarr', group='mean_stddev').sel(reference_time=init_date)
    return s['precipitation_amount'], s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0)


def fuxi_ensmean_day(init_str, day, channel):
    """Mean over 11 members of one FuXi day; channel='tp' or 5 (z500)."""
    fields = []
    for mem in range(11):
        p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p):
            continue
        da = xr.open_dataset(p)['__xarray_dataarray_variable__']
        da = da.sel(channel='tp') if channel == 'tp' else (da.isel(channel=5) / G)
        for d in list(da.dims):
            if d not in ('lat', 'lon', 'latitude', 'longitude'):
                da = da.mean(d)
        fields.append(da)
    if not fields:
        return None
    return xr.concat(fields, dim='m').mean('m')


def load_ecmwf_ncep(model, init_str):
    base = f'{DATA}/{model}/data'
    tp = gh = None
    try:
        d = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
        tp = d.mean('number') if 'number' in d.dims else d
    except Exception as e:
        print(f"   [{model}] tp load fail {init_str}: {e}", flush=True)
    try:
        d = xr.open_dataset(f'{base}/pl_pf_{init_str}.grib', filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
        if 'isobaricInhPa' in d.dims: d = d.sel(isobaricInhPa=500)
        gh = d.mean('number') if 'number' in d.dims else d
    except Exception as e:
        print(f"   [{model}] gh load fail {init_str}: {e}", flush=True)
    return tp, gh


def regional(field, region):
    mn_la, mx_la, mn_lo, mx_lo = REGION_BOUNDS[region]
    return field.sel(lat=slice(mx_la, mn_la), lon=slice(mn_lo, mx_lo))


def metrics(model, var, region, init_date, week, f, o, clim):
    if f is None or o is None:
        return None
    fr, orr, cr = regional(f, region), regional(o, region), regional(clim, region)
    w = get_cosine_latitude_weights(fr.lat.values)
    try:
        return dict(variable=var, region=region, week=week, init_date=init_date, model=model,
                    pcc=calc_wmo_acc(fr, orr, cr, w),
                    rmse=calc_wmo_rmse(fr, orr, w),
                    bias=calc_wmo_bias(fr, orr, w),
                    fcst_mean=float(fr.weighted(w).mean(['lat', 'lon'])),
                    obs_mean=float(orr.weighted(w).mean(['lat', 'lon'])))
    except Exception as e:
        print(f"   metric fail {model}/{var}/{region}/{week}/{init_date}: {e}", flush=True)
        return None


rows = []
sanity = []
for init_date in init_dates:
    init_str = pd.to_datetime(init_date).strftime('%Y%m%d')
    print(f"\n=== INIT {init_date} ===", flush=True)
    # preload ensemble-mean sources once
    try:
        sp_tp, sp_z = load_spire(init_date)
    except Exception as e:
        sp_tp = sp_z = None; print(f"   SPIRE fail: {e}", flush=True)
    ec_tp, ec_z = load_ecmwf_ncep('ecmwf', init_str)
    nc_tp, nc_z = load_ecmwf_ncep('ncep', init_str)
    # FuXi: preload ens-mean per day (1..42) once
    fx_tp_day, fx_z_day = {}, {}
    for d in range(1, 43):
        t = fuxi_ensmean_day(init_str, d, 'tp')
        z = fuxi_ensmean_day(init_str, d, 5)
        if t is not None: fx_tp_day[d] = t
        if z is not None: fx_z_day[d] = z

    for (wname, ds, de) in weeks:
        dates = pd.date_range(start=init_date, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-15']
        if not valid:
            continue
        try:
            o_tp = era_week(era_tp_raw, valid)
            o_z = era_week(era_z_raw, valid)
        except Exception as e:
            print(f"   ERA5 week {wname} fail: {e}", flush=True); continue

        # ---- forecast weekly-mean fields (ensemble means), on target grid, land-masked ----
        f_tp, f_z = {}, {}
        # SPIRE
        if sp_tp is not None:
            f_tp['SPIRE'] = to_grid(sp_tp.isel(step=slice(ds - 1, de)).mean('step'))
            f_z['SPIRE'] = to_grid(sp_z.isel(step=slice(ds - 1, de)).mean('step'))
        # FuXi (mean over the week's daily ens-means)
        tdays = [fx_tp_day[d] for d in range(ds, de + 1) if d in fx_tp_day]
        zdays = [fx_z_day[d] for d in range(ds, de + 1) if d in fx_z_day]
        if tdays: f_tp['FuXi'] = to_grid(xr.concat(tdays, dim='t').mean('t'))
        if zdays: f_z['FuXi'] = to_grid(xr.concat(zdays, dim='t').mean('t'))
        # ECMWF / NCEP (de-accumulate tp; mean step for gh)
        for nm, tp, z in [('ECMWF', ec_tp, ec_z), ('NCEP', nc_tp, nc_z)]:
            if tp is not None and tp.sizes.get('step', 0) >= de:
                f_tp[nm] = to_grid(weekly_mean_cumulative(tp, ds, de))
            if z is not None and z.sizes.get('step', 0) >= de:
                f_z[nm] = to_grid(z.isel(step=slice(ds - 1, de)).mean('step'))

        # ---- sanity gate (week 1 domain means) ----
        if wname == 'Week 1':
            wAll = get_cosine_latitude_weights(o_tp.lat.values)
            sanity.append(dict(init=init_date, ERA5_tp=float(o_tp.weighted(wAll).mean(['lat', 'lon'])),
                               **{f'{m}_tp': float(f_tp[m].weighted(wAll).mean(['lat', 'lon'])) for m in f_tp}))

        # ---- metrics over all regions ----
        for region in REGION_BOUNDS:
            for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']:
                r = metrics(m, 'TP', region, init_date, wname, f_tp.get(m), o_tp, era_clim_tp)
                if r: rows.append(r)
                r = metrics(m, 'Z500', region, init_date, wname, f_z.get(m), o_z, era_clim_z)
                if r: rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nWROTE {OUT_CSV}  ({len(df)} rows)", flush=True)

print("\n================ PRECIP SANITY GATE (Week 1 domain-mean, mm) ================", flush=True)
sdf = pd.DataFrame(sanity)
with pd.option_context('display.width', 160, 'display.max_columns', 20):
    print(sdf.round(3).to_string(index=False), flush=True)
print("\nMeans across inits:", flush=True)
print(sdf.drop(columns='init').mean().round(3).to_string(), flush=True)
print("\nNOTE: ERA5/FuXi ~0.05-0.1 (6-h ERA5 scale); SPIRE/ECMWF/NCEP ~1-1.5 (true daily). "
      "Expected -> TP verified by PCC only.", flush=True)
print("COMPUTE_DONE", flush=True)
