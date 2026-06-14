"""
Ensemble mean + spread (sigma) per system, for probabilistic verification under
a Gaussian-forecast framework (fair across systems with very different member
counts: SPIRE mean+stddev, FuXi 11, ECMWF 100, NCEP 15).

For each init x lead-week x land point and each variable (TP mm/day, Z500 m,
T2M K) we store the ensemble mean (mu) and spread (sigma):
  * SPIRE : provider mean and stddev (weekly-mean).
  * others: mean and std across members of the weekly-mean field.
Plus the verifying ERA5 weekly field. Output: analysis/prob_fields.nc
Also stores daily REGIONAL ensemble mean+spread (all inits) for the event
time-series figure: analysis/prob_daily_regional.npz

Run with argument "test" to process only the first init.
"""
import os, sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
G = 9.80665
TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
target_lat = np.arange(38, 5, -1.5); target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)
ny, nx = len(target_lat), len(target_lon)
init_dates = ['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
              '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26', '2026-03-05',
              '2026-03-12', '2026-03-19', '2026-03-26']
if TEST: init_dates = init_dates[:1]
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
# IMD 4 homogeneous regions — proper state-boundary masks (no overlaps)
_mask_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc')
REGION_MASKS = {k: _mask_ds[k].values.astype(bool) for k in _mask_ds.data_vars}
ALL_INDIA_MASK = np.zeros((ny, nx), dtype=bool)
for _m in REGION_MASKS.values(): ALL_INDIA_MASK |= _m
REGIONS = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGS = REGIONS


def crop(da):
    la = 'latitude' if 'latitude' in da.dims else 'lat'
    lo = 'longitude' if 'longitude' in da.dims else 'lon'
    lasl = slice(40, 3) if float(da[la][0]) > float(da[la][-1]) else slice(3, 40)
    losl = slice(60, 102) if float(da[lo][0]) < float(da[lo][-1]) else slice(102, 60)
    return da.sel({la: lasl, lo: losl})


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear'), LAND)


def wmean_cum(cum, ds, de):
    days = de - ds + 1
    return (cum.isel(step=de - 1) / days) if ds == 1 else (cum.isel(step=de - 1) - cum.isel(step=ds - 2)) / days


# ERA5 truth
era_tp = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_tp.nc')['tp']
era_t2 = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_t2m.nc')['t2m']
era_z = crop(xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G)


def era_week_daily(daily, valid):
    try:
        return to_grid(daily.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


def era_week_z(valid):
    try:
        return to_grid(era_z.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


def ms(da, mdim):
    """ensemble mean, std over member dim (ddof=1); returns (mu, sigma) gridded."""
    mu = to_grid(da.mean(mdim))
    sig = to_grid(da.std(mdim, ddof=1))
    return mu, sig


from scipy.stats import norm
SIG_FLOOR = {'tp': 0.05, 'z': 1.0, 't2': 0.1}  # avoid sigma=0 (degenerate ensembles)


def crps_gauss(mu, sig, y):
    """closed-form CRPS of a Gaussian(mu,sig) forecast vs scalar/array obs y."""
    sig = np.maximum(sig, 1e-6)
    w = (y - mu) / sig
    return sig * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))


# storage: gridded weekly mu/sig per model/var + obs
shape = (len(MODELS), len(init_dates), len(weeks), ny, nx)
F = {v: {'mu': np.full(shape, np.nan), 'sig': np.full(shape, np.nan)} for v in ['tp', 'z', 't2']}
O = {v: np.full((len(init_dates), len(weeks), ny, nx), np.nan) for v in ['tp', 'z', 't2']}
# daily regional: (model, init, 42, region) mean & spread for z and tp and t2
DR = {v: {'mu': np.full((len(MODELS), len(init_dates), 42, len(REGS)), np.nan),
          'sig': np.full((len(MODELS), len(init_dates), 42, len(REGS)), np.nan)} for v in ['tp', 'z', 't2']}

# --- gridded daily climatology (mean+std) for CRPSS, and event thresholds ---
CLIM = {'tp': (to_grid(era_tp.mean('time')), to_grid(era_tp.std('time'))),
        't2': (to_grid(era_t2.mean('time')), to_grid(era_t2.std('time'))),
        'z': (to_grid(era_z.mean('time')), to_grid(era_z.std('time')))}
THR_TP = 1.0                                   # heavy/wet day: precip > 1 mm/day
THR_T2_COLD = to_grid(era_t2.quantile(0.33, 'time'))   # cold day: T2M below climatological lower tercile

# weekly accumulators per (variable, model, init, week, region): daily-level, fair across systems
VARS3 = ['tp', 'z', 't2']
ACC = {k: np.zeros((len(VARS3), len(MODELS), len(init_dates), len(weeks), len(REGS))) for k in
       ('crps', 'se', 'sig', 'crps_clim', 'cnt')}
# reliability: per event, per model, 10 probability bins -> (sum outcome, count, sum prob)
NB = 10
REL = {ev: {m: np.zeros((3, NB)) for m in MODELS} for ev in ('tp_wet', 't2_cold')}


def _region_mask_da(rg):
    """Return xarray DataArray mask for region rg on target grid."""
    m = ALL_INDIA_MASK if rg == 'All India' else REGION_MASKS[rg]
    return xr.DataArray(m, dims=['lat', 'lon'],
                        coords={'lat': target_lat, 'lon': target_lon})


def rweights(g, rg):
    mda = _region_mask_da(rg)
    s = g if mda is None else g.where(mda)
    w = xr.DataArray(np.cos(np.deg2rad(target_lat)), dims=['lat'],
                     coords={'lat': target_lat})
    return s, w


def cosmean(g, rg):
    s, w = rweights(g, rg)
    return float(s.weighted(w).mean(['lat', 'lon']))


def obs_day(daily, date):
    try:
        o = to_grid(daily.sel(time=slice(date, date)).mean('time'))
        return o if not bool(np.isnan(o).all()) else None
    except Exception:
        return None


def accum_reliability(ev, m, prob_g, outcome_g):
    p = np.asarray(prob_g).ravel(); y = np.asarray(outcome_g).ravel()
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    idx = np.clip((p * NB).astype(int), 0, NB - 1)
    for b in range(NB):
        sel = idx == b
        if sel.any():
            REL[ev][m][0, b] += y[sel].sum(); REL[ev][m][1, b] += sel.sum(); REL[ev][m][2, b] += p[sel].sum()


def regional_scalars(mu_g, sig_g):
    """area-mean of gridded mu and sigma per region (cosine-weighted, mask-based)."""
    out_mu, out_sig = [], []
    w = xr.DataArray(np.cos(np.deg2rad(target_lat)), dims=['lat'],
                     coords={'lat': target_lat})
    for rg in REGS:
        mda = _region_mask_da(rg)
        m = mu_g  if mda is None else mu_g.where(mda)
        s = sig_g if mda is None else sig_g.where(mda)
        out_mu.append(float(m.weighted(w).mean(['lat', 'lon'])))
        out_sig.append(float(s.weighted(w).mean(['lat', 'lon'])))
    return out_mu, out_sig


def fuxi_member_day(init_str, mem, day, ch):
    p = f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
    if not os.path.exists(p): return None
    da = xr.open_dataset(p)['__xarray_dataarray_variable__']
    da = da.sel(channel='tp') if ch == 'tp' else (da.sel(channel='t2m') if ch == 't2' else da.isel(channel=5) / G)
    for dd in list(da.dims):
        if dd not in ('lat', 'lon', 'latitude', 'longitude'): da = da.mean(dd)
    return crop(da)


for ii, init in enumerate(init_dates):
    init_str = pd.to_datetime(init).strftime('%Y%m%d'); print(f"=== INIT {init} ===", flush=True)
    # ---- SPIRE (mean + stddev) ----
    try:
        s = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr", group='mean_stddev').sel(reference_time=init)
        sp = {'tp': (crop(s['precipitation_amount']), crop(s['precipitation_amount_stddev'])),
              'z': (crop(s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0)),
                    crop(s['geopotential_height_at_isobaric_levels_stddev'].sel(isobar=50000.0))),
              't2': (crop(s['air_temperature']), crop(s['air_temperature_stddev']))}
    except Exception as e:
        sp = None; print("  SPIRE fail", e, flush=True)
    # ---- ECMWF / NCEP members ----
    op = {}
    for mdl in ['ecmwf', 'ncep']:
        base = f'{DATA}/{mdl}/data'; ent = {}
        try: ent['tp'] = crop(xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'])
        except Exception as e: print(f"  {mdl} tp fail", e, flush=True)
        try:
            z = xr.open_dataset(f'{base}/pl_pf_{init_str}.grib', filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
            ent['z'] = crop(z.sel(isobaricInhPa=500) if 'isobaricInhPa' in z.dims else z)
        except Exception as e: print(f"  {mdl} z fail", e, flush=True)
        try:
            mx = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'mx2t6'}, **OPEN)['mx2t6']
            mn = xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib', filter_by_keys={'shortName': 'mn2t6'}, **OPEN)['mn2t6']
            ent['t2'] = crop((mx + mn) / 2.0)
        except Exception as e: print(f"  {mdl} t2 fail", e, flush=True)
        op[mdl.upper()] = ent
    # ---- FuXi members: preload all member/day fields per variable ----
    fx = {'tp': {}, 'z': {}, 't2': {}}
    for v in ['tp', 'z', 't2']:
        for mem in range(11):
            for day in range(1, 43):
                f = fuxi_member_day(init_str, mem, day, v)
                if f is not None: fx[v].setdefault(day, []).append(f.assign_coords(member=mem))
    print("  loaded; computing weekly mu/sig", flush=True)

    for wi, (wn, ds, de) in enumerate(weeks):
        dts = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dts if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid: continue
        otp = era_week_daily(era_tp, valid); ot2 = era_week_daily(era_t2, valid); oz = era_week_z(valid)
        for v, o in [('tp', otp), ('z', oz), ('t2', ot2)]:
            if o is not None: O[v][ii, wi] = o.values
        for mi, m in enumerate(MODELS):
            for v in ['tp', 'z', 't2']:
                mu = sig = None
                if m == 'SPIRE' and sp is not None:
                    muf, sigf = sp[v]
                    mu = to_grid(muf.isel(step=slice(ds - 1, de)).mean('step'))
                    sig = to_grid(sigf.isel(step=slice(ds - 1, de)).mean('step'))
                elif m in ('ECMWF', 'NCEP') and v in op[m]:
                    arr = op[m][v]
                    if arr.sizes.get('step', 0) >= de:
                        wk = wmean_cum(arr, ds, de) if v == 'tp' else arr.isel(step=slice(ds - 1, de)).mean('step')
                        mu, sig = ms(wk, 'number')
                elif m == 'FuXi':
                    perday = []
                    for day in range(ds, de + 1):
                        if day in fx[v]: perday.append(xr.concat(fx[v][day], 'member'))
                    if perday:
                        wk = xr.concat(perday, 't').mean('t')  # weekly-mean per member (member,lat,lon)
                        if v == 'tp': wk = wk * 24.0
                        mu, sig = ms(wk, 'member')
                if mu is not None:
                    F[v]['mu'][mi, ii, wi] = mu.values; F[v]['sig'][mi, ii, wi] = sig.values

    # ---- daily loop: fair (daily-level) CRPS/SSR + reliability + regional spread ----
    for di in range(42):
        day = di + 1; wk = min(di // 7, 5)
        date = (pd.to_datetime(init) + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        if date > '2026-05-10': break
        obs = {'tp': obs_day(era_tp, date), 't2': obs_day(era_t2, date), 'z': obs_day(era_z, date)}
        for mi, m in enumerate(MODELS):
            for vi, v in enumerate(VARS3):
                mu_g = sig_g = None
                if m == 'SPIRE' and sp is not None:
                    mu_g = to_grid(sp[v][0].isel(step=di)); sig_g = to_grid(sp[v][1].isel(step=di))
                elif m in ('ECMWF', 'NCEP') and v in op[m]:
                    arr = op[m][v]
                    if arr.sizes.get('step', 0) > di:
                        daily_c = ((arr.isel(step=di) - arr.isel(step=di - 1)) if di > 0 else arr.isel(step=di)) if v == 'tp' else arr.isel(step=di)
                        mu_g, sig_g = ms(daily_c, 'number')
                elif m == 'FuXi' and day in fx[v]:
                    ens = xr.concat(fx[v][day], 'member')
                    if v == 'tp': ens = ens * 24.0
                    mu_g, sig_g = ms(ens, 'member')
                if mu_g is None:
                    continue
                sig_g = sig_g.clip(min=SIG_FLOOR[v])
                # regional spread time-series
                for ri, rg in enumerate(REGS):
                    DR[v]['mu'][mi, ii, di, ri] = cosmean(mu_g, rg)
                    DR[v]['sig'][mi, ii, di, ri] = cosmean(sig_g, rg)
                o = obs[v]
                if o is None:
                    continue
                cm, cs = CLIM[v]
                crps_g = crps_gauss(mu_g.values, sig_g.values, o.values)
                crps_c = crps_gauss(cm.values, np.maximum(cs.values, SIG_FLOOR[v]), o.values)
                se_g = (mu_g - o) ** 2
                crps_da = mu_g.copy(data=crps_g); crpsc_da = mu_g.copy(data=crps_c); se_da = se_g
                for ri, rg in enumerate(REGS):
                    cval = cosmean(crps_da, rg)
                    if np.isfinite(cval):
                        ACC['crps'][vi, mi, ii, wk, ri] += cval
                        ACC['crps_clim'][vi, mi, ii, wk, ri] += cosmean(crpsc_da, rg)
                        ACC['se'][vi, mi, ii, wk, ri] += cosmean(se_da, rg)
                        ACC['sig'][vi, mi, ii, wk, ri] += cosmean(sig_g, rg)
                        ACC['cnt'][vi, mi, ii, wk, ri] += 1
                # reliability (All India grid points), only at the All-India domain
                if v == 'tp':
                    prob = 1 - norm.cdf((THR_TP - mu_g.values) / sig_g.values)
                    accum_reliability('tp_wet', m, prob, (o.values > THR_TP).astype(float))
                elif v == 't2':
                    prob = norm.cdf((THR_T2_COLD.values - mu_g.values) / sig_g.values)
                    accum_reliability('t2_cold', m, prob, (o.values < THR_T2_COLD.values).astype(float))

# ---------- write weekly probabilistic skill CSV ----------
rows = []
VLAB3 = {'tp': 'TP', 'z': 'Z500', 't2': 'T2M'}
for vi, v in enumerate(VARS3):
    for mi, m in enumerate(MODELS):
        for ii, init in enumerate(init_dates):
            for wi, (wn, _, _) in enumerate(weeks):
                for ri, rg in enumerate(REGS):
                    n = ACC['cnt'][vi, mi, ii, wi, ri]
                    if n == 0:
                        continue
                    crps = ACC['crps'][vi, mi, ii, wi, ri] / n
                    crps_c = ACC['crps_clim'][vi, mi, ii, wi, ri] / n
                    rmse = np.sqrt(ACC['se'][vi, mi, ii, wi, ri] / n)
                    spread = ACC['sig'][vi, mi, ii, wi, ri] / n
                    rows.append(dict(variable=VLAB3[v], region=rg, week=wn, init_date=init, model=m,
                                     crps=crps, crps_clim=crps_c,
                                     crpss=1 - crps / crps_c if crps_c > 0 else np.nan,
                                     rmse=rmse, spread=spread,
                                     ssr=spread / rmse if rmse > 0 else np.nan))
pd.DataFrame(rows).to_csv(f'{ADIR}/prob_skill{"_test" if TEST else ""}.csv', index=False)

# reliability summary
rel_out = {}
for ev in REL:
    for m in MODELS:
        rel_out[f'{ev}__{m}'] = REL[ev][m]
np.savez_compressed(f'{ADIR}/reliability{"_test" if TEST else ""}.npz', nbins=NB, models=MODELS, **rel_out)
np.savez_compressed(f'{ADIR}/prob_daily_regional{"_test" if TEST else ""}.npz',
                    models=MODELS, inits=init_dates, regions=REGS,
                    **{f'{v}_{k}': DR[v][k] for v in DR for k in ('mu', 'sig')})
# ensemble-mean weekly fields (for maps)
xr.Dataset({f'{v}_mu': (('model', 'init', 'week', 'lat', 'lon'), F[v]['mu']) for v in F} |
           {f'{v}_obs': (('init', 'week', 'lat', 'lon'), O[v]) for v in O},
           coords={'model': MODELS, 'init': init_dates, 'week': [w[0] for w in weeks],
                   'lat': target_lat, 'lon': target_lon}).to_netcdf(f'{ADIR}/prob_fields{"_test" if TEST else ""}.nc')
print('WROTE prob_skill.csv, reliability.npz, prob_daily_regional.npz, prob_fields.nc', flush=True)
print('PROB_FIELDS_DONE', flush=True)
