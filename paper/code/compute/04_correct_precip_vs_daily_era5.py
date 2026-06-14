"""
Re-score TOTAL PRECIPITATION against a TRUE 24-h daily ERA5 total, without
re-loading the forecast ensembles. Uses:
  * weekly_anom_fields.nc  -> stored model TP anomalies (= raw_weekly - clim_6h)
  * era5_surface.grib      -> recompute clim_6h to recover raw model weekly TP
  * era5_daily_tp.nc       -> TRUE daily-total ERA5 (mm/day), built from ARCO hourly

Recovered raw model TP (~0.8 mm/day true-daily scale) is then verified against
the true daily ERA5, giving LEGITIMATE TP PCC + RMSE + Bias. Prints FuXi-vs-others
before/after so the artifact can be quantified.

Output: analysis/skill_tp_corrected.csv
"""
import sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias
from utils.verification_extra import get_land_mask, mask_land, bootstrap_ci

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
DATA = '/storage/raj.ayush/s2s-forecast-data'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)
weeks = [('Week 1', 1, 7), ('Week 2', 8, 14), ('Week 3', 15, 21),
         ('Week 4', 22, 28), ('Week 5', 29, 35), ('Week 6', 36, 42)]
# IMD 4 homogeneous regions — proper state-boundary masks (no overlaps)
_mask_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc')
REGION_MASKS = {k: _mask_ds[k].values.astype(bool) for k in _mask_ds.data_vars}
ALL_INDIA_MASK = np.zeros((len(target_lat), len(target_lon)), dtype=bool)
for _m in REGION_MASKS.values(): ALL_INDIA_MASK |= _m
REGIONS = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME']


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)


def regional(f, rg):
    if rg == 'All India':
        mask = xr.DataArray(ALL_INDIA_MASK, dims=['lat', 'lon'],
                            coords={'lat': target_lat, 'lon': target_lon})
    else:
        mask = xr.DataArray(REGION_MASKS[rg], dims=['lat', 'lon'],
                            coords={'lat': target_lat, 'lon': target_lon})
    return f.where(mask)


# --- recover clim_6h to undo the stored anomaly ---
era6 = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
clim6 = to_grid(era6.mean('time'))

# --- TRUE daily ERA5 (mm/day) from ARCO build ---
ERA5_DAILY_TP = '/storage/raj.ayush/s2s-forecast-data/era5/daily/era5_daily_tp.nc'
daily = xr.open_dataset(ERA5_DAILY_TP)['tp']  # dims time,lat,lon (mm/day)
clim_daily = to_grid(daily.mean('time'))

fields = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
init_dates = [str(x) for x in fields['init'].values]
wk_names = [str(x) for x in fields['week'].values]


def era_daily_week(valid):
    try:
        return to_grid(daily.sel(time=slice(valid[0], valid[-1])).mean('time'))
    except Exception:
        return None


rows = []
for ii, init in enumerate(init_dates):
    # persistence reference = observed daily week immediately before init
    pre = pd.date_range(end=pd.to_datetime(init) - pd.Timedelta(days=1), periods=7)
    pre_field = era_daily_week([d.strftime('%Y-%m-%d') for d in pre])
    for wi, (wn, ds, de) in enumerate(weeks):
        dates = pd.date_range(start=init, periods=42)[ds - 1:de]
        valid = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') <= '2026-05-10']
        if not valid:
            continue
        o = era_daily_week(valid)
        if o is None or np.isnan(o).all():
            continue
        # persistence row(s)
        if pre_field is not None and not np.isnan(pre_field).all():
            for rg in REGIONS:
                fr, orr, cr = regional(pre_field, rg), regional(o, rg), regional(clim_daily, rg)
                w = get_cosine_latitude_weights(target_lat)
                try:
                    rows.append(dict(variable='TP', region=rg, week=wn, init_date=init, model='Persistence',
                                     pcc=calc_wmo_acc(fr, orr, cr, w), rmse=calc_wmo_rmse(fr, orr, w),
                                     bias=calc_wmo_bias(fr, orr, w),
                                     fcst_mean=float(fr.weighted(w).mean(['lat', 'lon'])),
                                     obs_mean=float(orr.weighted(w).mean(['lat', 'lon']))))
                except Exception:
                    pass
        # recover raw weekly-mean TP per base system, then rebuild MME from them.
        # FuXi-S2S 'tp' is a mean precip RATE (~mm h^-1); x24 -> mm day^-1 (unit
        # harmonization; the other systems are already mm day^-1). PCC is
        # scale-invariant so FuXi's correlation is unchanged; only its RMSE/bias
        # (and the MME that averages it) are corrected here.
        FUXI_TP_FACTOR = 24.0
        raws = {}
        for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']:
            anom = fields['tp_fcst'].sel(model=m).isel(init=ii, week=wi)
            if np.isnan(anom).all():
                continue
            raws[m] = (anom + clim6) * (FUXI_TP_FACTOR if m == 'FuXi' else 1.0)
        if len(raws) >= 3:
            raws['MME'] = sum(raws.values()) / len(raws)
        for m in MODELS:
            if m not in raws:
                continue
            raw = raws[m]
            for rg in REGIONS:
                fr, orr, cr = regional(raw, rg), regional(o, rg), regional(clim_daily, rg)
                w = get_cosine_latitude_weights(target_lat)
                try:
                    rows.append(dict(variable='TP', region=rg, week=wn, init_date=init, model=m,
                                     pcc=calc_wmo_acc(fr, orr, cr, w), rmse=calc_wmo_rmse(fr, orr, w),
                                     bias=calc_wmo_bias(fr, orr, w),
                                     fcst_mean=float(fr.weighted(w).mean(['lat', 'lon'])),
                                     obs_mean=float(orr.weighted(w).mean(['lat', 'lon']))))
                except Exception as e:
                    print('metric fail', m, rg, wn, init, e, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{ADIR}/skill_tp_corrected.csv', index=False)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
print(f"\nWROTE skill_tp_corrected.csv ({len(df)} rows)\n", flush=True)
print("### CORRECTED TP PCC vs TRUE daily ERA5 (All India) ###")
print(f"{'Model':<8}" + "".join(f"  Wk{w}" for w in range(1, 7)))
for m in MODELS:
    print(f"{m:<8}" + "".join(f"  {df[(df.region=='All India')&(df.model==m)&(df.wk==w)]['pcc'].mean():.2f}" for w in range(1, 7)))
print("\n### CORRECTED TP RMSE (mm/day, All India) — now LEGITIMATE ###")
for m in MODELS:
    print(f"{m:<8}" + "".join(f"  {df[(df.region=='All India')&(df.model==m)&(df.wk==w)]['rmse'].mean():.2f}" for w in range(1, 7)))
print("\n### domain-mean precip check (mm/day) ###")
print("ERA5 daily truth obs_mean Wk1:", round(df[(df.region=='All India')&(df.wk==1)]['obs_mean'].mean(), 3))
for m in MODELS:
    print(f"  {m} fcst_mean Wk1:", round(df[(df.region=='All India')&(df.model==m)&(df.wk==1)]['fcst_mean'].mean(), 3))
print("CORRECTED_DONE", flush=True)
