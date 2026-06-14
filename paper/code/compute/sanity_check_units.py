"""
Quick sanity check: plot TP and Z500 for all models + ERA5
for ONE init date and ONE week to verify units are consistent.

Run BEFORE or AFTER script 03 — loads raw data directly.
Usage: python sanity_check_units.py
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

# ─── Config (change these to test different dates/weeks) ──────────────────────
INIT    = '2026-01-08'       # init date to check
INIT_STR= '20260108'
WEEK    = 2                  # week number (1-6)
DS, DE  = (WEEK-1)*7+1, WEEK*7   # day start/end
G       = 9.80665
DATA    = '/storage/raj.ayush/s2s-forecast-data'
OPEN    = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})

target_lat = np.arange(38, 5, -1.5)
target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)

# Load exact All India political mask
_mask_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc')
REGION_MASKS = {k: _mask_ds[k].values.astype(bool) for k in _mask_ds.data_vars}
ALL_INDIA_MASK = np.zeros((len(target_lat), len(target_lon)), dtype=bool)
for _m in REGION_MASKS.values(): ALL_INDIA_MASK |= _m
all_india_mask_da = xr.DataArray(ALL_INDIA_MASK, dims=['lat', 'lon'], coords={'lat': target_lat, 'lon': target_lon})

def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    da = mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)
    # Apply strict All India mask
    return da.where(all_india_mask_da)

def india_mean(da):
    return float(da.mean(['lat','lon'], skipna=True))

def weekly_mean_cumulative(cum, ds, de):
    days = de - ds + 1
    return (cum.isel(step=de-1)/days) if ds==1 else (cum.isel(step=de-1)-cum.isel(step=ds-2))/days

# ─── Load all models ──────────────────────────────────────────────────────────
print(f"Loading data for INIT={INIT}  Week {WEEK} (days {DS}-{DE})...\n")
results_tp, results_z = {}, {}

# ERA5 observation
print("ERA5...")
era_tp = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib',
                         filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
era_z  = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
dates = [f'2026-{1:02d}-{d:02d}' for d in range(8, 15)]  # Jan 8-14 for Week 2
obs_tp = to_grid(era_tp.sel(time=slice(f'{INIT}', f'{INIT[:7]}-{int(INIT[-2:])+DE-1:02d}')).mean('time'))
obs_z  = to_grid(era_z.sel(time=slice(f'{INIT}', f'{INIT[:7]}-{int(INIT[-2:])+DE-1:02d}')).mean('time'))
# simpler: just use date range
from datetime import datetime, timedelta
start = datetime.strptime(INIT, '%Y-%m-%d') + timedelta(days=DS-1)
end   = datetime.strptime(INIT, '%Y-%m-%d') + timedelta(days=DE-1)
obs_tp = to_grid(era_tp.sel(time=slice(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))).mean('time'))
obs_z  = to_grid(era_z.sel(time=slice(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))).mean('time'))
results_tp['ERA5'] = obs_tp
results_z['ERA5']  = obs_z
print(f"  ERA5  TP: {india_mean(obs_tp):.3f} mm/day  |  Z500: {india_mean(obs_z):.1f} gpm")

# SPIRE
print("SPIRE...")
try:
    s = xr.open_zarr(f'{DATA}/spire/spire_hindcast_jfm.zarr', group='mean_stddev').sel(reference_time=INIT)
    sp_tp = to_grid(s['precipitation_amount'].isel(step=slice(DS-1, DE)).mean('step'))
    sp_z  = to_grid(s['geopotential_height_at_isobaric_levels'].sel(isobar=50000.0).isel(step=slice(DS-1, DE)).mean('step'))
    results_tp['SPIRE'] = sp_tp
    results_z['SPIRE']  = sp_z
    print(f"  SPIRE TP: {india_mean(sp_tp):.3f} mm/day  |  Z500: {india_mean(sp_z):.1f} gpm")
except Exception as e:
    print(f"  SPIRE FAIL: {e}")

# FuXi
print("FuXi...")
try:
    td, zd = [], []
    for day in range(DS, DE+1):
        fs_tp, fs_z = [], []
        for mem in range(11):
            p = f"{DATA}/fuxi/output/{INIT_STR}/member/{mem:02d}/{day:02d}.nc"
            if os.path.exists(p):
                da = xr.open_dataset(p)['__xarray_dataarray_variable__']
                fs_tp.append(da.sel(channel='tp').squeeze())
                fs_z.append((da.sel(channel='z500') / G).squeeze())
        if fs_tp: td.append(xr.concat(fs_tp, 'm').mean('m'))
        if fs_z:  zd.append(xr.concat(fs_z,  'm').mean('m'))
    fx_tp = to_grid(xr.concat(td, 't').mean('t'))
    fx_z  = to_grid(xr.concat(zd, 't').mean('t'))
    results_tp['FuXi'] = fx_tp
    results_z['FuXi']  = fx_z
    print(f"  FuXi  TP: {india_mean(fx_tp):.3f} mm/day  |  Z500: {india_mean(fx_z):.1f} gpm")
except Exception as e:
    print(f"  FuXi FAIL: {e}")

# ECMWF
print("ECMWF...")
try:
    d = xr.open_dataset(f'{DATA}/ecmwf/data/sfc_pf_{INIT_STR}.grib',
                        filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
    ec_tp = d.mean('number') if 'number' in d.dims else d
    d = xr.open_dataset(f'{DATA}/ecmwf/data/pl_pf_{INIT_STR}.grib',
                        filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
    if 'isobaricInhPa' in d.dims: d = d.sel(isobaricInhPa=500)
    ec_z = d.mean('number') if 'number' in d.dims else d
    results_tp['ECMWF'] = to_grid(weekly_mean_cumulative(ec_tp, DS, DE))
    results_z['ECMWF']  = to_grid(ec_z.isel(step=slice(DS-1, DE)).mean('step'))
    print(f"  ECMWF TP: {india_mean(results_tp['ECMWF']):.3f} mm/day  |  Z500: {india_mean(results_z['ECMWF']):.1f} gpm")
except Exception as e:
    print(f"  ECMWF FAIL: {e}")

# NCEP
print("NCEP...")
try:
    d = xr.open_dataset(f'{DATA}/ncep/data/sfc_pf_{INIT_STR}.grib',
                        filter_by_keys={'shortName': 'tp'}, **OPEN)['tp']
    nc_tp = d.mean('number') if 'number' in d.dims else d
    d = xr.open_dataset(f'{DATA}/ncep/data/pl_pf_{INIT_STR}.grib',
                        filter_by_keys={'shortName': 'gh'}, **OPEN)['gh']
    if 'isobaricInhPa' in d.dims: d = d.sel(isobaricInhPa=500)
    nc_z = d.mean('number') if 'number' in d.dims else d
    results_tp['NCEP'] = to_grid(weekly_mean_cumulative(nc_tp, DS, DE))
    results_z['NCEP']  = to_grid(nc_z.isel(step=slice(DS-1, DE)).mean('step'))
    print(f"  NCEP  TP: {india_mean(results_tp['NCEP']):.3f} mm/day  |  Z500: {india_mean(results_z['NCEP']):.1f} gpm")
except Exception as e:
    print(f"  NCEP FAIL: {e}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
models = list(results_tp.keys())
n = len(models)
COLORS = {'ERA5':'black','SPIRE':'royalblue','FuXi':'orangered','ECMWF':'seagreen','NCEP':'purple'}

fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
fig.suptitle(f'Unit Sanity Check — INIT {INIT}  Week {WEEK} (days {DS}–{DE})\n'
             f'Expected: TP ~0-5 mm/day (India Jan dry season)  |  Z500 ~5700-5800 gpm', 
             fontsize=11, fontweight='bold')

tp_vals = [india_mean(results_tp[m]) for m in models]
z_vals  = [india_mean(results_z[m])  for m in models]

for i, m in enumerate(models):
    col = COLORS.get(m, 'gray')

    # TP map
    ax = axes[0, i]
    da = results_tp[m]
    im = ax.pcolormesh(da.lon.values, da.lat.values, da.values,
                       cmap='Blues', vmin=0, vmax=10)
    ax.set_title(f'{m}\n{tp_vals[i]:.3f} mm/day', fontsize=9, color=col, fontweight='bold')
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat') if i==0 else None
    plt.colorbar(im, ax=ax, label='mm/day', shrink=0.7)

    # Z500 map
    ax = axes[1, i]
    da = results_z[m]
    im = ax.pcolormesh(da.lon.values, da.lat.values, da.values,
                       cmap='RdBu_r', vmin=5600, vmax=5900)
    ax.set_title(f'{m}\n{z_vals[i]:.1f} gpm', fontsize=9, color=col, fontweight='bold')
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat') if i==0 else None
    plt.colorbar(im, ax=ax, label='gpm', shrink=0.7)

axes[0,0].set_ylabel('TP (mm/day)', fontsize=10)
axes[1,0].set_ylabel('Z500 (gpm)',  fontsize=10)

plt.tight_layout()
OUT = '/home/raj.ayush/s2s/s2s_anlysis/paper/results/sanity_check_units.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"\nSAVED: {OUT}")

# ─── Print summary table ──────────────────────────────────────────────────────
print(f"\n{'Model':<12} {'TP (mm/day)':>12} {'Z500 (gpm)':>12}  {'TP ok?':>8}  {'Z500 ok?':>9}")
print("-"*60)
for m in models:
    tp_ok   = '✅' if 0 < tp_vals[models.index(m)] < 15  else '❌ WRONG'
    z_ok    = '✅' if 5500 < z_vals[models.index(m)] < 6000 else '❌ WRONG'
    print(f"{m:<12} {tp_vals[models.index(m)]:>12.3f} {z_vals[models.index(m)]:>12.1f}  {tp_ok:>8}  {z_ok:>9}")
