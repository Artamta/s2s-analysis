"""
Appendix: single-initialization forecast and error maps for all four systems.
For a representative initialization (the wettest week-1 of JFM 2026 by domain-mean
precip), shows week-1:
  Fig 12 (precip): row 1 = ERA5 + each system's forecast (mm/day); row 2 = error
                   (forecast - ERA5).  FuXi tp unit-harmonised (mm/h -> mm/day, x24).
  Fig 13 (Z500)  : row 1 = ERA5 + each system's Z500 anomaly (m); row 2 = error.
Fields reconstructed from weekly_anom_fields.nc (no model reload). Coastline only.
"""
import numpy as np, xarray as xr, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.verification_extra import get_land_mask, mask_land

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
DATA = '/storage/raj.ayush/s2s-forecast-data'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
target_lat = np.arange(38, 5, -1.5); target_lon = np.arange(65, 100, 1.5)
LAND = get_land_mask(target_lat, target_lon)
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
FUXI_TP_FACTOR = 24.0
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree()
except Exception:
    proj = None


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=target_lat, lon=target_lon, method='linear').squeeze(), LAND)


era6 = xr.open_dataset(f'{DATA}/era5/data/era5_surface.grib', filter_by_keys={'shortName': 'tp'}, **OPEN)['tp'] * 1000.0
clim6 = to_grid(era6.mean('time'))
daily = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
f = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
inits = [str(x) for x in f['init'].values]
lat = f['lat'].values; lon = f['lon'].values
import pandas as pd


def era_daily_week(valid):
    try:
        return to_grid(daily.sel(time=slice(valid[0], valid[-1])).mean('time')).values
    except Exception:
        return None


# pick wettest week-1 (domain-mean ERA5 daily precip)
wet = []
for ii, init in enumerate(inits):
    dts = pd.date_range(start=init, periods=42)[0:7]
    valid = [d.strftime('%Y-%m-%d') for d in dts if d.strftime('%Y-%m-%d') <= '2026-05-10']
    o = era_daily_week(valid)
    wet.append(np.nan if o is None else np.nanmean(o))
istar = int(np.nanargmax(wet))
init_v = inits[istar]
dts = pd.date_range(start=init_v, periods=42)[0:7]
valid = [d.strftime('%Y-%m-%d') for d in dts]
print(f'appendix case: init {init_v} week-1 (domain-mean precip {wet[istar]:.2f} mm/day)', flush=True)


def panel(ax, data, vmin, vmax, cmap):
    if proj:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', transform=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.5); ax.set_extent([65, 100, 5, 38], crs=proj)
    else:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    return im


def make_fig(kind):
    if kind == 'TP':
        obs = era_daily_week(valid)
        fc = {m: (f['tp_fcst'].sel(model=m).isel(init=istar, week=0).values + clim6.values)
              * (FUXI_TP_FACTOR if m == 'FuXi' else 1.0) for m in MODELS}
        fmax = np.nanpercentile(obs, 98); fc_cmap = 'YlGnBu'; flo = 0
        ename, eunit, fname = 'Precipitation', 'mm day$^{-1}$', 'fig12_appendix_precip_case'
    else:
        obs = f['z_obs'].isel(init=istar, week=0).values
        fc = {m: f['z_fcst'].sel(model=m).isel(init=istar, week=0).values for m in MODELS}
        fmax = np.nanpercentile(np.abs(obs[np.isfinite(obs)]), 98); flo = -fmax; fc_cmap = 'RdBu_r'
        ename, eunit, fname = 'Z500 anomaly', 'm', 'fig13_appendix_z500_case'
    emax = np.nanpercentile(np.abs(np.concatenate([(fc[m] - obs)[np.isfinite(fc[m] - obs)] for m in MODELS])), 98)
    cols = ['ERA5 (obs)'] + MODELS
    fig = plt.figure(figsize=(17, 7))
    for c, title in enumerate(cols):
        # row 1: forecast field (or obs in col 0)
        ax = fig.add_subplot(2, 5, c + 1, projection=proj) if proj else fig.add_subplot(2, 5, c + 1)
        data = obs if c == 0 else fc[cols[c]]
        im1 = panel(ax, data, flo, fmax, fc_cmap)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if c == 0:
            ax.text(-0.13, 0.5, 'Forecast', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
        # row 2: error (blank under obs)
        ax2 = fig.add_subplot(2, 5, 5 + c + 1, projection=proj) if proj else fig.add_subplot(2, 5, 5 + c + 1)
        if c == 0:
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'Error =\nforecast - ERA5', ha='center', va='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)
        else:
            im2 = panel(ax2, fc[cols[c]] - obs, -emax, emax, 'RdBu_r')
    cax1 = fig.add_axes([0.93, 0.55, 0.012, 0.33]); fig.colorbar(im1, cax=cax1, label=f'{ename} ({eunit})')
    cax2 = fig.add_axes([0.93, 0.12, 0.012, 0.33]); fig.colorbar(im2, cax=cax2, label=f'Error ({eunit})')
    fig.suptitle(f'{ename} forecast and error over India — init {init_v}, week-1 lead (representative case)',
                 fontsize=14, fontweight='bold', y=0.98)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{fname}.{ext}', bbox_inches='tight', dpi=300)
    print('WROTE', fname, flush=True)


make_fig('TP'); make_fig('Z500')
print('APPENDIX_MAPS_DONE', flush=True)
