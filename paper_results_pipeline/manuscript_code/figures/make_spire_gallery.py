"""
SPIRE distribution gallery (reference: TianXing-S2S Figs S11/S12 member galleries).
SPIRE does not ship individual members but provides 9 distribution percentiles
(1,5,10,20,50,80,90,95,99). For the week-6 forecast of the strongest-anomaly
init, we show the ERA5 anomaly, the SPIRE ensemble-mean anomaly, and the nine
percentile anomaly maps over India -- the spread analogue of a member gallery.
T2M and precipitation (the two fields SPIRE provides percentiles for).
-> paper/figs/fig26_spire_gallery_t2m.{pdf,png}, fig27_spire_gallery_precip.{pdf,png}
"""
import sys, warnings
import numpy as np, xarray as xr
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
tlat = np.arange(38, 5, -1.5); tlon = np.arange(65, 100, 1.5); LAND = get_land_mask(tlat, tlon)
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree()
except Exception:
    proj = None
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.dpi': 300})
WK6 = slice(35, 42)  # forecast days 36-42


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=tlat, lon=tlon, method='linear'), LAND)


pf = xr.open_dataset(f'{ADIR}/prob_fields.nc'); inits = [str(x) for x in pf['init'].values]
era_tp = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
era_t2 = xr.open_dataset(f'{ADIR}/era5_daily_t2m.nc')['t2m']
CLIM = {'t2': to_grid(era_t2.mean('time')), 'tp': to_grid(era_tp.mean('time'))}
pctl_grp = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr", group='percentiles')
mean_grp = xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr", group='mean_stddev')
PCTLS = [1, 5, 10, 20, 50, 80, 90, 95, 99]
VINFO = {'t2': ('2-m temperature', 'K', 'RdBu_r', 'air_temperature_pctl', 'air_temperature', 't2_obs'),
         'tp': ('precipitation', 'mm day$^{-1}$', 'BrBG', 'precipitation_amount_pctl', 'precipitation_amount', 'tp_obs')}
lat = pf['lat'].values; lon = pf['lon'].values


def make(v, fname):
    vname, unit, cmap, pkey, mkey, okey = VINFO[v]
    clim = CLIM[v]
    amp = np.array([np.sqrt(np.nanmean((pf[okey].isel(init=i, week=5).values - clim.values) ** 2)) for i in range(len(inits))])
    istar = int(np.nanargmax(amp)); init_v = inits[istar]
    obs_an = pf[okey].isel(init=istar, week=5).values - clim.values
    ens_mean = to_grid(mean_grp[mkey].sel(reference_time=init_v).isel(step=WK6).mean('step')).values - clim.values
    pf_p = pctl_grp[pkey].sel(reference_time=init_v).isel(step=WK6).mean('step')  # (percentile,lat,lon)
    pmaps = [to_grid(pf_p.isel(percentile=k)).values - clim.values for k in range(len(PCTLS))]
    vmax = np.nanpercentile(np.abs(obs_an[np.isfinite(obs_an)]), 97)

    panels = [('ERA5 (obs)', obs_an), ('SPIRE ens. mean', ens_mean)] + \
             [(f'P{p}', pmaps[k]) for k, p in enumerate(PCTLS)]
    fig = plt.figure(figsize=(15, 9))
    for i, (title, data) in enumerate(panels):
        ax = fig.add_subplot(3, 4, i + 1, projection=proj) if proj else fig.add_subplot(3, 4, i + 1)
        if proj:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto', transform=proj)
            ax.add_feature(cfeature.COASTLINE, lw=0.5); ax.set_extent([65, 100, 5, 38], crs=proj)
        else:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_title(title, fontsize=11, fontweight='bold' if i < 2 else 'normal',
                     color='k' if i < 2 else '0.25')
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5]); fig.colorbar(im, cax=cax, label=f'{vname} anomaly ({unit})')
    fig.suptitle(f'SPIRE week-6 forecast distribution for {vname} — init {init_v} '
                 f'(ERA5, ensemble mean, and the 1st–99th percentiles)', fontsize=13.5, fontweight='bold', y=0.97)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{fname}.{ext}', bbox_inches='tight')
    plt.close(fig); print(f'WROTE {fname} (init {init_v})', flush=True)


make('t2', 'fig26_spire_gallery_t2m')
make('tp', 'fig27_spire_gallery_precip')
print('SPIRE_GALLERY_DONE', flush=True)
