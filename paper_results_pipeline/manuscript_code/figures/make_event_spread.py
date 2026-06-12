"""
Event time-series with ensemble spread (reference: TianXing-S2S Fig 3C / S7C).
For the most anomalous circulation init of JFM 2026, the All-India area-averaged
anomaly is plotted versus forecast day (1-42): observed (ERA5, black) against
each system's ensemble mean (colour) with its +/-1 sigma spread band. One column
per system; row 1 = Z500 anomaly (m), row 2 = precipitation anomaly (mm/day).
Reads analysis/prob_daily_regional.npz + ERA5 files. -> paper/figs/fig18_event_spread.{pdf,png}
"""
import sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''}); G = 9.80665
tlat = np.arange(38, 5, -1.5); tlon = np.arange(65, 100, 1.5); LAND = get_land_mask(tlat, tlon)
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'font.family': 'DejaVu Sans', 'savefig.dpi': 300, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': ':'})
AI = (5., 38., 65., 100.)


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=tlat, lon=tlon, method='linear'), LAND)


def aimean(g):
    a, b, c, d = AI; s = g.sel(lat=slice(b, a), lon=slice(c, d)); w = np.cos(np.deg2rad(s.lat))
    return float(s.weighted(w).mean(['lat', 'lon']))


era_tp = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
era_z = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
clim_z = aimean(to_grid(era_z.mean('time'))); clim_tp = aimean(to_grid(era_tp.mean('time')))

r = np.load(f'{ADIR}/prob_daily_regional.npz', allow_pickle=True)
MODELS = list(r['models']); inits = list(r['inits']); regions = list(r['regions'])
ri_ai = regions.index('All India')


def obs_series(daily, init, raw_is_z=False):
    out = []
    for di in range(42):
        date = (pd.to_datetime(init) + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        try:
            g = to_grid((era_z if raw_is_z else daily).sel(time=slice(date, date)).mean('time'))
            out.append(aimean(g))
        except Exception:
            out.append(np.nan)
    return np.array(out)


# pick most anomalous init by observed week-1 Z500 anomaly amplitude
amp = []
for init in inits:
    z = obs_series(None, init, raw_is_z=True)[:7] - clim_z
    amp.append(np.sqrt(np.nanmean(z ** 2)))
istar = int(np.nanargmax(amp)); init_v = inits[istar]
print(f'event init {init_v} (wk1 Z500 anomaly RMS {amp[istar]:.1f} m)', flush=True)

obs_z = obs_series(None, init_v, raw_is_z=True) - clim_z
obs_tp = obs_series(era_tp, init_v) - clim_tp
days = np.arange(1, 43)

fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
for c, m in enumerate(MODELS):
    for rrow, (vkey, obs, clim, unit, vlab) in enumerate([
            ('z', obs_z, clim_z, 'm', 'Z500 anomaly'),
            ('tp', obs_tp, clim_tp, 'mm day$^{-1}$', 'Precip anomaly')]):
        a = ax[rrow, c]
        mu = r[f'{vkey}_mu'][MODELS.index(m), istar, :, ri_ai] - clim
        sg = r[f'{vkey}_sig'][MODELS.index(m), istar, :, ri_ai]
        a.plot(days, obs, 'k-', lw=2.2, label='ERA5', zorder=5)
        a.plot(days, mu, '-', color=COL[m], lw=2, label=f'{LAB[m]} mean')
        a.fill_between(days, mu - sg, mu + sg, color=COL[m], alpha=0.25, lw=0, label='$\\pm1\\sigma$ spread')
        a.axhline(0, color='0.5', lw=0.8)
        if rrow == 0:
            a.set_title(LAB[m])
        if c == 0:
            a.set_ylabel(f'{vlab} ({unit})', fontweight='bold')
        if rrow == 1:
            a.set_xlabel('forecast day')
        if c == 0 and rrow == 0:
            a.legend(loc='upper left', fontsize=8.5)
fig.suptitle(f'Event forecast with ensemble spread: All-India anomaly, init {init_v} — JFM 2026',
             fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.005, 'Black: ERA5; colour: ensemble mean; shaded: $\\pm1\\sigma$ ensemble spread. '
         'Narrow bands (e.g.\\ FuXi-S2S) indicate an overconfident ensemble.', ha='center', fontsize=10, style='italic')
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
for ext in ('pdf', 'png'): fig.savefig(f'{FIGDIR}/fig18_event_spread.{ext}', bbox_inches='tight')
print('WROTE fig18_event_spread', flush=True)
print('EVENT_TS_DONE', flush=True)
