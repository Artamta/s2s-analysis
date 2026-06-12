"""
Regional breakdown of the three week-6 case studies: for each case (same init
as fig20-22), the area-averaged anomaly versus forecast day is shown for the
whole-India domain and each of the four IMD homogeneous regions (5 panels),
with the observations (black) and the four systems' ensemble means; SPIRE's
+/-1 sigma spread is shaded. Reads prob_fields.nc (to pick inits, consistently
with fig20-22), prob_daily_regional.npz, and ERA5. -> fig23-25_regional_case_*.
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
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
RB = {'All India': (5., 38., 65., 100.), 'northwest_india': (22., 38., 68., 82.),
      'central_india': (18., 28., 72., 89.), 'south_peninsula': (8., 20., 72., 85.),
      'east_northeast_india': (20., 30., 85., 98.)}
RLAB = {'All India': 'All India', 'northwest_india': 'Northwest', 'central_india': 'Central',
        'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': ':'})


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=tlat, lon=tlon, method='linear'), LAND)


def rmean(g, rg):
    a, b, c, d = RB[rg]; s = g.sel(lat=slice(b, a), lon=slice(c, d)); w = np.cos(np.deg2rad(s.lat))
    return float(s.weighted(w).mean(['lat', 'lon']))


f = xr.open_dataset(f'{ADIR}/prob_fields.nc'); inits = [str(x) for x in f['init'].values]
era_tp = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
era_t2 = xr.open_dataset(f'{ADIR}/era5_daily_t2m.nc')['t2m']
era_z = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
SRC = {'tp': era_tp, 't2': era_t2, 'z': era_z}
CLIMr = {v: {rg: rmean(to_grid(SRC[v].mean('time')), rg) for rg in RB} for v in SRC}
r = np.load(f'{ADIR}/prob_daily_regional.npz', allow_pickle=True)
regions = list(r['regions'])
KEYNC = {'tp': 'tp_obs', 't2': 't2_obs', 'z': 'z_obs'}
VINFO = {'t2': ('2-m temperature', 'K'), 'tp': ('precipitation', 'mm day$^{-1}$'), 'z': ('Z500', 'm')}


def obs_region_series(v, init, rg):
    out = []
    for di in range(42):
        date = (pd.to_datetime(init) + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        try:
            out.append(rmean(to_grid(SRC[v].sel(time=slice(date, date)).mean('time')), rg) - CLIMr[v][rg])
        except Exception:
            out.append(np.nan)
    return np.array(out)


USED = set()


def make(v, fname, idx):
    clim_g = to_grid(SRC[v].mean('time')).values
    amp = np.array([np.sqrt(np.nanmean((f[KEYNC[v]].isel(init=i, week=5).values - clim_g) ** 2)) for i in range(len(inits))])
    istar = next(int(i) for i in np.argsort(amp)[::-1] if i not in USED); USED.add(istar); init_v = inits[istar]
    vname, unit = VINFO[v]; days = np.arange(1, 43)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True)
    for ci, rg in enumerate(RB):
        a = ax[ci]; ri = regions.index(rg)
        obs = obs_region_series(v, init_v, rg)
        a.plot(days, obs, 'k-', lw=2.2, zorder=6, label='ERA5')
        for m in MODELS:
            mu = r[f'{v}_mu'][MODELS.index(m), istar, :, ri] - CLIMr[v][rg]
            a.plot(days, mu, '-', color=COL[m], lw=1.8, label=LAB[m])
            if m == 'SPIRE':
                sg = r[f'{v}_sig'][MODELS.index(m), istar, :, ri]
                a.fill_between(days, mu - sg, mu + sg, color=COL[m], alpha=0.18, lw=0)
        a.axhline(0, color='0.6', lw=0.7); a.axvspan(36, 42, color='0.85', alpha=0.5, zorder=0)
        a.set_title(RLAB[rg], fontsize=12, fontweight='bold'); a.set_xlabel('forecast day')
        if ci == 0:
            a.set_ylabel(f'{vname} anomaly ({unit})', fontweight='bold'); a.legend(fontsize=8, loc='best')
    fig.suptitle(f'Case study {idx} — {vname}: regional area-averaged anomaly, init {init_v} '
                 f'(SPIRE $\\pm1\\sigma$ shaded; grey = week-6 window)', fontsize=13.5, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{fname}.{ext}', bbox_inches='tight')
    plt.close(fig); print(f'WROTE {fname} (init {init_v})', flush=True)


make('t2', 'fig23_regional_case_t2m', 1)
make('tp', 'fig24_regional_case_precip', 2)
make('z', 'fig25_regional_case_z500', 3)
print('REGIONAL_CASE_DONE', flush=True)
