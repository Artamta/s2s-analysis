"""
Three independent week-6 case studies (reference: TianXing-S2S Figs 3/S7/S10).
For three initializations (chosen as the strongest week-6 verifying anomaly for
each of T2M, precipitation, and Z500), each figure shows:
  TOP    : ERA5 week-6 anomaly + each system's week-6 ensemble-mean anomaly map.
  BOTTOM : per-system All-India area-averaged anomaly vs forecast day (1-42),
           observed (black) with the ensemble mean and +/-1 sigma spread band.
Week-6 ensemble means + obs come from prob_fields.nc; daily spread from
prob_daily_regional.npz. Coastline only. -> paper/figs/fig20-22_casestudy_*.{pdf,png}
"""
import sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
AI = (5., 38., 65., 100.)
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.PlateCarree()
except Exception:
    proj = None
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'savefig.dpi': 300})


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=tlat, lon=tlon, method='linear'), LAND)


def aimean(g):
    a, b, c, d = AI; s = g.sel(lat=slice(b, a), lon=slice(c, d)); w = np.cos(np.deg2rad(s.lat))
    return float(s.weighted(w).mean(['lat', 'lon']))


f = xr.open_dataset(f'{ADIR}/prob_fields.nc')
inits = [str(x) for x in f['init'].values]
lat = f['lat'].values; lon = f['lon'].values
era_tp = xr.open_dataset(f'{ADIR}/era5_daily_tp.nc')['tp']
era_t2 = xr.open_dataset(f'{ADIR}/era5_daily_t2m.nc')['t2m']
era_z = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
CLIMg = {'tp': to_grid(era_tp.mean('time')), 't2': to_grid(era_t2.mean('time')), 'z': to_grid(era_z.mean('time'))}
CLIMa = {k: aimean(v) for k, v in CLIMg.items()}
r = np.load(f'{ADIR}/prob_daily_regional.npz', allow_pickle=True)
regions = list(r['regions']); ri_ai = regions.index('All India')
KEYNC = {'tp': ('tp_mu', 'tp_obs'), 't2': ('t2_mu', 't2_obs'), 'z': ('z_mu', 'z_obs')}
VINFO = {'t2': ('2-m temperature', 'K', 'RdBu_r'), 'tp': ('precipitation', 'mm day$^{-1}$', 'BrBG'),
         'z': ('Z500', 'm', 'RdBu_r')}


def obs_daily_ai(v, init):
    src = {'tp': era_tp, 't2': era_t2, 'z': era_z}[v]
    out = []
    for di in range(42):
        date = (pd.to_datetime(init) + pd.Timedelta(days=di)).strftime('%Y-%m-%d')
        try:
            out.append(aimean(to_grid(src.sel(time=slice(date, date)).mean('time'))) - CLIMa[v])
        except Exception:
            out.append(np.nan)
    return np.array(out)


def panel_map(ax, data, vmax, cmap):
    if proj:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto', transform=proj)
        ax.add_feature(cfeature.COASTLINE, lw=0.5); ax.set_extent([65, 100, 5, 38], crs=proj)
    else:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
    return im


USED = set()


def make_case(v, fname, idx):
    mu_k, obs_k = KEYNC[v]
    clim = CLIMg[v].values
    # pick init with strongest week-6 obs anomaly RMS, excluding already-used inits
    amp = np.array([np.sqrt(np.nanmean((f[obs_k].isel(init=i, week=5).values - clim) ** 2)) for i in range(len(inits))])
    order = np.argsort(amp)[::-1]
    istar = next(int(i) for i in order if i not in USED); USED.add(istar); init_v = inits[istar]
    vname, unit, cmap = VINFO[v]
    obs_an = f[obs_k].isel(init=istar, week=5).values - clim
    fcs = {m: f[mu_k].sel(model=m).isel(init=istar, week=5).values - clim for m in MODELS}
    vmax = np.nanpercentile(np.abs(obs_an[np.isfinite(obs_an)]), 96)

    fig = plt.figure(figsize=(17, 8.2))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.12)
    # top: maps (week-6 average, i.e. forecast days 36-42)
    for c, (title, data) in enumerate([('ERA5 (obs)', obs_an)] + [(LAB[m], fcs[m]) for m in MODELS]):
        ax = fig.add_subplot(gs[0, c], projection=proj) if proj else fig.add_subplot(gs[0, c])
        im = panel_map(ax, data, vmax, cmap)
        ax.set_title(title, fontsize=12, fontweight='bold')
    fig.text(0.5, 0.905, 'Week-6 mean anomaly (forecast days 36–42)', ha='center', fontsize=11, style='italic')
    cax = fig.add_axes([0.92, 0.56, 0.011, 0.32]); fig.colorbar(im, cax=cax, label=f'{vname} anomaly ({unit})')
    # bottom: per-model time series with spread
    days = np.arange(1, 43); obs_ts = obs_daily_ai(v, init_v)
    for c, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, c + 1])
        mu = r[f'{v}_mu'][MODELS.index(m), istar, :, ri_ai] - CLIMa[v]
        sg = r[f'{v}_sig'][MODELS.index(m), istar, :, ri_ai]
        ax.plot(days, obs_ts, 'k-', lw=2, zorder=5, label='ERA5')
        ax.plot(days, mu, '-', color=COL[m], lw=2, label=f'{LAB[m]} mean')
        ax.fill_between(days, mu - sg, mu + sg, color=COL[m], alpha=0.25, lw=0, label='$\\pm1\\sigma$')
        ax.axhline(0, color='0.6', lw=0.7); ax.axvspan(36, 42, color='0.85', alpha=0.5, zorder=0)
        ax.set_xlabel('forecast day'); ax.set_title(LAB[m], fontsize=10)
        if c == 0:
            ax.set_ylabel(f'All-India anomaly ({unit})', fontweight='bold'); ax.legend(fontsize=8, loc='upper left')
    ax0 = fig.add_subplot(gs[1, 0]); ax0.axis('off')
    ax0.text(0.5, 0.5, f'All-India area-averaged\n{vname} anomaly\nover the FULL\n42-day forecast\n\n(grey band =\nweek-6 window,\ndays 36–42)',
             ha='center', va='center', fontsize=10, fontweight='bold', transform=ax0.transAxes)
    fig.suptitle(f'Case study {idx} — {vname}: init {init_v}, verified at week-6 lead (ocean masked)',
                 fontsize=14, fontweight='bold', y=0.96)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{fname}.{ext}', bbox_inches='tight')
    plt.close(fig); print(f'WROTE {fname} (init {init_v})', flush=True)


make_case('t2', 'fig20_casestudy_t2m', 1)
make_case('tp', 'fig21_casestudy_precip', 2)
make_case('z', 'fig22_casestudy_z500', 3)
print('CASE_STUDIES_DONE', flush=True)
