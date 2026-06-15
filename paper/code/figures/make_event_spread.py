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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''}); G = 9.80665
tlat = np.arange(38, 5, -1.5); tlon = np.arange(65, 100, 1.5); LAND = get_land_mask(tlat, tlon)
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11.5,
    'font.family': 'DejaVu Sans',
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})
AI = (5., 38., 65., 100.)

# ── Week boundaries for shading ──────────────────────────────────────────────
WEEK_BOUNDS = [1, 8, 15, 22, 29, 36, 43]   # day 1-7, 8-14, ... 36-42


def to_grid(da):
    ren = {}
    if 'latitude' in da.dims: ren['latitude'] = 'lat'
    if 'longitude' in da.dims: ren['longitude'] = 'lon'
    if ren: da = da.rename(ren)
    return mask_land(da.interp(lat=tlat, lon=tlon, method='linear'), LAND)


def aimean(g):
    a, b, c, d = AI; s = g.sel(lat=slice(b, a), lon=slice(c, d)); w = np.cos(np.deg2rad(s.lat))
    return float(s.weighted(w).mean(['lat', 'lon']))


era_tp = xr.open_dataset(f'{DATA}/era5/daily/era5_daily_tp.nc')['tp']
era_z = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
if 'latitude' in era_z.dims: era_z = era_z.rename({'latitude': 'lat', 'longitude': 'lon'})
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

# ── Pre-compute consistent y-limits per row ──────────────────────────────────
all_z_vals, all_tp_vals = [obs_z], [obs_tp]
for m in MODELS:
    mu_z = r['z_mu'][MODELS.index(m), istar, :, ri_ai] - clim_z
    sg_z = r['z_sig'][MODELS.index(m), istar, :, ri_ai]
    mu_tp = r['tp_mu'][MODELS.index(m), istar, :, ri_ai] - clim_tp
    sg_tp = r['tp_sig'][MODELS.index(m), istar, :, ri_ai]
    all_z_vals.extend([mu_z - sg_z, mu_z + sg_z])
    all_tp_vals.extend([mu_tp - sg_tp, mu_tp + sg_tp])

z_all = np.concatenate(all_z_vals); tp_all = np.concatenate(all_tp_vals)
z_pad = (np.nanmax(z_all) - np.nanmin(z_all)) * 0.08
tp_pad = (np.nanmax(tp_all) - np.nanmin(tp_all)) * 0.08
ylim_z = (np.nanmin(z_all) - z_pad, np.nanmax(z_all) + z_pad)
ylim_tp = (np.nanmin(tp_all) - tp_pad, np.nanmax(tp_all) + tp_pad)

# ── Panel labels ─────────────────────────────────────────────────────────────
PANEL_LABELS = [
    '(a)', '(b)', '(c)', '(d)',
    '(e)', '(f)', '(g)', '(h)',
]

fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
for c, m in enumerate(MODELS):
    for rrow, (vkey, obs, clim, unit, vlab, ylim) in enumerate([
            ('z',  obs_z,  clim_z,  'm',               'Z500 anomaly',   ylim_z),
            ('tp', obs_tp, clim_tp, 'mm day$^{-1}$',   'Precip anomaly', ylim_tp)]):
        a = ax[rrow, c]
        mu = r[f'{vkey}_mu'][MODELS.index(m), istar, :, ri_ai] - clim
        sg = r[f'{vkey}_sig'][MODELS.index(m), istar, :, ri_ai]

        # Alternate week shading for readability
        for wi in range(0, 6, 2):
            a.axvspan(WEEK_BOUNDS[wi], WEEK_BOUNDS[wi + 1] - 0.5,
                      color='0.93', zorder=0, lw=0)
        # Week labels along x
        if rrow == 1:
            for wi in range(6):
                a.text((WEEK_BOUNDS[wi] + WEEK_BOUNDS[wi + 1] - 1) / 2, ylim[0] - tp_pad * 0.1,
                       f'W{wi + 1}', ha='center', va='top', fontsize=7.5,
                       color='0.45', fontweight='bold', clip_on=False)

        # Data
        a.fill_between(days, mu - sg, mu + sg, color=COL[m], alpha=0.22, lw=0)
        a.plot(days, obs, 'k-', lw=2.0, zorder=5)
        a.plot(days, mu, '-', color=COL[m], lw=1.8)
        a.axhline(0, color='0.55', lw=0.7, ls='--', zorder=1)

        # Consistent y-limits
        a.set_ylim(ylim)
        a.set_xlim(0.5, 42.5)

        # Panel label
        pidx = rrow * 4 + c
        a.text(0.02, 0.96, PANEL_LABELS[pidx], transform=a.transAxes,
               fontsize=11, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='0.7', alpha=0.85))

        # Titles and labels
        if rrow == 0:
            a.set_title(LAB[m], fontsize=13, pad=8)
        if c == 0:
            a.set_ylabel(f'{vlab} ({unit})', fontweight='bold')
        else:
            a.tick_params(labelleft=False)
        if rrow == 1:
            a.set_xlabel('Forecast day', fontsize=11)
            a.set_xticks([1, 7, 14, 21, 28, 35, 42])

# ── Shared legend at top ─────────────────────────────────────────────────────
h_era  = mlines.Line2D([], [], color='k', lw=2.0, label='ERA5 (observed)')
h_mean = mlines.Line2D([], [], color='0.4', lw=1.8, label='Ensemble mean')
h_spr  = mpatches.Patch(fc='0.6', alpha=0.3, label=r'$\pm$1$\sigma$ ensemble spread')
fig.legend(handles=[h_era, h_mean, h_spr], loc='upper center',
           ncol=3, fontsize=11, frameon=True, fancybox=True,
           edgecolor='0.7', facecolor='white',
           bbox_to_anchor=(0.5, 1.005))

init_pretty = pd.to_datetime(init_v).strftime('%d %b %Y')
fig.suptitle(f'Event forecast with ensemble spread — All-India anomaly\n'
             f'Init: {init_pretty}  |  JFM 2026',
             fontsize=14, fontweight='bold', y=1.06)

fig.text(0.5, -0.01,
         r'Alternate grey bands mark forecast weeks (W1–W6). '
         r'Narrow $\pm$1$\sigma$ bands indicate overconfident (under-dispersed) ensembles.',
         ha='center', fontsize=9.5, style='italic', color='0.35')

fig.tight_layout(rect=[0, 0.01, 1, 0.96], w_pad=1.0, h_pad=2.5)
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig18_event_spread.{ext}', bbox_inches='tight', facecolor='white')
print('WROTE fig18_event_spread', flush=True)
print('EVENT_TS_DONE', flush=True)
