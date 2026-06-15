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
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import get_land_mask, mask_land

DATA = '/storage/raj.ayush/s2s-forecast-data'
ADIR     = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
ADIR_OLD = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'  # only for T2M spatial maps
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
OPEN = dict(engine='cfgrib', backend_kwargs={'indexpath': ''}); G = 9.80665
tlat = np.arange(38, 5, -1.5); tlon = np.arange(65, 100, 1.5); LAND = get_land_mask(tlat, tlon)
# IMD All-India mask — same 126 pixels used for all metrics
_mask_ds = xr.open_dataset('/storage/raj.ayush/s2s-forecast-data/era5/daily/imd_region_masks.nc')
ALL_INDIA_MASK = np.zeros((len(tlat), len(tlon)), dtype=bool)
for _v in _mask_ds.data_vars: ALL_INDIA_MASK |= _mask_ds[_v].values.astype(bool)
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


f    = xr.open_dataset(f'{ADIR}/prob_fields.nc')       # TP + Z500 (new)
f_t2 = xr.open_dataset(f'{ADIR_OLD}/prob_fields.nc')  # T2M (old file, T2M not in new)
inits = [str(x) for x in f['init'].values]
lat = f['lat'].values; lon = f['lon'].values
era_tp = xr.open_dataset(f'{DATA}/era5/daily/era5_daily_tp.nc')['tp']
era_t2 = xr.open_dataset(f'{DATA}/era5/daily/era5_daily_t2m.nc')['t2m']
era_z = xr.open_dataset(f'{DATA}/era5/data/era5_pressure_500hpa.grib', **OPEN)['z'] / G
CLIMg = {'tp': to_grid(era_tp.mean('time')), 't2': to_grid(era_t2.mean('time')), 'z': to_grid(era_z.mean('time'))}
CLIMa = {k: aimean(v) for k, v in CLIMg.items()}
r     = np.load(f'{ADIR}/prob_daily_regional.npz', allow_pickle=True)      # TP + Z500
r_old = np.load(f'{ADIR_OLD}/prob_daily_regional.npz', allow_pickle=True)  # T2M
regions = list(r['regions']); ri_ai = regions.index('All India')
KEYNC = {'tp': ('tp_mu', 'tp_obs', f,    r),     't2': ('t2_mu', 't2_obs', f_t2, r_old),
          'z':  ('z_mu',  'z_obs',  f,    r)}
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


def ai_mean_val(data):
    """Area-weighted mean over the ALL_INDIA_MASK pixels."""
    vals = data[ALL_INDIA_MASK]
    w = np.cos(np.deg2rad(
        np.tile(lat[:, None], (1, len(tlon)))[ALL_INDIA_MASK]))
    valid = np.isfinite(vals)
    if valid.sum() == 0: return np.nan
    return float(np.average(vals[valid], weights=w[valid]))


def panel_map(ax, data, vmax, cmap, unit):
    # Apply ALL_INDIA_MASK — only show the 126 IMD pixels, everything else NaN
    data_masked = np.where(ALL_INDIA_MASK, data, np.nan)
    if proj:
        # Fill ocean white first so non-India pixels are white
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('50m'),  facecolor='0.95', zorder=0)
        im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='auto', transform=proj, zorder=1)
        # Only the coastline (ocean/land edge) — no political lines inside
        coast = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                             edgecolor='#333333', facecolor='none')
        ax.add_feature(coast, lw=0.6, zorder=2)
        ax.set_extent([65, 100, 5, 38], crs=proj)
    else:
        im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
    # annotate All-India area mean
    mean_val = ai_mean_val(data)
    sign = '+' if mean_val >= 0 else ''
    ax.text(0.97, 0.04, f'{sign}{mean_val:.1f} {unit}',
            transform=ax.transAxes, fontsize=8.5, ha='right', va='bottom',
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
    return im


USED = set()


def make_case(v, fname, idx):
    mu_k, obs_k, fds, rr = KEYNC[v]
    clim = CLIMg[v].values
    # pick init with strongest week-6 obs anomaly RMS, excluding already-used inits
    amp = np.array([np.sqrt(np.nanmean((fds[obs_k].isel(init=i, week=5).values - clim) ** 2)) for i in range(len(inits))])
    order = np.argsort(amp)[::-1]
    istar = next(int(i) for i in order if i not in USED); USED.add(istar); init_v = inits[istar]
    vname, unit, cmap = VINFO[v]
    obs_an = fds[obs_k].isel(init=istar, week=5).values - clim
    fcs = {m: fds[mu_k].sel(model=m).isel(init=istar, week=5).values - clim for m in MODELS}
    vmax = np.nanpercentile(np.abs(obs_an[np.isfinite(obs_an)]), 96)

    fig = plt.figure(figsize=(18, 8.5))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.18,
                  width_ratios=[1.45, 1, 1, 1, 1])
    # top: maps (week-6 average, i.e. forecast days 36-42)
    for c, (title, data) in enumerate([('ERA5 (obs)', obs_an)] + [(LAB[m], fcs[m]) for m in MODELS]):
        ax = fig.add_subplot(gs[0, c], projection=proj) if proj else fig.add_subplot(gs[0, c])
        im = panel_map(ax, data, vmax, cmap, unit)
        ax.set_title(title, fontsize=12, fontweight='bold')
    fig.text(0.5, 0.905, 'Week-6 mean anomaly (forecast days 36\u201342) \u2014 India land pixels only (value = All-India area mean)',
             ha='center', fontsize=10.5, style='italic')
    cax = fig.add_axes([0.92, 0.56, 0.011, 0.32]); fig.colorbar(im, cax=cax, label=f'{vname} anomaly ({unit})')
    # bottom: per-model time series with spread
    days = np.arange(1, 43); obs_ts = obs_daily_ai(v, init_v)
    week_ticks = [7, 14, 21, 28, 35, 42]   # end of each week
    week_labels = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    ts_axes = []
    for c, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, c + 1])
        ts_axes.append(ax)
        mu = rr[f'{v}_mu'][MODELS.index(m), istar, :, ri_ai] - CLIMa[v]
        sg = rr[f'{v}_sig'][MODELS.index(m), istar, :, ri_ai]
        # grey week-6 band first (zorder=0)
        ax.axvspan(36, 42, color='0.85', alpha=0.6, zorder=0, label='Week-6 window')
        ax.axhline(0, color='0.55', lw=0.8, ls='--')
        # thin vertical week separators
        for wt in week_ticks[:-1]:
            ax.axvline(wt, color='0.75', lw=0.4, ls=':')
        ax.plot(days, obs_ts, 'k-', lw=2.2, zorder=5, label='ERA5 (obs)')
        ax.plot(days, mu, '-', color=COL[m], lw=2, label='Forecast mean')
        ax.fill_between(days, mu - sg, mu + sg, color=COL[m], alpha=0.22, lw=0, label='±1σ spread')
        # Week labels on x-axis
        ax.set_xticks(week_ticks)
        ax.set_xticklabels(week_labels, fontsize=8)
        ax.set_xlabel('Forecast week', fontsize=8.5)
        ax.set_title(LAB[m], fontsize=10.5, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        # Label "Week 6" inside the shaded band on first panel
        if c == 0:
            ax.set_ylabel(f'All-India mean anomaly ({unit})', fontsize=9, fontweight='bold')
            leg = ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85, edgecolor='0.8')
        # "Week 6" text inside grey band
        ymin, ymax = ax.get_ylim()
        ax.text(39, ymax * 0.92, 'Wk6', ha='center', va='top', fontsize=7,
                color='0.4', style='italic')

    # Scientific inference per variable
    INFERENCE = {
        'tp': (
            '📌 Key Finding',
            'No model fully captures\nthe observed +ve precip\nanomaly over NE India\nat week-6 lead.\n\n'
            'SPIRE gets the sign\ncorrect but underestimates\nmagnitude. FuXi-S2S and\nECMWF show ~zero anomaly\n(skill lost beyond W4).'),
        'z': (
            '📌 Key Finding',
            'ERA5 shows a strong\npositive Z500 anomaly\n(warm ridge, NW India)\nat week-6.\n\n'
            'FuXi-S2S predicts the\nopposite sign (cold trough)\n— a systematic dynamical\nbias at long leads.\n\n'
            'SPIRE/ECMWF/NCEP get\nthe sign but strongly\nunderestimate amplitude.'),
    }
    inf_title, inf_body = INFERENCE.get(v, ('', ''))
    ax0 = fig.add_subplot(gs[1, 0]); ax0.axis('off')

    # "How to read" box at top
    how_to = (
        'How to read:\n'
        '━━━━━━━━━━━━━\n'
        '  Black line  = ERA5 observed\n'
        '  Colour line = Model forecast\n'
        '  Shaded band = ±1σ ensemble\n'
        '  Grey area   = Week-6 window\n'
        '  Dashed zero = climatology'
    )
    ax0.text(0.5, 1.0, how_to, ha='center', va='top', fontsize=7.8,
             color='#111111', transform=ax0.transAxes, linespacing=1.5,
             family='monospace',
             bbox=dict(fc='#f5f5f5', ec='#cccccc', boxstyle='round,pad=0.5'))

    # Key finding box below
    ax0.text(0.5, 0.40, inf_title, ha='center', va='top', fontsize=9,
             fontweight='bold', color='#1a1a2e', transform=ax0.transAxes)
    ax0.text(0.5, 0.30, inf_body, ha='center', va='top', fontsize=8,
             color='#222222', transform=ax0.transAxes, linespacing=1.5)

    fig.suptitle(f'Case study {idx} — {vname}: init {init_v}, verified at week-6 lead (India only)',
                 fontsize=13.5, fontweight='bold', y=0.97)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{fname}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig); print(f'WROTE {fname} (init {init_v})', flush=True)


make_case('tp', 'fig20_casestudy_precip', 1)
make_case('z',  'fig21_casestudy_z500',   2)
print('CASE_STUDIES_DONE', flush=True)
