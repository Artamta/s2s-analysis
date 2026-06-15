"""
Additional figures (main + appendix) for the JFM-2026 SPIRE benchmark:

  Fig 6  MME + persistence skill horizon (TP PCC | Z500 PCC | Z500 RMSE)
  Fig 7  Spatial temporal-PCC maps (where each system has skill), Z500 Week 2
  Fig 8  Spatial Z500 mean-bias maps, Week 2 (appendix)
  Fig 9  Per-IMD-region PCC vs lead, small multiples (appendix)

Reads: analysis/skill_per_init_full.csv and analysis/weekly_anom_fields.nc
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import bootstrap_ci

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
os.makedirs(FIGDIR, exist_ok=True)
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7',
       'MME': 'black', 'Persistence': '#888888'}
STY = {'SPIRE': ('-', 's'), 'FuXi': ('-', 'o'), 'ECMWF': ('--', '^'), 'NCEP': ('-.', 'd'),
       'MME': ('-', '*'), 'Persistence': (':', 'x')}
REG = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'northwest_india': 'Northwest', 'central_india': 'Central',
        'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelweight': 'bold',
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

df = pd.read_csv(f'{ADIR}/skill_per_init_full.csv')
df = df[df.variable != 'TP']
tp_corr = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv')
df = pd.concat([df, tp_corr], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
try:
    fields = xr.open_dataset(f'{ADIR}/weekly_anom_fields.nc')
except Exception:
    fields = None


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  wrote', name, flush=True)


def horizon(ax, variable, metric, models, refline=None, ylabel=''):
    """Clean skill-horizon curves — no CI ribbons."""
    for m in models:
        sub = df[(df.variable == variable) & (df.region == 'All India') & (df.model == m)]
        xs, ys = [], []
        for wk in sorted(sub.wk.unique()):
            vals = sub[sub.wk == wk][metric].values
            mean = np.nanmean(vals) if len(vals) else np.nan
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean)
        if not xs:
            continue
        ls, mk = STY[m]
        lw = 2.8 if m == 'MME' else (1.6 if m == 'Persistence' else 2.2)
        alpha = 0.45 if m == 'Persistence' else 1.0
        zord = 5 if m == 'MME' else (2 if m == 'Persistence' else 3)
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=lw, ms=7,
                label=m, zorder=zord, alpha=alpha,
                markerfacecolor='white' if m == 'Persistence' else COL[m],
                markeredgecolor=COL[m], markeredgewidth=1.2)
    if refline is not None:
        ax.axhline(refline, color='#555555', lw=0.9, ls=(0, (6, 3)), zorder=1)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 7)])
    ax.set_xlim(0.55, 6.45)
    ax.grid(axis='y', ls=':', alpha=0.4, color='#CCCCCC')
    ax.set_ylabel(ylabel)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#333333')


# ---------- Fig 6: MME + persistence ----------
def fig6():
    mods = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8),
                           gridspec_kw={'wspace': 0.30})
    horizon(ax[0], 'TP', 'pcc', mods, refline=0.5,
            ylabel='Anomaly Correlation Coefficient (ACC)')
    ax[0].set_title('(a)  Precipitation — ACC', fontweight='bold', pad=10)
    ax[0].set_ylim(-0.3, 1.04)
    # no-skill zone
    ax[0].axhspan(-0.3, 0.5, color='#F7F7F7', zorder=0)

    horizon(ax[1], 'Z500', 'pcc', mods, refline=0.5,
            ylabel='Anomaly Correlation Coefficient (ACC)')
    ax[1].set_title('(b)  Z500 — ACC', fontweight='bold', pad=10)
    ax[1].set_ylim(-0.1, 1.04)
    ax[1].axhspan(-0.1, 0.5, color='#F7F7F7', zorder=0)

    horizon(ax[2], 'Z500', 'rmse', mods,
            ylabel='RMSE  (gpm)')
    ax[2].set_title('(c)  Z500 — RMSE', fontweight='bold', pad=10)

    ax[0].legend(loc='upper right', fontsize=8.5, ncol=2, frameon=True,
                 fancybox=False, edgecolor='#CCCCCC', framealpha=0.96,
                 borderpad=0.5, handlelength=2.2)
    fig.suptitle(
        'Multi-model Ensemble (MME) and Persistence Baseline — JFM 2026',
        fontsize=12.5, fontweight='bold', y=1.02, color='#111111')
    fig.text(0.5, -0.02,
             'All-India (land only, IMD boundaries). '
             'Cosine-latitude weighted area mean over 13 initialisations.',
             ha='center', fontsize=7.8, color='#777777', fontstyle='italic')
    fig.tight_layout()
    save(fig, 'fig06_mme_persistence')


# ---------- spatial helpers ----------
def gridpoint_pcc(F, O):  # (init,lat,lon) anomaly arrays -> (lat,lon) temporal corr
    Fa = F - np.nanmean(F, 0); Oa = O - np.nanmean(O, 0)
    num = np.nansum(Fa * Oa, 0)
    den = np.sqrt(np.nansum(Fa ** 2, 0) * np.nansum(Oa ** 2, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    r[den == 0] = np.nan
    return r


def spatial_panel(fig, gs_or_axes, data, models, title, cmap, vmin, vmax, cbar_label):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
    except Exception:
        proj = None
    lat = fields['lat'].values; lon = fields['lon'].values
    axes = []
    for j, m in enumerate(models):
        ax = fig.add_subplot(1, len(models), j + 1, projection=proj) if proj else fig.add_subplot(1, len(models), j + 1)
        im = ax.pcolormesh(lon, lat, data[m], cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=proj, shading='auto') if proj else ax.pcolormesh(lon, lat, data[m], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        if proj:
            ax.add_feature(cfeature.COASTLINE, lw=0.5)
            ax.set_extent([65, 100, 5, 38], crs=proj)
        ax.set_title(m, fontsize=11)
        axes.append((ax, im))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.04)
    fig.colorbar(axes[-1][1], ax=[a for a, _ in axes], fraction=0.025, pad=0.02, label=cbar_label)


# ---------- Fig 7: spatial temporal-PCC, Z500 Week 2 ----------
def fig7():
    wk_i = 1  # Week 2 index
    models = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME']
    O = fields['z_obs'].isel(week=wk_i).values
    data = {m: gridpoint_pcc(fields['z_fcst'].sel(model=m).isel(week=wk_i).values, O) for m in models}
    fig = plt.figure(figsize=(18, 4))
    spatial_panel(fig, None, data, models,
                  'Local temporal anomaly correlation of Z500, Week 2 (across 13 inits) — JFM 2026',
                  'RdBu_r', -1, 1, 'temporal PCC')
    save(fig, 'fig07_spatial_pcc_z500')


# ---------- Fig 8: spatial Z500 bias, Week 2 ----------
def fig8():
    wk_i = 1
    models = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
    O = fields['z_obs'].isel(week=wk_i).values
    data = {m: np.nanmean(fields['z_fcst'].sel(model=m).isel(week=wk_i).values - O, 0) for m in models}
    vmax = np.nanpercentile(np.abs(np.concatenate([d[np.isfinite(d)] for d in data.values()])), 95)
    fig = plt.figure(figsize=(16, 4))
    spatial_panel(fig, None, data, models,
                  'Z500 mean anomaly bias (forecast - ERA5), Week 2 — JFM 2026',
                  'RdBu_r', -vmax, vmax, 'bias (m)')
    save(fig, 'fig08_spatial_bias_z500')


# ---------- Fig 9: per-region PCC vs lead ----------
def fig9():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for col, rg in enumerate(REG):
        for row, var in enumerate(['TP', 'Z500']):
            ax = axes[row, col]
            for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']:
                sub = df[(df.variable == var) & (df.region == rg) & (df.model == m)]
                xs, ys = [], []
                for wk in sorted(sub.wk.unique()):
                    mean, _, _ = bootstrap_ci(sub[sub.wk == wk]['pcc'].values)
                    if np.isfinite(mean):
                        xs.append(wk); ys.append(mean)
                ls, mk = STY[m]
                ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2, ms=5, label=m)
            ax.axhline(0.5, color='k', lw=0.8, ls=':'); ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(range(1, 7)); ax.set_ylim(-0.4, 1.0); ax.grid(axis='y', ls=':', alpha=0.5)
            if row == 0: ax.set_title(REGL[rg])
            if col == 0: ax.set_ylabel(f'{var} PCC')
    axes[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Skill by IMD homogeneous region and lead — JFM 2026', fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout(); save(fig, 'fig09_regional_horizon')


if __name__ == '__main__':
    print('Building extra figures ...', flush=True)
    fig6(); fig7(); fig8(); fig9()
    print('EXTRA_FIGURES_DONE', flush=True)
