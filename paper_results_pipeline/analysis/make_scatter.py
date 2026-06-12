"""
Density (hexbin) scatter of forecast vs ERA5 (reference: FuXi-S2S / TianXing-S2S
Fig 10). Rows = variables (TP, Z500, T2M); columns = systems (SPIRE, FuXi, ECMWF,
NCEP). Each panel pools ALL 13 inits x 6 lead-weeks (forecast days 1-42) x land
points (~26k pairs); colour = log point density. Annotated with R^2, MAE, Bias
and a 1:1 line. FuXi precipitation is unit-harmonised to mm day^-1 at the dump
stage. Reads analysis/scatter_points.npz -> paper/figs/fig11_scatter_density.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
d = np.load(f'{ADIR}/scatter_points.npz')
present_vars = [v for v in ['TP', 'Z500', 'T2M'] if f'{v}_SPIRE_fcst' in d.files]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
VLAB = {'TP': 'Precipitation\n(mm day$^{-1}$)', 'Z500': 'Z500 anomaly\n(m)', 'T2M': 'T2M (K)'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'savefig.dpi': 300, 'figure.dpi': 110,
                     'font.family': 'DejaVu Sans'})


def metrics(f, o):
    r = np.corrcoef(f, o)[0, 1]
    return r ** 2, np.mean(np.abs(f - o)), np.mean(f - o)


def panel(ax, var, model, lim):
    f = d[f'{var}_{model}_fcst']; o = d[f'{var}_{model}_obs']
    hb = ax.hexbin(o, f, gridsize=42, cmap='magma_r', bins='log', mincnt=1,
                   extent=(lim[0], lim[1], lim[0], lim[1]), linewidths=0.0)
    r2, mae, bias = metrics(f, o)
    ax.plot(lim, lim, '-', color='#2c7fb8', lw=1.3, zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.04, 0.96, f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nBias = {bias:+.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    ax.set_aspect('equal', 'box')
    return hb


def robust_lim(var):
    vals = []
    for m in MODELS:
        if f'{var}_{m}_fcst' in d.files:
            vals.append(d[f'{var}_{m}_fcst']); vals.append(d[f'{var}_{m}_obs'])
    a = np.concatenate(vals)
    lo, hi = np.nanpercentile(a, [0.2, 99.6])
    if var == 'TP':
        lo = 0.0
    pad = 0.03 * (hi - lo)
    return (float(lo), float(hi + pad))


nrow = len(present_vars)
fig, axes = plt.subplots(nrow, 4, figsize=(15, 3.9 * nrow))
axes = np.atleast_2d(axes)
for r, var in enumerate(present_vars):
    lim = robust_lim(var)
    hb = None
    for c, m in enumerate(MODELS):
        ax = axes[r, c]
        if f'{var}_{m}_fcst' in d.files:
            hb = panel(ax, var, m, lim)
        if r == 0:
            ax.set_title(LAB[m])
        if c == 0:
            ax.set_ylabel(VLAB[var] + '\nForecast', fontweight='bold')
        if r == nrow - 1:
            ax.set_xlabel('ERA5 (ground truth)')
    if hb is not None:
        cb = fig.colorbar(hb, ax=list(axes[r, :]), fraction=0.012, pad=0.01)
        cb.set_label('count (log)', fontsize=9)
fig.suptitle('Forecast vs ERA5 density scatter over India — all 13 inits and lead weeks 1–6 (days 1–42) pooled, JFM 2026',
             fontsize=13.5, fontweight='bold', y=0.995)
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig11_scatter_density.{ext}', bbox_inches='tight')
print('WROTE fig11_scatter_density', flush=True)
print('SCATTER_FIG_DONE', flush=True)
