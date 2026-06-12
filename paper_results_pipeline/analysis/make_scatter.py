"""
Density scatter of forecast vs ERA5 (reference: FuXi-S2S / TianXing-S2S Fig 10).
One row per variable (TP, Z500, T2M), one column per system (SPIRE, FuXi, ECMWF,
NCEP). Each panel pools all 13 inits x 6 lead-weeks x land points; colour = point
density (Gaussian KDE, subsampled). Annotated with R^2, MAE, Bias and a 1:1 line.
Reads analysis/scatter_points.npz -> writes paper/figs/fig11_scatter_density.{pdf,png}
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
d = np.load(f'{ADIR}/scatter_points.npz')
present_vars = [v for v in ['TP', 'Z500', 'T2M'] if f'{v}_SPIRE_fcst' in d.files]
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
VLAB = {'TP': 'Precipitation\n(mm day$^{-1}$)', 'Z500': 'Z500 anomaly\n(m)', 'T2M': 'T2M (K)'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'savefig.dpi': 300, 'figure.dpi': 110,
                     'font.family': 'DejaVu Sans'})
rng = np.random.default_rng(0)


def metrics(f, o):
    r = np.corrcoef(f, o)[0, 1]
    return r ** 2, np.mean(np.abs(f - o)), np.mean(f - o)


def panel(ax, var, model, lim):
    f = d[f'{var}_{model}_fcst']; o = d[f'{var}_{model}_obs']
    # density colour via KDE on a subsample (speed), evaluated at all points
    n = f.size
    idx = rng.choice(n, size=min(n, 4000), replace=False)
    try:
        kde = gaussian_kde(np.vstack([o[idx], f[idx]]))
        # evaluate on a thinned set for plotting
        pidx = rng.choice(n, size=min(n, 12000), replace=False)
        dens = kde(np.vstack([o[pidx], f[pidx]]))
        order = np.argsort(dens)
        sc = ax.scatter(o[pidx][order], f[pidx][order], c=dens[order], s=5,
                        cmap='viridis', linewidths=0, rasterized=True)
    except Exception:
        sc = ax.scatter(o, f, s=4, color='#4477AA', alpha=0.3, rasterized=True)
    r2, mae, bias = metrics(f, o)
    ax.plot(lim, lim, 'k-', lw=1.0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nBias = {bias:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=9.5,
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    ax.set_aspect('equal', 'box')
    return sc


def robust_lim(var):
    """common axis limits per variable, from pooled obs+fcst percentiles."""
    vals = []
    for m in MODELS:
        if f'{var}_{m}_fcst' in d.files:
            vals.append(d[f'{var}_{m}_fcst']); vals.append(d[f'{var}_{m}_obs'])
    a = np.concatenate(vals)
    lo, hi = np.nanpercentile(a, [0.5, 99.5])
    if var == 'TP':
        lo = 0.0
    return (float(lo), float(hi))


nrow = len(present_vars)
fig, axes = plt.subplots(nrow, 4, figsize=(15, 3.8 * nrow))
axes = np.atleast_2d(axes)
for r, var in enumerate(present_vars):
    lim = robust_lim(var)
    sc = None
    for c, m in enumerate(MODELS):
        ax = axes[r, c]
        if f'{var}_{m}_fcst' in d.files:
            sc = panel(ax, var, m, lim)
        if r == 0:
            ax.set_title(LAB[m])
        if c == 0:
            ax.set_ylabel(VLAB[var] + '\nPrediction', fontweight='bold')
        if r == nrow - 1:
            ax.set_xlabel('ERA5 (ground truth)')
    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes[r, :]), fraction=0.012, pad=0.01)
        cb.set_label('Density', fontsize=9); cb.set_ticks([])
fig.suptitle('Forecast vs ERA5 density scatter over India (all inits and lead weeks pooled) — JFM 2026',
             fontsize=14, fontweight='bold', y=0.995)
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig11_scatter_density.{ext}', bbox_inches='tight')
print('WROTE fig11_scatter_density', flush=True)
print('SCATTER_FIG_DONE', flush=True)
