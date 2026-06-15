"""
FIG 12 – Week-wise density scatter grid.
For each variable (TP, Z500, T2M):
  - Rows = Weeks 1–6, Columns = 4 models
  - Each panel: hexbin density scatter + R², MAE, Bias + 1:1 line
  - Log colour scale (magma_r)
Reads: paper/results/scatter_points_weekwise.npz
Writes: paper/figs/fig12_scatter_weekwise_{TP,Z500,T2M}.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

DATAFILE = '/home/raj.ayush/s2s/s2s_anlysis/paper/results/scatter_points_weekwise.npz'
FIGDIR   = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
MODELS   = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
MLAB     = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
VUNIT    = {'TP': 'mm day$^{-1}$', 'Z500': 'gpm', 'T2M': 'K'}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'savefig.dpi': 600, 'figure.dpi': 120,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 0.7,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})


def metrics(f, o):
    r = np.corrcoef(f, o)[0, 1]
    return r**2, np.mean(np.abs(f - o)), np.mean(f - o)


def robust_lim(d, var):
    vals = []
    for m in MODELS:
        for w in range(1, 7):
            for suf in ('_fcst', '_obs'):
                k = f'{var}_{m}_W{w}{suf}'
                if k in d.files:
                    vals.append(d[k])
    if not vals:
        return (-1, 1)
    a = np.concatenate(vals)
    lo, hi = np.nanpercentile(a, [0.5, 99.5])
    mx = max(abs(lo), abs(hi))
    return (-mx, mx)


def make_figure(d, var):
    lim = robust_lim(d, var)
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.04,
                        hspace=0.30, wspace=0.30)

    hb_global = None
    for wi in range(6):
        wk = wi + 1
        for ci, m in enumerate(MODELS):
            ax = axes[wi, ci]
            fk = f'{var}_{m}_W{wk}_fcst'
            ok = f'{var}_{m}_W{wk}_obs'
            if fk in d.files and ok in d.files:
                f = d[fk]; o = d[ok]
                hb = ax.hexbin(o, f, gridsize=35, cmap='magma_r', bins='log', mincnt=1,
                               extent=(lim[0], lim[1], lim[0], lim[1]), linewidths=0.0)
                r2, mae, bias = metrics(f, o)
                ax.plot(lim, lim, '-', color='#2c7fb8', lw=1.1, zorder=5)
                ax.text(0.04, 0.96,
                        f'$R^2$={r2:.3f}\nMAE={mae:.2f}\nbias={bias:+.2f}',
                        transform=ax.transAxes, va='top', ha='left', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.92))
                hb_global = hb
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='0.5')

            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=7.5)

            # Column titles (top row only)
            if wi == 0:
                ax.set_title(MLAB[m], fontsize=13, fontweight='bold', pad=8)

            # Row labels (left column only)
            if ci == 0:
                ax.set_ylabel(f'Forecast ({VUNIT[var]})', fontsize=9)
            else:
                ax.set_ylabel('')

            # Bottom row x-label
            if wi == 5:
                ax.set_xlabel(f'ERA5 ({VUNIT[var]})', fontsize=9)
            else:
                ax.set_xticklabels([])

        # Week label on the right side
        axes[wi, -1].annotate(f'Week {wk}', xy=(1.08, 0.5),
                              xycoords='axes fraction', fontsize=12,
                              fontweight='bold', rotation=-90,
                              ha='center', va='center')

    # Single shared colourbar
    if hb_global is not None:
        cax = fig.add_axes([0.94, 0.08, 0.015, 0.82])
        cb = fig.colorbar(hb_global, cax=cax)
        cb.set_label('Count (log scale)', fontsize=10)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Forecast vs ERA5 Density Scatter — {var} Anomaly\n'
        f'India land only · JFM 2026 · 13 initialisations pooled',
        fontsize=14, fontweight='bold')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/fig12_scatter_weekwise_{var}.{ext}',
                    bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'WROTE fig12_scatter_weekwise_{var}', flush=True)


if __name__ == '__main__':
    d = np.load(DATAFILE)
    for var in ['TP', 'Z500', 'T2M']:
        if f'{var}_SPIRE_W1_fcst' in d.files:
            make_figure(d, var)
        else:
            print(f'SKIP {var} — no data')
    print('ALL_DONE', flush=True)
