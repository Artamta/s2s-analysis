"""
FIG 14 – Conditional quantile bias scatter.
For each variable and each model: bin the ERA5 truth into deciles,
compute the mean forecast for each bin, and plot forecast-bin-mean vs
ERA5-bin-centre. Reveals systematic over/underprediction at extremes.
One figure per variable, 2×3 layout (Weeks 1-6 panels, all 4 models overlaid).
Reads: paper/results/scatter_points_weekwise.npz
Writes: paper/figs/fig14_quantile_bias_{TP,Z500,T2M}.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATAFILE = '/home/raj.ayush/s2s/s2s_anlysis/paper/results/scatter_points_weekwise.npz'
FIGDIR   = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
MODELS   = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']

STYLE = {
    'SPIRE': dict(color='#1b9e77', marker='o', ls='-',  lw=2.2, ms=7, zorder=5),
    'FuXi':  dict(color='#d95f02', marker='s', ls='--', lw=1.8, ms=6, zorder=4),
    'ECMWF': dict(color='#7570b3', marker='^', ls='-',  lw=2.2, ms=7, zorder=5),
    'NCEP':  dict(color='#e7298a', marker='D', ls='-.', lw=1.8, ms=6, zorder=4),
}
MLAB   = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
VUNIT  = {'TP': 'mm day$^{-1}$', 'Z500': 'gpm', 'T2M': 'K'}
VLABEL = {'TP': 'Precipitation Anomaly', 'Z500': 'Z500 Anomaly', 'T2M': 'T2M Anomaly'}
N_BINS = 10  # deciles

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'savefig.dpi': 600, 'figure.dpi': 120,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 0.7,
})


def quantile_means(obs, fcst, n_bins=N_BINS):
    """Return (bin_centres, mean_fcst, n_per_bin) for each obs quantile bin."""
    edges = np.nanpercentile(obs, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return None, None, None
    centres = []; means = []; counts = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (obs >= edges[i]) & (obs <= edges[i + 1])
        else:
            mask = (obs >= edges[i]) & (obs < edges[i + 1])
        if mask.sum() < 5:
            continue
        centres.append(np.mean(obs[mask]))
        means.append(np.mean(fcst[mask]))
        counts.append(mask.sum())
    return np.array(centres), np.array(means), np.array(counts)


def make_figure(d, var):
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.08,
                        hspace=0.28, wspace=0.28)
    axes_flat = axes.ravel()

    # Determine global axis range across all weeks/models
    all_centres = []
    all_means = []
    for wk in range(1, 7):
        for m in MODELS:
            fk = f'{var}_{m}_W{wk}_fcst'
            ok = f'{var}_{m}_W{wk}_obs'
            if fk in d.files:
                c, mn, _ = quantile_means(d[ok], d[fk])
                if c is not None:
                    all_centres.extend(c); all_means.extend(mn)
    if not all_centres:
        return
    lo = min(min(all_centres), min(all_means))
    hi = max(max(all_centres), max(all_means))
    pad = (hi - lo) * 0.08
    lim = (lo - pad, hi + pad)

    for wi, wk in enumerate(range(1, 7)):
        ax = axes_flat[wi]
        for m in MODELS:
            fk = f'{var}_{m}_W{wk}_fcst'
            ok = f'{var}_{m}_W{wk}_obs'
            if fk not in d.files:
                continue
            centres, means, counts = quantile_means(d[ok], d[fk])
            if centres is None:
                continue
            ax.plot(centres, means, label=MLAB[m], **STYLE[m])

        # 1:1 reference
        ax.plot(lim, lim, color='0.4', lw=1.0, ls=':', zorder=0, label='1:1')
        ax.fill_between(lim, [lim[0]], [lim[1]], alpha=0.04, color='grey', zorder=0)

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Week {wk}', fontsize=14, fontweight='bold', pad=8)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=11)

        if wi >= 3:
            ax.set_xlabel(f'ERA5 bin mean ({VUNIT[var]})', fontsize=12)
        else:
            ax.set_xticklabels([])
        if wi % 3 == 0:
            ax.set_ylabel(f'Forecast bin mean ({VUNIT[var]})', fontsize=12)
        if wi == 0:
            ax.legend(fontsize=11, framealpha=0.9, edgecolor='0.7', loc='upper left')

    fig.suptitle(
        f'Conditional Quantile Bias \u2014 {VLABEL[var]} ({VUNIT[var]})\n'
        f'India land only \u00b7 JFM 2026 \u00b7 ERA5 decile bins \u00b7 13 inits pooled',
        fontsize=15, fontweight='bold')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/fig14_quantile_bias_{var}.{ext}', bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'WROTE fig14_quantile_bias_{var}', flush=True)


if __name__ == '__main__':
    d = np.load(DATAFILE)
    for var in ['TP', 'Z500', 'T2M']:
        if f'{var}_SPIRE_W1_fcst' in d.files:
            make_figure(d, var)
        else:
            print(f'SKIP {var} — no data')
    print('ALL_DONE', flush=True)
