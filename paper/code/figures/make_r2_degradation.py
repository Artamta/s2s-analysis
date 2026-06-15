"""
FIG 13 – R² degradation with lead week.
One figure, 3 panels (TP, Z500, T2M): R² vs lead-week for all 4 models.
Quick, clean summary of how scatter tightness decays.
Reads: paper/results/scatter_points_weekwise.npz
Writes: paper/figs/fig13_r2_degradation.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATAFILE = '/home/raj.ayush/s2s/s2s_anlysis/paper/results/scatter_points_weekwise.npz'
FIGDIR   = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
MODELS   = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']

STYLE = {
    'SPIRE': dict(color='#1b9e77', marker='o', ls='-',  lw=2.4, ms=8, zorder=5),
    'FuXi':  dict(color='#d95f02', marker='s', ls='--', lw=2.0, ms=7, zorder=4),
    'ECMWF': dict(color='#7570b3', marker='^', ls='-',  lw=2.4, ms=8, zorder=5),
    'NCEP':  dict(color='#e7298a', marker='D', ls='-.', lw=2.0, ms=7, zorder=4),
}
MLAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
VLABEL = {'TP': 'Precipitation', 'Z500': 'Z500', 'T2M': '2-m Temperature'}

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'savefig.dpi': 600, 'figure.dpi': 120,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
})


def get_r2(d, var, model, wk):
    fk = f'{var}_{model}_W{wk}_fcst'
    ok = f'{var}_{model}_W{wk}_obs'
    if fk not in d.files or ok not in d.files:
        return np.nan
    f, o = d[fk], d[ok]
    if len(f) < 10:
        return np.nan
    r = np.corrcoef(f, o)[0, 1]
    return r**2


if __name__ == '__main__':
    d = np.load(DATAFILE)
    present_vars = [v for v in ['TP', 'Z500', 'T2M'] if f'{v}_SPIRE_W1_fcst' in d.files]
    nv = len(present_vars)

    fig, axes = plt.subplots(1, nv, figsize=(6.5 * nv, 5.5))
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.97, wspace=0.28)
    if nv == 1:
        axes = [axes]

    wks = np.arange(1, 7)

    for ax, var in zip(axes, present_vars):
        for m in MODELS:
            r2_vals = [get_r2(d, var, m, w) for w in wks]
            ax.plot(wks, r2_vals, label=MLAB[m], **STYLE[m])

        ax.set_xlabel('Lead Week', fontsize=13, fontweight='bold')
        ax.set_ylabel('$R^2$', fontsize=13, fontweight='bold')
        ax.set_title(VLABEL[var], fontsize=14, pad=10)
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(wks)
        ax.set_xticklabels([f'W{w}' for w in wks], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.axhline(0.5, color='0.5', ls=':', lw=0.8, zorder=0)
        ax.grid(axis='y', alpha=0.3, lw=0.5)
        ax.legend(fontsize=11, framealpha=0.9, edgecolor='0.7', loc='lower left')

    fig.suptitle('Forecast\u2013Observation $R^2$ Degradation with Lead Week\nIndia land only \u00b7 JFM 2026',
                 fontsize=15, fontweight='bold')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/fig13_r2_degradation.{ext}', bbox_inches='tight', dpi=600)
    plt.close(fig)
    print('WROTE fig13_r2_degradation', flush=True)
