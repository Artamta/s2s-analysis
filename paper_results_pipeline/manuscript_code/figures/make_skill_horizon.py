"""
3-variable publication figures (TP, Z500, T2M). Overwrites the horizon,
SPIRE-FuXi, and scorecard figures with three-variable versions.
TP from skill_tp_corrected.csv, Z500 from skill_per_init_full.csv, T2M from skill_t2m.csv.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
STY = {'SPIRE': ('-', 's'), 'FuXi': ('-', 'o'), 'ECMWF': ('--', '^'), 'NCEP': ('-.', 'd')}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
CORE = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
REG = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'northwest_india': 'Northwest', 'central_india': 'Central', 'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
VARS = ['TP', 'Z500', 'T2M']
VLAB = {'TP': 'Precipitation', 'Z500': 'Z500', 'T2M': 'T2M'}
RUNIT = {'TP': 'mm day$^{-1}$', 'Z500': 'm', 'T2M': 'K'}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
                     'axes.labelweight': 'bold', 'legend.fontsize': 10, 'axes.grid': True, 'grid.alpha': 0.35,
                     'grid.linestyle': ':', 'savefig.dpi': 300, 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

z = pd.read_csv(f'{ADIR}/skill_per_init_full.csv'); z = z[z.variable == 'Z500']
tp = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv'); tp = tp[tp.variable == 'TP']
t2 = pd.read_csv(f'{ADIR}/skill_t2m.csv'); t2 = t2[t2.variable == 'T2M']
df = pd.concat([tp, z, t2], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
# bias-corrected (centered) RMSE: removes systematic bias (key for T2M, where
# sub-daily temporal sampling gives FuXi/ECMWF/NCEP a ~4 K cold bias)
df['crmse'] = np.sqrt(np.clip(df['rmse'] ** 2 - df['bias'] ** 2, 0, None))
RMSE_COL = {'TP': 'rmse', 'Z500': 'rmse', 'T2M': 'crmse'}
RMSE_TTL = {'TP': 'RMSE', 'Z500': 'RMSE', 'T2M': 'RMSE (bias-corrected)'}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig); print('  wrote', name, flush=True)


def line(ax, variable, metric, refline=None):
    for m in CORE:
        sub = df[(df.variable == variable) & (df.region == 'All India') & (df.model == m)]
        xs, ys, lo, hi = [], [], [], []
        for wk in range(1, 7):
            mean, l, h = bootstrap_ci(sub[sub.wk == wk][metric].values)
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean); lo.append(l); hi.append(h)
        ls, mk = STY[m]
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.3, ms=6, label=LAB[m])
        ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.12, lw=0)
    if refline is not None:
        ax.axhline(refline, color='0.2', lw=1.1, ls=(0, (4, 3)))
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)]); ax.set_xlim(0.7, 6.3)


def fig2():
    fig, ax = plt.subplots(3, 2, figsize=(13, 13))
    for r, v in enumerate(VARS):
        line(ax[r, 0], v, 'pcc', refline=0.5); ax[r, 0].set_ylabel('PCC')
        ax[r, 0].set_title(f'({chr(97+2*r)}) {VLAB[v]} — PCC')
        line(ax[r, 1], v, RMSE_COL[v]); ax[r, 1].set_ylabel(f'{RMSE_TTL[v]} ({RUNIT[v]})')
        ax[r, 1].set_title(f'({chr(98+2*r)}) {VLAB[v]} — {RMSE_TTL[v]}')
    ax[0, 0].set_ylim(-0.3, 1.0); ax[1, 0].set_ylim(-0.1, 1.02); ax[2, 0].set_ylim(0, 1.02)
    ax[0, 0].legend(loc='upper right', ncol=2)
    fig.suptitle('Weekly S2S skill horizon over India: precipitation, Z500, and T2M — JFM 2026',
                 fontsize=14, fontweight='bold', y=0.997)
    fig.text(0.5, -0.005, 'Land points, cosine-weighted, ensemble means. Shading: bootstrap 95% CI over 13 inits; dashed: PCC = 0.5.',
             ha='center', fontsize=10, style='italic')
    fig.tight_layout(rect=[0, 0.005, 1, 0.99]); save(fig, 'fig02_skill_horizon')


def fig3():
    fig, ax = plt.subplots(3, 2, figsize=(13, 13))
    for r, v in enumerate(VARS):
        for c, (mlabel, lower) in enumerate([('pcc', False), ('rmse', True)]):
            met = 'pcc' if mlabel == 'pcc' else RMSE_COL[v]
            a = ax[r, c]
            wks, mn, elo, ehi, sig = [], [], [], [], []
            for wk in range(1, 7):
                A = df[(df.variable == v) & (df.region == 'All India') & (df.model == 'SPIRE') & (df.wk == wk)]
                B = df[(df.variable == v) & (df.region == 'All India') & (df.model == 'FuXi') & (df.wk == wk)]
                md, lo, hi, p = paired_bootstrap_diff(dict(zip(A.init_date, A[met])), dict(zip(B.init_date, B[met])))
                if np.isfinite(md):
                    wks.append(wk); mn.append(md); elo.append(md - lo); ehi.append(hi - md); sig.append(p < 0.05)
            cols = [COL['SPIRE'] if ((md < 0) if lower else (md > 0)) else COL['FuXi'] for md in mn]
            a.bar(wks, mn, yerr=[elo, ehi], color=cols, capsize=3, edgecolor='k', lw=0.5, alpha=0.9)
            mxabs = max(map(abs, mn)) if mn else 1
            for x, md, eh, s in zip(wks, mn, ehi, sig):
                if s:
                    a.text(x, md + np.sign(md) * (eh + 0.04 * mxabs), '*', ha='center', fontsize=14, fontweight='bold')
            a.axhline(0, color='k', lw=1); a.set_xticks(range(1, 7)); a.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
            a.set_ylabel(('ΔPCC' if mlabel == 'pcc' else f'ΔRMSE ({RUNIT[v]})'))
            a.set_title(f'({chr(97+2*r+c)}) {VLAB[v]} — Δ{mlabel.upper()}')
    fig.suptitle('SPIRE − FuXi paired comparison across three variables — JFM 2026', fontsize=14, fontweight='bold', y=0.997)
    fig.text(0.5, -0.005, 'Orange = SPIRE better, blue = FuXi better. * : paired-bootstrap 95% CI excludes zero.', ha='center', fontsize=10, style='italic')
    fig.tight_layout(rect=[0, 0.005, 1, 0.99]); save(fig, 'fig03_spire_vs_fuxi')


def fig4():
    import matplotlib.cm as cm
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    cmap = plt.get_cmap('RdYlGn'); cmap.set_bad('0.85')
    for k, (ax, v) in enumerate(zip(axes, VARS)):
        M = np.array([[df[(df.variable == v) & (df.region == rg) & (df.model == m) & (df.wk <= 4)]['pcc'].mean() for m in CORE] for rg in REG])
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=0.8, aspect='auto')
        # highlight the best system per region with a bold border
        for i in range(4):
            j = int(np.nanargmax(M[i]))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='k', lw=2.4, zorder=5))
        ax.set_xticks(range(4)); ax.set_xticklabels([LAB[m] for m in CORE], rotation=20, ha='right', fontsize=11)
        ax.set_yticks(range(4)); ax.set_yticklabels([REGL[r] for r in REG], fontsize=11)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontweight='bold', fontsize=12,
                        color='white' if M[i, j] < 0.35 else 'black')
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which='minor', color='w', lw=2); ax.tick_params(which='minor', length=0)
        ax.set_title(f'({chr(97+k)}) {VLAB[v]}', fontsize=13, fontweight='bold')
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02); cb.set_label('mean PCC, weeks 1–4', fontsize=11)
    fig.suptitle('Regional skill scorecard by IMD homogeneous region (best system boxed) — JFM 2026',
                 fontsize=14, fontweight='bold', y=1.0)
    save(fig, 'fig04_regional_scorecard')


if __name__ == '__main__':
    print('Building 3-variable figures ...', flush=True)
    fig2(); fig3(); fig4()
    print('V3_FIGURES_DONE', flush=True)
