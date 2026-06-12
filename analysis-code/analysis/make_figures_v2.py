"""
Polished publication figures (v2) using CORRECTED 24-h ERA5 precipitation.
TP from skill_tp_corrected.csv (true daily totals), Z500 from skill_per_init_full.csv.

  Fig 2  Skill horizon 2x2 (rows TP/Z500, cols PCC/RMSE)   [full metrics now]
  Fig 3  SPIRE - FuXi 2x2 (dPCC/dRMSE for TP and Z500)
  Fig 4  Regional scorecard (diverging colormap, both variables)
  Fig 5  Bias vs lead (Z500 and precipitation; both real now)
  Fig 6  MME + persistence 2x2
  Fig 9  Per-IMD-region PCC vs lead
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7',
       'MME': '#000000', 'Persistence': '#7f7f7f'}
STY = {'SPIRE': ('-', 's'), 'FuXi': ('-', 'o'), 'ECMWF': ('--', '^'), 'NCEP': ('-.', 'd'),
       'MME': ('-', '*'), 'Persistence': (':', 'x')}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP', 'MME': 'MME', 'Persistence': 'Persistence'}
CORE = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
REG = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'northwest_india': 'Northwest', 'central_india': 'Central',
        'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'axes.labelweight': 'bold', 'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.grid': True, 'grid.alpha': 0.35, 'grid.linestyle': ':', 'savefig.dpi': 300,
    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

z = pd.read_csv(f'{ADIR}/skill_per_init_full.csv'); z = z[z.variable == 'Z500']
tp = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv'); tp = tp[tp.variable == 'TP']
df = pd.concat([tp, z], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig); print('  wrote', name, flush=True)


def line(ax, variable, metric, models, refline=None, ribbon=True):
    for m in models:
        sub = df[(df.variable == variable) & (df.region == 'All India') & (df.model == m)]
        xs, ys, lo, hi = [], [], [], []
        for wk in range(1, 7):
            v = sub[sub.wk == wk][metric].values
            mean, l, h = bootstrap_ci(v)
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean); lo.append(l); hi.append(h)
        ls, mk = STY[m]
        lw = 3 if m == 'MME' else 2.3
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=lw, ms=7, label=LAB[m], zorder=6 if m == 'MME' else 4)
        if ribbon and (m in CORE or m == 'MME'):
            ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.13, lw=0)
    if refline is not None:
        ax.axhline(refline, color='0.2', lw=1.2, ls=(0, (4, 3)))
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)]); ax.set_xlim(0.7, 6.3)


# ---- Fig 2: 2x2 horizon ----
def fig2():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    line(ax[0, 0], 'TP', 'pcc', CORE, refline=0.5); ax[0, 0].set_title('(a) Precipitation — PCC')
    ax[0, 0].set_ylabel('Pattern correlation'); ax[0, 0].set_ylim(-0.3, 1.0)
    line(ax[0, 1], 'TP', 'rmse', CORE); ax[0, 1].set_title('(b) Precipitation — RMSE')
    ax[0, 1].set_ylabel('RMSE (mm day$^{-1}$)')
    line(ax[1, 0], 'Z500', 'pcc', CORE, refline=0.5); ax[1, 0].set_title('(c) Z500 — PCC')
    ax[1, 0].set_ylabel('Pattern correlation'); ax[1, 0].set_ylim(-0.1, 1.02)
    line(ax[1, 1], 'Z500', 'rmse', CORE); ax[1, 1].set_title('(d) Z500 — RMSE'); ax[1, 1].set_ylabel('RMSE (m)')
    ax[0, 0].legend(loc='upper right', ncol=2, frameon=True, framealpha=0.9)
    fig.suptitle('Weekly S2S skill horizon over India (land, cosine-weighted, ensemble means) — JFM 2026',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.text(0.5, -0.01, 'Precipitation verified against true 24-h ERA5 daily totals (ARCO-ERA5). '
             'Shading: bootstrap 95% CI over 13 inits; dashed line: PCC = 0.5.', ha='center', fontsize=10, style='italic')
    fig.tight_layout(rect=[0, 0.01, 1, 0.98]); save(fig, 'fig02_skill_horizon')


# ---- Fig 3: SPIRE - FuXi 2x2 ----
def fig3():
    panels = [('TP', 'pcc', 'ΔPCC', '(a) Precipitation — ΔPCC', False),
              ('TP', 'rmse', 'ΔRMSE (mm day$^{-1}$)', '(b) Precipitation — ΔRMSE', True),
              ('Z500', 'pcc', 'ΔPCC', '(c) Z500 — ΔPCC', False),
              ('Z500', 'rmse', 'ΔRMSE (m)', '(d) Z500 — ΔRMSE', True)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (var, met, ylab, title, lower_better) in zip(axes.ravel(), panels):
        wks, mean, err_lo, err_hi, sig = [], [], [], [], []
        for wk in range(1, 7):
            a = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'SPIRE') & (df.wk == wk)]
            b = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'FuXi') & (df.wk == wk)]
            md, lo, hi, p = paired_bootstrap_diff(dict(zip(a.init_date, a[met])), dict(zip(b.init_date, b[met])))
            if np.isfinite(md):
                wks.append(wk); mean.append(md); err_lo.append(md - lo); err_hi.append(hi - md); sig.append(p < 0.05)
        cols = []
        for md in mean:
            spire_better = (md < 0) if lower_better else (md > 0)
            cols.append(COL['SPIRE'] if spire_better else COL['FuXi'])
        ax.bar(wks, mean, yerr=[err_lo, err_hi], color=cols, capsize=4, edgecolor='k', lw=0.6, alpha=0.9)
        for x, md, eh, s in zip(wks, mean, err_hi, sig):
            if s:
                ax.text(x, md + np.sign(md) * (eh + 0.02 * (max(map(abs, mean)) or 1)), '*', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
        ax.set_ylabel(ylab); ax.set_title(title)
    fig.suptitle('SPIRE versus FuXi-S2S, paired (both ingest & verify against ERA5) — JFM 2026', fontsize=14, fontweight='bold', y=0.995)
    fig.text(0.5, -0.01, 'Orange = SPIRE better, blue = FuXi better. * : paired-bootstrap 95% CI excludes zero.', ha='center', fontsize=10, style='italic')
    fig.tight_layout(rect=[0, 0.01, 1, 0.98]); save(fig, 'fig03_spire_vs_fuxi')


# ---- Fig 4: regional scorecard (diverging) ----
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, var in zip(axes, ['TP', 'Z500']):
        M = np.array([[df[(df.variable == var) & (df.region == rg) & (df.model == m) & (df.wk <= 4)]['pcc'].mean()
                       for m in CORE] for rg in REG])
        im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
        ax.set_xticks(range(len(CORE))); ax.set_xticklabels([LAB[m] for m in CORE])
        ax.set_yticks(range(len(REG))); ax.set_yticklabels([REGL[r] for r in REG])
        for i in range(len(REG)):
            for j in range(len(CORE)):
                ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontweight='bold',
                        color='white' if abs(M[i, j]) > 0.45 else 'black')
        ax.set_title(f'{"Precipitation" if var=="TP" else "Z500"} — mean PCC (Weeks 1–4)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='PCC')
    fig.suptitle('Regional skill scorecard by IMD homogeneous region — JFM 2026', fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(); save(fig, 'fig04_regional_scorecard')


# ---- Fig 5: bias vs lead ----
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, (var, unit, ttl) in zip(ax, [('Z500', 'm', '(a) Z500 mean bias'), ('TP', 'mm day$^{-1}$', '(b) Precipitation mean bias')]):
        for m in CORE:
            sub = df[(df.variable == var) & (df.region == 'All India') & (df.model == m)]
            xs, ys, lo, hi = [], [], [], []
            for wk in range(1, 7):
                mean, l, h = bootstrap_ci(sub[sub.wk == wk]['bias'].values)
                if np.isfinite(mean):
                    xs.append(wk); ys.append(mean); lo.append(l); hi.append(h)
            ls, mk = STY[m]
            a.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.3, ms=6, label=LAB[m])
            a.fill_between(xs, lo, hi, color=COL[m], alpha=0.12, lw=0)
        a.axhline(0, color='k', lw=1)
        a.set_xticks(range(1, 7)); a.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
        a.set_ylabel(f'bias ({unit})'); a.set_title(ttl)
    ax[0].legend(fontsize=10)
    fig.suptitle('Systematic bias vs lead — JFM 2026 (precipitation vs true 24-h ERA5)', fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(); save(fig, 'fig05_bias')


# ---- Fig 6: MME + persistence 2x2 ----
def fig6():
    mods = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    line(ax[0, 0], 'TP', 'pcc', mods, refline=0.5); ax[0, 0].set_title('(a) Precipitation — PCC')
    ax[0, 0].set_ylabel('Pattern correlation'); ax[0, 0].set_ylim(-0.4, 1.0)
    line(ax[0, 1], 'TP', 'rmse', mods); ax[0, 1].set_title('(b) Precipitation — RMSE'); ax[0, 1].set_ylabel('RMSE (mm day$^{-1}$)')
    line(ax[1, 0], 'Z500', 'pcc', mods, refline=0.5); ax[1, 0].set_title('(c) Z500 — PCC')
    ax[1, 0].set_ylabel('Pattern correlation'); ax[1, 0].set_ylim(-0.4, 1.02)
    line(ax[1, 1], 'Z500', 'rmse', mods); ax[1, 1].set_title('(d) Z500 — RMSE'); ax[1, 1].set_ylabel('RMSE (m)')
    ax[0, 0].legend(loc='upper right', ncol=2, fontsize=9)
    fig.suptitle('Multi-model ensemble (MME) and persistence baseline — JFM 2026', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); save(fig, 'fig06_mme_persistence')


# ---- Fig 9: regional horizon ----
def fig9():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True)
    for col, rg in enumerate(REG):
        for row, var in enumerate(['TP', 'Z500']):
            ax = axes[row, col]
            for m in CORE:
                sub = df[(df.variable == var) & (df.region == rg) & (df.model == m)]
                xs, ys = [], []
                for wk in range(1, 7):
                    mean, _, _ = bootstrap_ci(sub[sub.wk == wk]['pcc'].values)
                    if np.isfinite(mean):
                        xs.append(wk); ys.append(mean)
                ls, mk = STY[m]
                ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2, ms=5, label=LAB[m])
            ax.axhline(0.5, color='0.3', lw=0.9, ls=(0, (4, 3))); ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
            ax.set_ylim(-0.45, 1.0)
            if row == 0: ax.set_title(REGL[rg])
            if col == 0: ax.set_ylabel(f'{"Precip" if var=="TP" else "Z500"} PCC')
    axes[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Skill by IMD homogeneous region and lead — JFM 2026', fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(); save(fig, 'fig09_regional_horizon')


if __name__ == '__main__':
    print('Building v2 figures (corrected TP) ...', flush=True)
    fig2(); fig3(); fig4(); fig5(); fig6(); fig9()
    print('V2_FIGURES_DONE', flush=True)
