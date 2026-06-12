"""
Publication figures for the JFM-2026 SPIRE benchmark, driven entirely by
analysis/skill_per_init_full.csv (written by compute_skill_horizon_final.py).

  Fig 1  India domain + IMD homogeneous regions (cartopy)
  Fig 2  Dual-variable skill horizon  (TP PCC | Z500 PCC | Z500 RMSE)
  Fig 3  SPIRE - FuXi paired head-to-head (DeltaPCC TP, DeltaPCC/DeltaRMSE Z500)
  Fig 4  Regional scorecard (IMD regions x models; PCC for TP and Z500)
  Fig 5  Bias: Z500 mean bias vs lead + TP wet/dry tendency (qualitative, caveated)

TP shows PCC only (ERA5 tp on disk is a 6-h window/day -> absolute mm/day not
cross-comparable; PCC is scale-invariant). Z500 carries full PCC+RMSE+bias.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

CSV = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/skill_per_init_full.csv'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
os.makedirs(FIGDIR, exist_ok=True)
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
STY = {'SPIRE': ('-', 's'), 'FuXi': ('-', 'o'), 'ECMWF': ('--', '^'), 'NCEP': ('-.', 'd')}
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
REG_LABEL = {'northwest_india': 'Northwest', 'central_india': 'Central',
             'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelweight': 'bold'})

df = pd.read_csv(CSV)
# use the unit-corrected TP (true daily ERA5; FuXi mm/h->mm/day) for any TP panel
import os as _os
_tpc = f'{_os.path.dirname(CSV)}/skill_tp_corrected.csv'
if _os.path.exists(_tpc):
    df = df[df.variable != 'TP']
    df = pd.concat([df, pd.read_csv(_tpc)], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {name}.pdf/.png', flush=True)


def horizon(ax, variable, metric, region='All India', refline=None, ylabel=''):
    for m in MODELS:
        sub = df[(df.variable == variable) & (df.region == region) & (df.model == m)]
        if sub.empty:
            continue
        xs, ys, los, his = [], [], [], []
        for wk in sorted(sub.wk.unique()):
            vals = sub[sub.wk == wk][metric].values
            mean, lo, hi = bootstrap_ci(vals)
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean); los.append(lo); his.append(hi)
        ls, mk = STY[m]
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.4, ms=7, label=m)
        ax.fill_between(xs, los, his, color=COL[m], alpha=0.15)
    if refline is not None:
        ax.axhline(refline, color='k', lw=1.2, ls=':')
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
    ax.set_xlim(0.6, 6.4); ax.grid(axis='y', ls=':', alpha=0.6); ax.set_ylabel(ylabel)


# ---------------- Figure 1: domain ----------------
def fig1():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([60, 102, 3, 40], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#e8e4d8')
        ax.add_feature(cfeature.OCEAN, facecolor='#cfe2f3')
        ax.add_feature(cfeature.COASTLINE, lw=0.6)
        boxes = {'Northwest': (68, 82, 22, 38, '#D55E00'), 'Central': (72, 89, 18, 28, '#009E73'),
                 'S. Peninsula': (72, 85, 8, 20, '#0072B2'), 'East/NE': (85, 98, 20, 30, '#CC79A7')}
        for name, (lo, hi, la, ha, c) in boxes.items():
            ax.add_patch(Rectangle((lo, la), hi - lo, ha - la, fill=False, edgecolor=c, lw=2.2,
                                   transform=ccrs.PlateCarree(), zorder=5))
            ax.text(lo + 0.3, ha - 1.3, name, color=c, fontsize=9, fontweight='bold',
                    transform=ccrs.PlateCarree(), zorder=6)
        ax.add_patch(Rectangle((65, 5), 35, 33, fill=False, edgecolor='k', lw=1.4, ls='--',
                               transform=ccrs.PlateCarree(), zorder=4))
        gl = ax.gridlines(draw_labels=True, lw=0.3, color='gray', alpha=0.4)
        gl.top_labels = gl.right_labels = False
        ax.set_title('India verification domain (5–38°N, 65–100°E) and IMD homogeneous regions\n'
                     '13 weekly initializations, 1 Jan – 26 Mar 2026 · land points only')
        save(fig, 'fig01_domain')
    except Exception as e:
        print('  fig1 FAILED:', e, flush=True)


# ---------------- Figure 2: dual-variable horizon ----------------
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    horizon(axes[0], 'TP', 'pcc', refline=0.5, ylabel='Pattern correlation (PCC)')
    axes[0].set_title('(a) Total precipitation — PCC'); axes[0].set_ylim(-0.2, 1.0)
    horizon(axes[1], 'Z500', 'pcc', refline=0.5, ylabel='Pattern correlation (PCC)')
    axes[1].set_title('(b) Z500 — PCC'); axes[1].set_ylim(0, 1.02)
    horizon(axes[2], 'Z500', 'rmse', ylabel='RMSE (m)')
    axes[2].set_title('(c) Z500 — RMSE')
    axes[0].legend(loc='lower left', fontsize=10, framealpha=0.9)
    fig.suptitle('Weekly S2S skill horizon over India (land, cosine-weighted, ensemble means) — JFM 2026',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.04, 'TP shown as PCC only (scale-invariant): ERA5 tp on disk is a 6-h window/day; '
             'dashed line = PCC 0.5. Shading = bootstrap 95% CI over 13 inits.', ha='center', fontsize=9, style='italic')
    fig.tight_layout()
    save(fig, 'fig02_skill_horizon')


# ---------------- Figure 3: SPIRE - FuXi ----------------
def fig3():
    panels = [('TP', 'pcc', 'ΔPCC (SPIRE − FuXi)', '(a) Total precipitation — ΔPCC'),
              ('Z500', 'pcc', 'ΔPCC (SPIRE − FuXi)', '(b) Z500 — ΔPCC'),
              ('Z500', 'rmse', 'ΔRMSE (SPIRE − FuXi), m', '(c) Z500 — ΔRMSE (negative = SPIRE better)')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (var, met, ylab, title) in zip(axes, panels):
        wks, means, los, his, sig = [], [], [], [], []
        for wk in range(1, 7):
            a = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'SPIRE') & (df.wk == wk)]
            b = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'FuXi') & (df.wk == wk)]
            da = dict(zip(a.init_date, a[met])); db = dict(zip(b.init_date, b[met]))
            md, lo, hi, p = paired_bootstrap_diff(da, db)
            if np.isfinite(md):
                wks.append(wk); means.append(md); los.append(md - lo); his.append(hi - md); sig.append(p < 0.05)
        colors = ['#D55E00' if (('rmse' in met) == (m < 0)) else '#0072B2' for m in means]
        # for PCC: positive => SPIRE better (orange). for RMSE: negative => SPIRE better (orange).
        colors = []
        for m in means:
            spire_better = (m > 0) if met == 'pcc' else (m < 0)
            colors.append('#D55E00' if spire_better else '#0072B2')
        ax.bar(wks, means, yerr=[los, his], color=colors, capsize=4, edgecolor='k', lw=0.6)
        for x, m, s in zip(wks, means, sig):
            if s: ax.text(x, m + (his[wks.index(x)] + 0.01) * np.sign(m or 1), '*', ha='center', fontsize=15, fontweight='bold')
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)])
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis='y', ls=':', alpha=0.6)
    fig.suptitle('SPIRE vs FuXi paired head-to-head over India (both ingest & verify against ERA5) — JFM 2026',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.04, 'Orange = SPIRE better; blue = FuXi better. * = paired bootstrap 95% CI excludes zero.',
             ha='center', fontsize=9, style='italic')
    fig.tight_layout()
    save(fig, 'fig03_spire_vs_fuxi')


# ---------------- Figure 4: regional scorecard ----------------
def fig4():
    regions = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, ['TP', 'Z500']):
        M = np.full((len(regions), len(MODELS)), np.nan)
        for i, rg in enumerate(regions):
            for j, m in enumerate(MODELS):
                sub = df[(df.variable == var) & (df.region == rg) & (df.model == m) & (df.wk <= 4)]
                if not sub.empty:
                    M[i, j] = sub['pcc'].mean()
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=20)
        ax.set_yticks(range(len(regions))); ax.set_yticklabels([REG_LABEL[r] for r in regions])
        for i in range(len(regions)):
            for j in range(len(MODELS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                            color='white' if M[i, j] < 0.6 else 'black', fontweight='bold')
        ax.set_title(f'{var} — mean PCC (Weeks 1–4)')
        fig.colorbar(im, ax=ax, fraction=0.046, label='PCC')
    fig.suptitle('Regional skill scorecard by IMD homogeneous region — JFM 2026',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig04_regional_scorecard')


# ---------------- Figure 5: bias ----------------
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) Z500 bias vs lead
    for m in MODELS:
        sub = df[(df.variable == 'Z500') & (df.region == 'All India') & (df.model == m)]
        xs, ys, los, his = [], [], [], []
        for wk in sorted(sub.wk.unique()):
            mean, lo, hi = bootstrap_ci(sub[sub.wk == wk]['bias'].values)
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean); los.append(lo); his.append(hi)
        ls, mk = STY[m]
        axes[0].plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.2, ms=6, label=m)
        axes[0].fill_between(xs, los, his, color=COL[m], alpha=0.12)
    axes[0].axhline(0, color='k', lw=1)
    axes[0].set_xticks(range(1, 7)); axes[0].set_xticklabels([f'Wk{i}' for i in range(1, 7)])
    axes[0].set_ylabel('Z500 mean bias (m)'); axes[0].set_title('(a) Z500 mean bias vs lead')
    axes[0].grid(axis='y', ls=':', alpha=0.6); axes[0].legend(fontsize=10)
    # (b) TP mean bias vs lead (against true 24-h daily ERA5; FuXi unit-harmonized)
    for m in MODELS:
        sub = df[(df.variable == 'TP') & (df.region == 'All India') & (df.model == m)]
        xs, ys, los, his = [], [], [], []
        for wk in sorted(sub.wk.unique()):
            mean, lo, hi = bootstrap_ci(sub[sub.wk == wk]['bias'].values)
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean); los.append(lo); his.append(hi)
        ls, mk = STY[m]
        axes[1].plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.2, ms=6, label=m)
        axes[1].fill_between(xs, los, his, color=COL[m], alpha=0.12)
    axes[1].axhline(0, color='k', lw=1)
    axes[1].set_xticks(range(1, 7)); axes[1].set_xticklabels([f'Wk{i}' for i in range(1, 7)])
    axes[1].set_ylabel('Precipitation mean bias (mm day$^{-1}$)')
    axes[1].set_title('(b) Precipitation mean bias vs lead')
    axes[1].grid(axis='y', ls=':', alpha=0.6); axes[1].legend(fontsize=10)
    fig.suptitle('Systematic bias diagnostics — JFM 2026', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig05_bias')


if __name__ == '__main__':
    print('Generating figures from', CSV, flush=True)
    fig1(); fig5()  # fig02-04 are produced by make_skill_horizon.py (3-variable versions)
    print('FIGURES_DONE', flush=True)
