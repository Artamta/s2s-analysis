"""
Two manuscript figures for the JFM-2026 SPIRE benchmark:

  Fig 1  India verification domain + IMD homogeneous regions (cartopy map).
  Fig 5  Systematic bias: Z500 mean bias vs lead, and precipitation mean bias
         vs lead (against the true 24-h daily ERA5; FuXi unit-harmonised).

The three-variable skill-horizon, SPIRE-vs-FuXi, and regional-scorecard figures
(fig02/fig03/fig04) are produced by make_skill_horizon.py and intentionally NOT
duplicated here. Driven by analysis/skill_per_init_full.csv (+ skill_tp_corrected.csv
for the precipitation panel).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import bootstrap_ci

CSV = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/skill_per_init_full.csv'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
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
