"""
Probabilistic verification figures (Gaussian-forecast framework; reference:
TianXing-S2S Fig 1 SSR/CRPS scorecards):
  fig15_crpss        : CRPSS vs lead, 4 systems, per variable (skill vs climatology).
  fig16_spread_skill : spread-skill ratio (SSR) vs lead with the SSR=1 calibration
                       line; SPIRE best-calibrated, FuXi-S2S strongly overconfident.
  fig17_reliability  : reliability diagrams for a heavy-rain day (precip>1 mm/day)
                       and a cold day (T2M below the climatological lower tercile).
Reads analysis/prob_skill.csv and analysis/reliability.npz.
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.verification_extra import bootstrap_ci

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
STY = {'SPIRE': ('-', 's'), 'FuXi': ('-', 'o'), 'ECMWF': ('--', '^'), 'NCEP': ('-.', 'd')}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
CORE = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
VARS = ['TP', 'Z500', 'T2M']
VLAB = {'TP': 'Precipitation', 'Z500': 'Z500', 'T2M': 'T2M'}
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 12, 'legend.fontsize': 10, 'axes.grid': True, 'grid.alpha': 0.35,
                     'grid.linestyle': ':', 'savefig.dpi': 300, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

df = pd.read_csv(f'{ADIR}/prob_skill.csv'); df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
ai = df[df.region == 'All India']


def line(ax, var, metric):
    for m in CORE:
        sub = ai[(ai.variable == var) & (ai.model == m)]
        xs, ys, lo, hi = [], [], [], []
        for wk in range(1, 7):
            mn, l, h = bootstrap_ci(sub[sub.wk == wk][metric].values)
            if np.isfinite(mn):
                xs.append(wk); ys.append(mn); lo.append(l); hi.append(h)
        ls, mk = STY[m]
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=2.3, ms=6, label=LAB[m])
        ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.12, lw=0)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'Wk{i}' for i in range(1, 7)]); ax.set_xlim(0.7, 6.3)


def fig_crpss():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, v in enumerate(VARS):
        line(ax[i], v, 'crpss'); ax[i].axhline(0, color='k', lw=1.1, ls='--')
        ax[i].set_title(f'({chr(97+i)}) {VLAB[v]}'); ax[i].set_ylabel('CRPSS (vs climatology)' if i == 0 else '')
        ax[i].set_ylim(-0.6, 0.9)
    ax[0].legend(loc='upper right', ncol=2)
    fig.suptitle('Probabilistic skill: continuous ranked probability skill score vs lead — JFM 2026',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, 'CRPSS $>0$ beats climatology; dashed line = no skill. Gaussian(ensemble mean, spread) forecasts, daily-level, land points.',
             ha='center', fontsize=10, style='italic')
    fig.tight_layout();
    for ext in ('pdf', 'png'): fig.savefig(f'{FIGDIR}/fig15_crpss.{ext}', bbox_inches='tight')
    plt.close(fig); print('wrote fig15_crpss', flush=True)


def fig_ssr():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, v in enumerate(VARS):
        line(ax[i], v, 'ssr'); ax[i].axhline(1.0, color='k', lw=1.2, ls='--')
        ax[i].set_title(f'({chr(97+i)}) {VLAB[v]}'); ax[i].set_ylabel('spread-skill ratio (SSR)' if i == 0 else '')
        ax[i].set_ylim(0, 1.5)
    ax[0].legend(loc='lower right', ncol=2)
    fig.suptitle('Ensemble calibration: spread-skill ratio vs lead — JFM 2026', fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, 'SSR $=$ ensemble spread / ensemble-mean RMSE; dashed line (SSR$=1$) is perfect calibration, '
             'SSR$<1$ is overconfident (under-dispersed).', ha='center', fontsize=10, style='italic')
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(f'{FIGDIR}/fig16_spread_skill.{ext}', bbox_inches='tight')
    plt.close(fig); print('wrote fig16_spread_skill', flush=True)


def fig_reliability():
    r = np.load(f'{ADIR}/reliability.npz')
    NB = int(r['nbins']); centers = (np.arange(NB) + 0.5) / NB
    events = [('tp_wet', 'Heavy-rain day (precip $>$ 1 mm day$^{-1}$)'),
              ('t2_cold', 'Cold day (T2M $<$ climatological lower tercile)')]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    for i, (ev, title) in enumerate(events):
        ax[i].plot([0, 1], [0, 1], 'k--', lw=1)
        for m in CORE:
            arr = r[f'{ev}__{m}']  # (3, NB): sum_out, cnt, sum_p
            cnt = arr[1]; ok = cnt > 0
            obs_freq = np.where(ok, arr[0] / np.maximum(cnt, 1), np.nan)
            fcst_p = np.where(ok, arr[2] / np.maximum(cnt, 1), np.nan)
            ls, mk = STY[m]
            ax[i].plot(fcst_p[ok], obs_freq[ok], ls, marker=mk, color=COL[m], lw=2, ms=6, label=LAB[m])
        ax[i].set_xlim(0, 1); ax[i].set_ylim(0, 1); ax[i].set_aspect('equal')
        ax[i].set_xlabel('forecast probability'); ax[i].set_ylabel('observed frequency' if i == 0 else '')
        ax[i].set_title(f'({chr(97+i)}) {title}', fontsize=11)
    ax[0].legend(loc='upper left')
    fig.suptitle('Reliability of probabilistic event forecasts — JFM 2026', fontsize=14, fontweight='bold', y=1.0)
    fig.text(0.5, -0.01, 'Points on the diagonal are perfectly reliable; below the diagonal = overforecasting.',
             ha='center', fontsize=10, style='italic')
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(f'{FIGDIR}/fig17_reliability.{ext}', bbox_inches='tight')
    plt.close(fig); print('wrote fig17_reliability', flush=True)


if __name__ == '__main__':
    fig_crpss(); fig_ssr(); fig_reliability()
    # headline numbers
    print('\n=== CRPSS / SSR (All India, weeks 1-3 mean) ===')
    for v in VARS:
        print(f'\n{v}:')
        for m in CORE:
            s = ai[(ai.variable == v) & (ai.model == m) & (ai.wk <= 3)]
            print(f'  {m:6s} CRPSS={s.crpss.mean():.2f}  SSR={s.ssr.mean():.2f}')
    print('PROB_FIGS_DONE', flush=True)
