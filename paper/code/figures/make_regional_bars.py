"""
Region-wise PCC and RMSE bar charts (visual companion to the regional tables).
Rows = variables (TP, Z500, T2M); columns = PCC (left) and RMSE (right). Within
each panel: x = IMD homogeneous regions, grouped bars per system. Weeks 1-4 mean,
13-init average. T2M RMSE uses the bias-corrected (centered) RMSE.
TP from skill_tp_corrected.csv, Z500 from skill_per_init_full.csv, T2M from skill_t2m.csv.
-> paper/figs/fig19_regional_bars.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
CORE = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
REG = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'northwest_india': 'NW', 'central_india': 'Central', 'south_peninsula': 'S.Pen', 'east_northeast_india': 'E/NE'}
VARS = ['TP', 'Z500', 'T2M']
VLAB = {'TP': 'Precipitation', 'Z500': 'Z500', 'T2M': 'T2M'}
RUNIT = {'TP': 'mm day$^{-1}$', 'Z500': 'm', 'T2M': 'K'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'savefig.dpi': 300, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})

z = pd.read_csv(f'{ADIR}/skill_per_init_full.csv'); z = z[z.variable == 'Z500']
tp = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv'); tp = tp[tp.variable == 'TP']
t2 = pd.read_csv(f'{ADIR}/skill_t2m.csv'); t2 = t2[t2.variable == 'T2M']
df = pd.concat([tp, z, t2], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
df['crmse'] = np.sqrt(np.clip(df['rmse'] ** 2 - df['bias'] ** 2, 0, None))


def val(var, rg, m, col):
    return df[(df.variable == var) & (df.region == rg) & (df.model == m) & (df.wk <= 4)][col].mean()


fig, ax = plt.subplots(3, 2, figsize=(14, 12))
x = np.arange(len(REG)); w = 0.2
for r, v in enumerate(VARS):
    rmse_col = 'crmse' if v == 'T2M' else 'rmse'
    for c, (col, lab) in enumerate([('pcc', 'PCC'), (rmse_col, f'RMSE ({RUNIT[v]})')]):
        a = ax[r, c]
        for k, m in enumerate(CORE):
            vals = [val(v, rg, m, col) for rg in REG]
            a.bar(x + (k - 1.5) * w, vals, w, color=COL[m], edgecolor='k', lw=0.4, label=LAB[m])
        a.set_xticks(x); a.set_xticklabels([REGL[rg] for rg in REG])
        a.set_title(f'({chr(97 + 2 * r + c)}) {VLAB[v]} — {"PCC" if c == 0 else "RMSE"}')
        a.set_ylabel(lab); a.grid(axis='y', ls=':', alpha=0.5)
        if c == 0:
            a.axhline(0.5, color='0.3', ls='--', lw=1); a.set_ylim(0, 1.0)
        a.set_title(f'({chr(97 + 2 * r + c)}) {VLAB[v]} — {"PCC" if c == 0 else "RMSE (bias-corrected)" if v == "T2M" else "RMSE"}')
ax[0, 0].legend(loc='upper right', ncol=2, fontsize=9)
fig.suptitle('Region-wise skill by IMD homogeneous region (weeks 1–4 mean, 13-init average) — JFM 2026',
             fontsize=14, fontweight='bold', y=1.0)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig19_regional_bars.{ext}', bbox_inches='tight')
print('WROTE fig19_regional_bars', flush=True)
print('REGIONAL_BARS_DONE', flush=True)
