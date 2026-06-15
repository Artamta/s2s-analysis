"""
Publication-quality S2S skill-horizon figures (TP, Z500, T2M).
TP from skill_tp_corrected.csv, Z500 from skill_per_init_full.csv.

Designed for high-impact journal submission (≥600 DPI, vector PDF).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper/code')
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

ADIR   = '/home/raj.ayush/s2s/s2s_anlysis/paper/results'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper/figs'

# ── Refined colour palette (print-safe, colourblind-considered) ───────────────
COL = {
    'SPIRE':       '#D45E00',   # burnt orange — warm, distinct
    'FuXi':        '#2171B5',   # steel blue
    'ECMWF':       '#238B45',   # forest green
    'NCEP':        '#7B2D8E',   # deep purple
    'MME':         '#222222',   # near-black
    'Persistence': '#999999',   # warm grey
}
# Line styles: (linestyle, marker, linewidth, markersize, markerfacecolor_mode)
# markerfacecolor_mode: 'fill' = solid, 'open' = white interior
STY = {
    'SPIRE':       ('-',  's', 2.6,  7.5, 'fill'),
    'FuXi':        ('-',  'o', 2.6,  7.5, 'fill'),
    'ECMWF':       ('--', '^', 2.0,  7.5, 'open'),
    'NCEP':        ('-.', 'D', 2.0,  6.5, 'open'),
    'MME':         ('-',  'P', 2.8,  8.5, 'fill'),    # plus-filled
    'Persistence': (':',  'X', 1.8,  7.5, 'open'),    # X marker
}
LAB = {
    'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S',
    'ECMWF': 'ECMWF', 'NCEP': 'NCEP',
    'MME': 'MME', 'Persistence': 'Persistence',
}
CORE    = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
ALL_MDL = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']
REG  = ['All India', 'northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'All India': 'All India', 'northwest_india': 'Northwest', 'central_india': 'Central',
        'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
VARS  = ['TP', 'Z500', 'T2M']
VLAB  = {'TP': 'Precipitation', 'Z500': 'Z500', 'T2M': 'T2M'}
RUNIT = {'TP': 'mm day$^{-1}$', 'Z500': 'gpm', 'T2M': 'K'}

# ── Journal-grade rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':            10.5,
    'mathtext.default':     'regular',
    'axes.titlesize':       11.5,
    'axes.titleweight':     'bold',
    'axes.labelsize':       10.5,
    'axes.labelweight':     'medium',
    'axes.linewidth':       0.7,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.color':           '#E8E8E8',
    'grid.linestyle':       '-',
    'grid.linewidth':       0.45,
    'grid.alpha':           1.0,
    'legend.fontsize':      9.0,
    'legend.framealpha':    0.95,
    'legend.edgecolor':     '#CCCCCC',
    'legend.borderpad':     0.6,
    'legend.handlelength':  2.2,
    'legend.handletextpad': 0.6,
    'legend.columnspacing': 1.0,
    'xtick.labelsize':      9.5,
    'ytick.labelsize':      9.5,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.major.size':     3.5,
    'ytick.major.size':     3.5,
    'xtick.minor.size':     2.0,
    'ytick.minor.size':     2.0,
    'savefig.dpi':          600,
    'figure.dpi':           150,
    'figure.facecolor':     'white',
    'savefig.facecolor':    'white',
    'pdf.fonttype':         42,       # TrueType fonts in PDF (editable text)
    'ps.fonttype':          42,
})

# ── Data loading ──────────────────────────────────────────────────────────────
z  = pd.read_csv(f'{ADIR}/skill_per_init_full.csv'); z  = z[z.variable == 'Z500']
tp = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv');  tp = tp[tp.variable == 'TP']
# t2 = pd.read_csv(f'{ADIR}/skill_t2m.csv'); t2 = t2[t2.variable == 'T2M']
df = pd.concat([tp, z], ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
df['crmse'] = np.sqrt(np.clip(df['rmse'] ** 2 - df['bias'] ** 2, 0, None))
RMSE_COL = {'TP': 'rmse', 'Z500': 'rmse', 'T2M': 'crmse'}
RMSE_TTL = {'TP': 'RMSE', 'Z500': 'RMSE', 'T2M': 'RMSE (centred)'}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  wrote {name}', flush=True)


def _polish_ax(ax, xlabel='Forecast lead'):
    """Clean axis styling consistent across all panels."""
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'Week {i}' for i in range(1, 7)])
    ax.set_xlim(0.55, 6.45)
    ax.set_xlabel(xlabel, labelpad=5)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#333333')
    ax.tick_params(axis='both', which='both', direction='out')


def _draw_lines(ax, variable, metric, models=None, refline=None):
    """Skill horizon curves — clean lines, no CI ribbons."""
    if models is None:
        models = CORE
    for m in models:
        sub = df[(df.variable == variable) & (df.region == 'All India') & (df.model == m)]
        if sub.empty:
            continue
        xs, ys = [], []
        for wk in range(1, 7):
            vals = sub[sub.wk == wk][metric].values
            mean = np.nanmean(vals) if len(vals) else np.nan
            if np.isfinite(mean):
                xs.append(wk); ys.append(mean)
        if not xs:
            continue
        ls, mk, lw, ms, mfc_mode = STY[m]
        mfc = COL[m] if mfc_mode == 'fill' else 'white'
        zord = 5 if m == 'SPIRE' else (4 if m == 'MME' else 3)

        # Main line
        ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=lw, ms=ms,
                label=LAB[m], zorder=zord,
                markerfacecolor=mfc, markeredgecolor=COL[m],
                markeredgewidth=1.3 if mfc_mode == 'open' else 0.8,
                path_effects=[pe.Stroke(linewidth=lw + 1.0, foreground='white'),
                              pe.Normal()] if m == 'SPIRE' else None)

    if refline is not None:
        ax.axhline(refline, color='#555555', lw=0.9,
                   ls=(0, (6, 3)), zorder=1, label='_nolegend_')
    _polish_ax(ax)


def _add_legend(ax, ncol=3, loc='upper right'):
    """Compact, refined legend."""
    leg = ax.legend(loc=loc, ncol=ncol, frameon=True, fancybox=False,
                    edgecolor='#CCCCCC', facecolor='white', framealpha=0.96,
                    fontsize=8.5, borderpad=0.5, labelspacing=0.35,
                    handlelength=2.2, handletextpad=0.5, columnspacing=0.8)
    leg.get_frame().set_linewidth(0.6)


def _add_noskill_zone(ax, ymin, threshold=0.5, label=True):
    """Shade below ACC threshold as 'no useful skill' zone."""
    ax.axhspan(ymin, threshold, color='#F7F7F7', zorder=0, lw=0)
    # subtle edge line at threshold
    ax.axhline(threshold, color='#AAAAAA', lw=0.6, ls='-', zorder=0.5)
    if label:
        ax.text(6.40, threshold + 0.015, 'ACC = 0.5', va='bottom', ha='right',
                fontsize=7.5, color='#666666', fontstyle='italic')


def _add_panel_label(ax, label, loc='upper left'):
    """Bold panel label (a, b, c, ...) positioned outside axes at top-left."""
    ax.text(-0.02, 1.06, f'({label})', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='left',
            color='#111111')


def _suptitle(fig, variable):
    fig.suptitle(
        f'JFM 2026 S2S Skill Horizon — {VLAB[variable]}',
        fontsize=13, fontweight='bold', y=1.01, color='#111111'
    )


def _caption(fig):
    fig.text(
        0.5, -0.025,
        'All-India (land only, IMD boundaries). '
        'Cosine-latitude weighted area mean over 13 initialisations.',
        ha='center', fontsize=7.8, color='#777777', fontstyle='italic'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2a — Precipitation skill horizon
# ══════════════════════════════════════════════════════════════════════════════

def fig2a():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={'wspace': 0.32})
    variable = 'TP'

    # ── ACC panel ─────────────────────────────────────────────────────────
    ax = axes[0]
    _draw_lines(ax, variable, 'pcc', refline=0.5)
    ax.set_ylabel('Anomaly Correlation Coefficient (ACC)', labelpad=6)
    ax.set_ylim(-0.15, 1.04)
    _add_noskill_zone(ax, -0.15)
    _add_panel_label(ax, '2a')
    ax.set_title('Precipitation — ACC', pad=10, fontsize=11)
    _add_legend(ax, ncol=2)

    # ── RMSE panel ────────────────────────────────────────────────────────
    ax = axes[1]
    _draw_lines(ax, variable, RMSE_COL[variable])
    ax.set_ylabel(f'RMSE  ({RUNIT[variable]})', labelpad=6)
    ax.set_title('Precipitation — RMSE', pad=10, fontsize=11)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    _add_panel_label(ax, '2b')

    _suptitle(fig, variable)
    _caption(fig)
    save(fig, 'fig02a_skill_tp')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2b — Z500 skill horizon (with FuXi collapse annotation)
# ══════════════════════════════════════════════════════════════════════════════

def fig2b():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={'wspace': 0.32})
    variable = 'Z500'

    # ── ACC panel ─────────────────────────────────────────────────────────
    ax = axes[0]
    _draw_lines(ax, variable, 'pcc', refline=0.5)
    ax.set_ylabel('Anomaly Correlation Coefficient (ACC)', labelpad=6)
    ax.set_ylim(-0.38, 1.04)
    _add_noskill_zone(ax, -0.38)
    _add_panel_label(ax, '2c')
    ax.set_title('Z500 — ACC', pad=10, fontsize=11)

    # FuXi collapse annotation — clean callout with rounded box
    ax.annotate(
        'FuXi-S2S reverts\nto climatology',
        xy=(3, 0.04), xytext=(1.6, -0.24),
        fontsize=7.8, color=COL['FuXi'], fontweight='semibold',
        arrowprops=dict(arrowstyle='-|>', color=COL['FuXi'], lw=1.1,
                        connectionstyle='arc3,rad=0.25',
                        shrinkA=2, shrinkB=4),
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.35', fc='#EAF2FA',
                  ec=COL['FuXi'], alpha=0.95, lw=0.7,
                  mutation_aspect=0.8))
    _add_legend(ax, ncol=2)

    # ── RMSE panel ────────────────────────────────────────────────────────
    ax = axes[1]
    _draw_lines(ax, variable, RMSE_COL[variable])
    ax.set_ylabel(f'RMSE  ({RUNIT[variable]})', labelpad=6)
    ax.set_title('Z500 — RMSE', pad=10, fontsize=11)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    _add_panel_label(ax, '2d')

    _suptitle(fig, variable)
    _caption(fig)
    save(fig, 'fig02b_skill_z500')


# def fig2c():
#     """Fig 2c — T2M skill horizon. Uncomment after running script 05."""
#     pass


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — SPIRE vs FuXi paired bootstrap difference
# ══════════════════════════════════════════════════════════════════════════════

def fig3():
    active_vars = [v for v in VARS if not df[df.variable == v].empty]
    nv = len(active_vars)
    fig, ax = plt.subplots(nv, 2, figsize=(13, 4.2 * nv),
                           gridspec_kw={'hspace': 0.50, 'wspace': 0.30})
    if nv == 1:
        ax = ax[np.newaxis, :]

    for r, v in enumerate(active_vars):
        for c, (mlabel, lower) in enumerate([('pcc', False), ('rmse', True)]):
            met = 'pcc' if mlabel == 'pcc' else RMSE_COL[v]
            a = ax[r, c]
            wks, mn, elo, ehi, sig = [], [], [], [], []
            for wk in range(1, 7):
                A = df[(df.variable == v) & (df.region == 'All India') &
                       (df.model == 'SPIRE') & (df.wk == wk)]
                B = df[(df.variable == v) & (df.region == 'All India') &
                       (df.model == 'FuXi') & (df.wk == wk)]
                md, lo, hi, p = paired_bootstrap_diff(
                    dict(zip(A.init_date, A[met])),
                    dict(zip(B.init_date, B[met])))
                if np.isfinite(md):
                    wks.append(wk); mn.append(md)
                    elo.append(md - lo); ehi.append(hi - md)
                    sig.append(p < 0.05)

            cols = [COL['SPIRE'] if ((md < 0) if lower else (md > 0))
                    else COL['FuXi'] for md in mn]
            a.bar(wks, mn, yerr=[elo, ehi], color=cols, capsize=4,
                  edgecolor='white', lw=0.6, alpha=0.85, zorder=3,
                  error_kw=dict(ecolor='#444444', lw=1.0, capthick=1.0),
                  width=0.65)
            mxabs = max(map(abs, mn)) if mn else 1
            for x, md, eh, s in zip(wks, mn, ehi, sig):
                if s:
                    a.text(x, md + np.sign(md) * (eh + 0.06 * mxabs), '★',
                           ha='center', fontsize=11, color='#222222')
            a.axhline(0, color='#444444', lw=0.8, zorder=4)
            _polish_ax(a)
            a.set_ylabel(r'$\Delta$ACC' if mlabel == 'pcc'
                         else f'$\\Delta$RMSE ({RUNIT[v]})')
            lbl = chr(97 + 2 * r + c)
            _add_panel_label(a, lbl)
            a.set_title(f'{VLAB[v]} \u2014 $\\Delta${mlabel.upper()}',
                        pad=10, fontsize=11)

    legend_els = [
        Line2D([0], [0], color=COL['SPIRE'], lw=0, marker='s',
               markersize=8, markerfacecolor=COL['SPIRE'], label='SPIRE better'),
        Line2D([0], [0], color=COL['FuXi'], lw=0, marker='o',
               markersize=8, markerfacecolor=COL['FuXi'], label='FuXi-S2S better'),
        Line2D([0], [0], color='#222222', lw=0, marker='$\u2605$',
               markersize=9, label='$p < 0.05$ (paired bootstrap)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.95, edgecolor='#CCCCCC',
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('SPIRE \u2212 FuXi-S2S Paired Skill Difference  |  JFM 2026',
                 fontsize=13, fontweight='bold', y=1.01, color='#111111')
    save(fig, 'fig03_spire_vs_fuxi')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Regional scorecard heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig4():
    active_vars = [v for v in VARS if not df[df.variable == v].empty]
    nv = len(active_vars)
    n_reg = len(REG)
    n_mdl = len(CORE)
    fig, axes = plt.subplots(1, nv, figsize=(5.2 * nv, 1.0 + 1.0 * n_reg),
                             gridspec_kw={'wspace': 0.35})
    if nv == 1:
        axes = [axes]
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad('#EEEEEE')
    im = None
    for k, (ax, v) in enumerate(zip(axes, active_vars)):
        M = np.array([[
            df[(df.variable == v) & (df.region == rg) &
               (df.model == m) & (df.wk <= 6)]['pcc'].mean()
            for m in CORE] for rg in REG])
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=0.85, aspect='auto')
        for i in range(n_reg):
            j = int(np.nanargmax(M[i]))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='#111111',
                                       lw=2.2, zorder=5))
        ax.set_xticks(range(n_mdl))
        ax.set_xticklabels([LAB[m] for m in CORE], rotation=30,
                           ha='right', fontsize=9)
        ax.set_yticks(range(n_reg))
        ax.set_yticklabels([REGL[r] for r in REG], fontsize=9.5)
        # Bold font for All India row
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == 'All India':
                lbl.set_fontweight('bold')
        for i in range(n_reg):
            for j in range(n_mdl):
                val = M[i, j]
                txt_col = 'white' if val < 0.35 or val > 0.72 else '#111111'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontweight='bold', fontsize=11, color=txt_col)
        ax.set_xticks(np.arange(-0.5, n_mdl, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_reg, 1), minor=True)
        ax.grid(which='minor', color='white', lw=2.0)
        ax.tick_params(which='minor', length=0)
        ax.tick_params(which='major', length=0)
        # Separator line between All India and sub-regions
        ax.axhline(0.5, color='#333333', lw=1.5, zorder=6)
        lbl = chr(97 + k)
        ax.set_title(f'({lbl})  {VLAB[v]}', fontsize=11, fontweight='bold', pad=8)

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.03, shrink=0.85)
    cb.set_label('Mean ACC  (Weeks 1\u20136)', fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_linewidth(0.5)
    fig.suptitle(
        'Regional Skill Scorecard  |  IMD Homogeneous Regions  |  JFM 2026',
        fontsize=12, fontweight='bold', y=1.02, color='#111111')
    fig.text(0.5, -0.02,
             'Mean ACC averaged over all 6 forecast weeks. '
             'Best-performing system per region outlined in black.',
             ha='center', fontsize=8, color='#777777', fontstyle='italic')
    save(fig, 'fig04_regional_scorecard')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Bias (forecast − observation) vs lead time — grouped bars
# ══════════════════════════════════════════════════════════════════════════════

def fig5_bias():
    """Bias (forecast − obs) vs forecast lead — clean line plot, 4 core models."""
    active_vars = [v for v in VARS if not df[df.variable == v].empty]
    nv = len(active_vars)
    fig, axes = plt.subplots(1, nv, figsize=(6.5 * nv, 4.8),
                             gridspec_kw={'wspace': 0.32})
    if nv == 1:
        axes = [axes]

    bias_units = {'TP': 'mm day$^{-1}$', 'Z500': 'gpm', 'T2M': 'K'}
    bias_dir   = {'TP': ('wet bias ↑', 'dry bias ↓'),
                  'Z500': ('high bias ↑', 'low bias ↓'),
                  'T2M': ('warm bias ↑', 'cold bias ↓')}
    panel_labels = [chr(97 + i) for i in range(nv)]
    BIAS_MDL = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']

    for k, (ax, v) in enumerate(zip(axes, active_vars)):
        for m in BIAS_MDL:
            sub = df[(df.variable == v) & (df.region == 'All India') & (df.model == m)]
            if sub.empty:
                continue
            xs, ys = [], []
            for wk in range(1, 7):
                vals = sub[sub.wk == wk]['bias'].values
                if len(vals) == 0:
                    continue
                mean = np.nanmean(vals)
                if np.isfinite(mean):
                    xs.append(wk); ys.append(mean)
            if not xs:
                continue
            ls, mk, lw, ms, mfc_mode = STY[m]
            mfc = COL[m] if mfc_mode == 'fill' else 'white'
            ax.plot(xs, ys, ls, marker=mk, color=COL[m], lw=lw, ms=ms,
                    label=LAB[m], zorder=5,
                    markerfacecolor=mfc, markeredgecolor=COL[m],
                    markeredgewidth=1.3 if mfc_mode == 'open' else 0.8)

        # Zero-bias reference — thick black line
        ax.axhline(0, color='#222222', lw=1.2, ls='-', zorder=2)

        # Symmetric y-axis with warm/cold shading
        yabs = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.25
        ax.set_ylim(-yabs, yabs)
        ax.axhspan(0, yabs, color='#FFF0E8', alpha=0.4, zorder=0, lw=0)
        ax.axhspan(-yabs, 0, color='#E8F0FF', alpha=0.4, zorder=0, lw=0)

        # Direction annotations — top-right and bottom-right
        pos_lbl, neg_lbl = bias_dir[v]
        ax.text(6.38, yabs * 0.88, pos_lbl, fontsize=8.5,
                color='#BB4400', fontstyle='italic', fontweight='semibold',
                ha='right', va='top')
        ax.text(6.38, -yabs * 0.88, neg_lbl, fontsize=8.5,
                color='#0044BB', fontstyle='italic', fontweight='semibold',
                ha='right', va='bottom')

        _polish_ax(ax)
        ax.set_ylabel(f'Mean Bias  ({bias_units[v]})', labelpad=6)
        ax.set_title(f'{VLAB[v]} — Forecast Bias', pad=10, fontsize=11)
        _add_panel_label(ax, panel_labels[k])
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        if k == 0:
            _add_legend(ax, ncol=2, loc='upper left')

    fig.suptitle(
        'JFM 2026 S2S Mean Forecast Bias  (forecast $-$ observation)',
        fontsize=13, fontweight='bold', y=1.01, color='#111111'
    )
    fig.text(
        0.5, -0.025,
        'All-India (land only, IMD boundaries). '
        'Cosine-latitude weighted area mean over 13 initialisations. '
        'Positive = forecast exceeds observation (overestimate).',
        ha='center', fontsize=7.8, color='#777777', fontstyle='italic'
    )
    save(fig, 'fig05_bias')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Bias scorecard heatmap (regions × models)
# ══════════════════════════════════════════════════════════════════════════════

def fig6_bias_scorecard():
    """Heatmap of mean bias per region × model, diverging colormap."""
    active_vars = [v for v in VARS if not df[df.variable == v].empty]
    nv = len(active_vars)
    n_reg = len(REG)
    n_mdl = len(CORE)
    fig, axes = plt.subplots(1, nv, figsize=(5.2 * nv, 1.0 + 1.0 * n_reg),
                             gridspec_kw={'wspace': 0.40})
    if nv == 1:
        axes = [axes]

    bias_units = {'TP': 'mm day⁻¹', 'Z500': 'gpm', 'T2M': 'K'}

    for k, (ax, v) in enumerate(zip(axes, active_vars)):
        # Build bias matrix: rows = regions, cols = models
        M = np.array([[
            df[(df.variable == v) & (df.region == rg) &
               (df.model == m)]['bias'].mean()
            for m in CORE] for rg in REG])

        # Symmetric color limits centred on zero
        vmax = np.nanmax(np.abs(M)) * 1.05
        cmap = plt.get_cmap('RdBu_r')   # red = positive (wet/warm), blue = negative (dry/cold)
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        # Outline the model closest to zero bias per region
        for i in range(n_reg):
            j = int(np.nanargmin(np.abs(M[i])))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='#111111',
                                       lw=2.2, zorder=5))

        ax.set_xticks(range(n_mdl))
        ax.set_xticklabels([LAB[m] for m in CORE], rotation=30,
                           ha='right', fontsize=9)
        ax.set_yticks(range(n_reg))
        ax.set_yticklabels([REGL[r] for r in REG], fontsize=9.5)
        # Bold font for All India row
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == 'All India':
                lbl.set_fontweight('bold')

        # Cell text
        for i in range(n_reg):
            for j in range(n_mdl):
                val = M[i, j]
                # Pick text colour for readability
                intensity = abs(val) / vmax if vmax > 0 else 0
                txt_col = 'white' if intensity > 0.55 else '#111111'
                fmt = f'{val:+.2f}' if v == 'TP' else f'{val:+.1f}'
                ax.text(j, i, fmt, ha='center', va='center',
                        fontweight='bold', fontsize=10.5, color=txt_col)

        ax.set_xticks(np.arange(-0.5, n_mdl, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_reg, 1), minor=True)
        ax.grid(which='minor', color='white', lw=2.0)
        ax.tick_params(which='minor', length=0)
        ax.tick_params(which='major', length=0)
        # Separator line between All India and sub-regions
        ax.axhline(0.5, color='#333333', lw=1.5, zorder=6)
        lbl = chr(97 + k)
        ax.set_title(f'({lbl})  {VLAB[v]}  ({bias_units[v]})',
                     fontsize=11, fontweight='bold', pad=8)

        # Per-panel colourbar
        cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.03, shrink=0.85)
        cb.set_label(f'Mean Bias ({bias_units[v]})', fontsize=9)
        cb.ax.tick_params(labelsize=8.5)
        cb.outline.set_linewidth(0.5)

    fig.suptitle(
        'Regional Bias Scorecard  |  IMD Homogeneous Regions  |  JFM 2026',
        fontsize=12, fontweight='bold', y=1.02, color='#111111')
    fig.text(0.5, -0.02,
             'Mean bias (forecast − obs) averaged over all 6 weeks × '
             '13 initialisations. '
             'Least-biased system per region outlined in black. '
             'Red = overestimate, Blue = underestimate.',
             ha='center', fontsize=8, color='#777777', fontstyle='italic')
    save(fig, 'fig06_bias_scorecard')


if __name__ == '__main__':
    print('Building publication figures (600 DPI) ...', flush=True)
    fig2a(); fig2b()
    # fig2c()   # uncomment after running script 05
    fig3(); fig4(); fig5_bias(); fig6_bias_scorecard()
    print('V3_FIGURES_DONE', flush=True)
