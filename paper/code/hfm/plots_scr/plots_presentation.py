#!/usr/bin/env python3
"""
plots_presentation.py — Comprehensive S2S verification figures for presentation.

Reads:  skill_deterministic.csv, skill_probabilistic.csv, skill_brier.csv
        from BASE_DIR (default: parent of this script → presentationv2/)
        Falls back to old_1/ if CSVs not yet in BASE_DIR.

Writes: plots_results_pres/
  fig01_apcc_allIndia.png         — PCC skill horizon (All India, TP + Z500)
  fig02_apcc_regional_TP.png      — TP PCC all 5 regions
  fig03_apcc_regional_Z500.png    — Z500 PCC all 5 regions
  fig04_bias_rmse_ssr.png         — Bias / RMSE / SSR (All India, TP + Z500)
  fig05_msss_allIndia.png         — MSSS vs clim + pers (All India)
  fig06_crpss_allIndia.png        — CRPSS vs clim + pers (All India)
  fig07_bss_events.png            — BSS for above/below-normal events
  fig08_scatter_r2_mae.png        — Scatter fcst vs obs with R², MAE, regression
  fig09_scorecard.png             — Heatmap scorecard (model × region × metric)
  fig10_daily_pcc.png             — Daily PCC horizon
  fig11_msss_regional.png         — MSSS all 5 regions
  fig12_crpss_regional.png        — CRPSS all 5 regions
  fig13_spread_skill.png          — Spread-skill ratio (SSR) all regions

Usage:
  python plots_presentation.py                    # use default data dir
  python plots_presentation.py --data-dir /path   # explicit data dir
  python plots_presentation.py --figs 1 3 8       # only specific figures
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.stats import pearsonr

# ── paths ─────────────────────────────────────────────────────────────────────
HERE  = os.path.dirname(os.path.abspath(__file__))
ODIR  = os.path.join(HERE, 'plots_results_pres')
os.makedirs(ODIR, exist_ok=True)


def _find_data_dir(cli_dir=None):
    candidates = []
    if cli_dir:
        candidates.append(cli_dir)
    parent = os.path.dirname(HERE)   # presentationv2/
    candidates += [parent, os.path.join(parent, 'old_1'), os.path.join(parent, 'old_2')]
    for d in candidates:
        if os.path.isfile(os.path.join(d, 'skill_deterministic.csv')):
            print(f"[data] reading from {d}")
            return d
    raise FileNotFoundError("skill_deterministic.csv not found in any candidate dir")


# ── display config ────────────────────────────────────────────────────────────
REGION_LABEL = {
    'All India':            'All India',
    'northwest_india':      'NW India',
    'central_india':        'Central India',
    'south_peninsula':      'South Peninsula',
    'east_northeast_india': 'East & NE India',
}
REGIONS = ['All India', 'northwest_india', 'central_india',
           'south_peninsula', 'east_northeast_india']

MODEL_STYLE = {
    'FuXi':        dict(color='#ff7f0e', marker='s', lw=2.0, ms=6, label='FuXi'),
    'ECMWF':       dict(color='#2ca02c', marker='^', lw=2.0, ms=6, label='ECMWF'),
    'MME':         dict(color='#9467bd', marker='D', lw=2.0, ms=6, label='MME'),
    'Persistence': dict(color='#7f7f7f', marker='x', lw=1.5, ms=6, ls='--', label='Persistence'),
}
DET_MODELS  = ['FuXi', 'ECMWF', 'MME', 'Persistence']
PROB_MODELS = ['FuXi', 'ECMWF', 'MME']
WEEKS       = [1, 2, 3, 4, 5, 6]
WEEK_LABELS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VAR_UNIT    = {'TP': 'mm/day', 'Z500': 'gpm'}

# ── helpers ───────────────────────────────────────────────────────────────────
def _agg(df, var, model, region, scale, metric, weeks=WEEKS):
    """Mean over init_dates for each lead/week."""
    sub = df[(df.variable == var) & (df.model == model) &
             (df.region == region) & (df.scale == scale)]
    return np.array([sub[sub.lead == w][metric].mean() for w in weeks])


def _style_ax(ax, ylabel='', ylim=None, title='', xlabel='Forecast Week'):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(WEEKS)
    ax.set_xticklabels(WEEK_LABELS)
    ax.tick_params(labelsize=8)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--')


def _plot_models(ax, df, var, region, scale, metric, models=DET_MODELS, **extra):
    for m in models:
        if m not in MODEL_STYLE:
            continue
        vals = _agg(df, var, m, region, scale, metric)
        st = {**MODEL_STYLE[m], **extra}
        ax.plot(WEEKS, vals, **st)


def _savefig(fig, name):
    path = os.path.join(ODIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — APCC skill horizon, All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig01_apcc_allIndia(det, prob):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle('Pattern Correlation (APCC) — All India, Weekly Aggregated', fontsize=12, y=1.01)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        _plot_models(ax, det, var, 'All India', 'weekly', 'pcc')
        _style_ax(ax, ylabel='PCC', ylim=(-0.2, 1.05),
                  title=f'{var}  ({VAR_UNIT[var]})')
        ax.axhline(0.5, color='orange', lw=0.7, ls=':', alpha=0.8)
        ax.text(6.05, 0.5, '0.5', color='orange', fontsize=7, va='center')

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig01_apcc_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — APCC regional TP (5 regions × 1 panel each)
# ══════════════════════════════════════════════════════════════════════════════
def fig02_apcc_regional_TP(det):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    fig.suptitle('TP Pattern Correlation (APCC) by IMD Region', fontsize=12, y=1.01)
    for ai, reg in enumerate(REGIONS):
        ax = axes[ai]
        _plot_models(ax, det, 'TP', reg, 'weekly', 'pcc')
        _style_ax(ax, ylabel='PCC' if ai == 0 else '',
                  ylim=(-0.4, 1.05), title=REGION_LABEL[reg])
        ax.axhline(0.5, color='orange', lw=0.7, ls=':', alpha=0.7)
    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.1), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig02_apcc_regional_TP.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — APCC regional Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig03_apcc_regional_Z500(det):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    fig.suptitle('Z500 Pattern Correlation (APCC) by IMD Region', fontsize=12, y=1.01)
    for ai, reg in enumerate(REGIONS):
        ax = axes[ai]
        _plot_models(ax, det, 'Z500', reg, 'weekly', 'pcc')
        _style_ax(ax, ylabel='PCC' if ai == 0 else '',
                  ylim=(-0.4, 1.05), title=REGION_LABEL[reg])
        ax.axhline(0.5, color='orange', lw=0.7, ls=':', alpha=0.7)
    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.1), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig03_apcc_regional_Z500.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Bias / RMSE / SSR, All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig04_bias_rmse_ssr(det, prob):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey='row')
    fig.suptitle('Bias / RMSE / Spread-Skill Ratio — All India', fontsize=12, y=1.01)

    metrics   = ['bias', 'rmse', 'ssr']
    ylabels   = ['Bias', 'RMSE', 'SSR']
    ylims     = [None, None, (0, 2.5)]

    for ci, var in enumerate(['TP', 'Z500']):
        unit = VAR_UNIT[var]
        for ri, (met, ylab, ylim) in enumerate(zip(metrics, ylabels, ylims)):
            ax = axes[ri, ci]
            src = det if met != 'ssr' else prob
            mods = DET_MODELS if met != 'ssr' else PROB_MODELS
            _plot_models(ax, src, var, 'All India', 'weekly', met, models=mods)
            if met == 'ssr':
                ax.axhline(1.0, color='red', lw=1.0, ls='-', alpha=0.5, label='perfect')
            title = f'{var} ({unit})' if ri == 0 else ''
            _style_ax(ax, ylabel=f'{ylab}' + (f' ({unit})' if met != 'ssr' else ''),
                      ylim=ylim, title=title)

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    handles.append(plt.Line2D([0], [0], color='red', lw=1.0, ls='-', alpha=0.5, label='SSR=1 (perfect)'))
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig04_bias_rmse_ssr.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — MSSS (vs clim + pers), All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig05_msss_allIndia(det):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('MSSS — All India (model-own climatology baseline for FuXi/ECMWF)', fontsize=11, y=1.01)

    metrics = ['msss_clim', 'msss_pers']
    col_titles = ['MSSS vs Climatology', 'MSSS vs Persistence']
    for ri, var in enumerate(['TP', 'Z500']):
        for ci, (met, ctit) in enumerate(zip(metrics, col_titles)):
            ax = axes[ri, ci]
            _plot_models(ax, det, var, 'All India', 'weekly', met)
            _style_ax(ax, ylabel='MSSS', ylim=(-0.5, 1.05),
                      title=(f'{var} ({VAR_UNIT[var]}) — {ctit}' if ri == 0 else ctit))

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig05_msss_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — CRPSS, All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig06_crpss_allIndia(prob):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CRPSS vs Climatology — All India (Gaussian ensemble CRPS)', fontsize=12, y=1.01)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        _plot_models(ax, prob, var, 'All India', 'weekly', 'crpss_clim', models=PROB_MODELS)
        _style_ax(ax, ylabel='CRPSS', ylim=(-0.3, 1.05), title=f'{var} ({VAR_UNIT[var]})')

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in PROB_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig06_crpss_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — BSS for key events (above/below normal, TP gt thresholds)
# ══════════════════════════════════════════════════════════════════════════════
def fig07_bss_events(brier):
    events = [
        ('tp_above_normal',   'TP Above Normal'),
        ('tp_below_normal',   'TP Below Normal'),
        ('z500_above_normal', 'Z500 Above Normal'),
        ('z500_below_normal', 'Z500 Below Normal'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('Brier Skill Score (BSS vs Climatology) — All India', fontsize=12, y=1.01)

    bw = brier[brier.scale == 'weekly']
    for idx, (event, title) in enumerate(events):
        ax = axes[idx // 2, idx % 2]
        var = 'TP' if event.startswith('tp') else 'Z500'
        sub_e = bw[(bw.event == event) & (bw.region == 'All India') & (bw.variable == var)]
        for m in PROB_MODELS:
            if m not in MODEL_STYLE:
                continue
            sub_m = sub_e[sub_e.model == m]
            vals = np.array([sub_m[sub_m.lead == w]['briss_clim'].mean() for w in WEEKS])
            st = MODEL_STYLE[m]
            ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                    lw=st['lw'], ms=st['ms'], label=st['label'])
        _style_ax(ax, ylabel='BSS', ylim=(-0.5, 1.1), title=title)

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in PROB_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig07_bss_events.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Scatter fcst_mean vs obs_mean with R², MAE, regression line
# ══════════════════════════════════════════════════════════════════════════════
def fig08_scatter_r2_mae(det):
    """Scatter of weekly fcst_mean vs obs_mean; each point = one (init, week).
    Shows R², MAE, and regression line per model."""
    dw = det[(det.scale == 'weekly') & (det.region == 'All India')]
    scats_models = ['FuXi', 'ECMWF', 'MME']
    vars_ = ['TP', 'Z500']

    fig, axes = plt.subplots(len(vars_), len(scats_models),
                             figsize=(11, 6.5), sharey='row', sharex='row')
    fig.suptitle('Forecast vs Observed Mean (Weekly, All India) — R² and MAE', fontsize=12, y=1.01)

    for ri, var in enumerate(vars_):
        unit = VAR_UNIT[var]
        sub_v = dw[dw.variable == var]
        all_obs = sub_v['obs_mean'].dropna().values
        vmin, vmax = np.nanmin(all_obs), np.nanmax(all_obs)
        pad = (vmax - vmin) * 0.1

        for ci, m in enumerate(scats_models):
            ax = axes[ri, ci]
            sub_m = sub_v[sub_v.model == m].dropna(subset=['fcst_mean', 'obs_mean'])
            x = sub_m['obs_mean'].values   # observed
            y = sub_m['fcst_mean'].values  # forecast

            # compute stats
            if len(x) > 2:
                r, p = pearsonr(x, y)
                r2 = r ** 2
                mae = np.mean(np.abs(y - x))
                slope, intercept, *_ = stats.linregress(x, y)
                xlim = (vmin - pad, vmax + pad)
                xline = np.array(xlim)
                yline = slope * xline + intercept
                ax.plot(xline, yline, 'r-', lw=1.2, alpha=0.7, label='Regression')
            else:
                r2, mae = np.nan, np.nan

            col = MODEL_STYLE[m]['color']
            # colour by week
            sc = ax.scatter(x, y, c=sub_m['lead'].values, cmap='viridis',
                            s=35, alpha=0.8, zorder=3, vmin=1, vmax=6,
                            edgecolors=col, linewidths=0.5)
            diag = np.array([vmin - pad, vmax + pad])
            ax.plot(diag, diag, 'k--', lw=0.8, alpha=0.4, label='1:1')
            ax.set_xlim(vmin - pad, vmax + pad)
            ax.set_ylim(vmin - pad, vmax + pad)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{m}', fontsize=10, pad=3, color=col)
            ax.set_xlabel(f'Obs ({unit})', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{var} Fcst ({unit})', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(lw=0.3, alpha=0.4)
            if not np.isnan(r2):
                ax.text(0.04, 0.96, f'R²={r2:.2f}\nMAE={mae:.3f}',
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(fc='white', ec='none', alpha=0.8))

        # one colorbar per row
        cbar = fig.colorbar(sc, ax=axes[ri, -1], fraction=0.046, pad=0.04)
        cbar.set_label('Forecast Week', fontsize=8)
        cbar.set_ticks([1, 2, 3, 4, 5, 6])

    fig.tight_layout()
    _savefig(fig, 'fig08_scatter_r2_mae.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Scorecard heatmap: model × week, metric = PCC (All India)
# ══════════════════════════════════════════════════════════════════════════════
def fig09_scorecard(det, prob):
    """Heatmap: rows=model, cols=week, separate panels per (var, metric)."""
    metrics = [
        ('det',  'pcc',        'PCC',     (-0.2, 1.0),  'RdYlGn'),
        ('det',  'bias',       'Bias',    None,          'RdBu_r'),
        ('det',  'msss_clim',  'MSSS',    (-0.5, 1.0),  'RdYlGn'),
        ('det',  'rmse',       'RMSE',    None,          'YlOrRd_r'),
        ('prob', 'crpss_clim', 'CRPSS',   (-0.3, 1.0),  'RdYlGn'),
    ]
    vars_ = ['TP', 'Z500']
    mods  = ['FuXi', 'ECMWF', 'MME']

    nrows = len(vars_) * len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, nrows * 1.5 + 1))
    if nrows == 1:
        axes = [axes]
    fig.suptitle(f'Scorecard — All India (mean over {len(det.init_date.unique())} inits)', fontsize=12, y=1.01)

    row = 0
    for var in vars_:
        for src_key, met, mlab, vlim, cmap in metrics:
            ax = axes[row]
            src = det if src_key == 'det' else prob
            data = np.array([[_agg(src, var, m, 'All India', 'weekly', met)[w - 1]
                               for w in WEEKS] for m in mods])
            vabs = np.nanmax(np.abs(data)) if vlim is None else None
            kw = {}
            if vlim:
                kw = dict(vmin=vlim[0], vmax=vlim[1])
            elif met == 'bias':
                kw = dict(norm=TwoSlopeNorm(vcenter=0))
            else:
                kw = dict(vmin=-vabs, vmax=vabs)

            im = ax.imshow(data, aspect='auto', cmap=cmap, **kw)
            ax.set_yticks(range(len(mods)))
            ax.set_yticklabels(mods, fontsize=8)
            ax.set_xticks(range(len(WEEKS)))
            ax.set_xticklabels(WEEK_LABELS, fontsize=8)
            ax.set_title(f'{var} — {mlab}  ({VAR_UNIT[var]})', fontsize=9, pad=2)
            for i in range(len(mods)):
                for j in range(len(WEEKS)):
                    v = data[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
            row += 1

    fig.tight_layout()
    _savefig(fig, 'fig09_scorecard.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Daily PCC horizon (days 1-42, all models)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_daily_pcc(det):
    DAYS = list(range(1, 43))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'Daily Pattern Correlation — All India (mean over {len(det.init_date.unique())} inits)', fontsize=12, y=1.01)

    week_bounds = [0.5, 7.5, 14.5, 21.5, 28.5, 35.5, 42.5]
    week_colors = ['#f0f0f0', '#e0e0e0']

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for i in range(6):
            ax.axvspan(week_bounds[i], week_bounds[i + 1],
                       alpha=0.3, color=week_colors[i % 2], zorder=0)
            ax.text((week_bounds[i] + week_bounds[i + 1]) / 2, 1.02,
                    f'W{i+1}', ha='center', fontsize=7, transform=ax.get_xaxis_transform())

        for m in DET_MODELS:
            sub = det[(det.variable == var) & (det.model == m) &
                      (det.region == 'All India') & (det.scale == 'daily')]
            vals = np.array([sub[sub.lead == d]['pcc'].mean() for d in DAYS])
            st = MODEL_STYLE[m]
            ax.plot(DAYS, vals, color=st['color'], lw=st.get('lw', 1.5),
                    ls=st.get('ls', '-'), alpha=0.9, label=st['label'])

        ax.axhline(0.5, color='orange', lw=0.8, ls=':', alpha=0.8)
        ax.axhline(0,   color='#aaaaaa', lw=0.8, ls='--')
        ax.set_xlim(1, 42)
        ax.set_ylim(-0.3, 1.05)
        ax.set_xlabel('Forecast Lead Day', fontsize=9)
        ax.set_ylabel('PCC', fontsize=9)
        ax.set_xticks([1, 7, 14, 21, 28, 35, 42])
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', lw=0.4, alpha=0.5)
        ax.set_title(f'{var} ({VAR_UNIT[var]})', fontsize=10)

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','lw','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig10_daily_pcc.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — MSSS all 5 regions (TP + Z500)
# ══════════════════════════════════════════════════════════════════════════════
def fig11_msss_regional(det):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    fig.suptitle('MSSS vs Climatology — All IMD Regions', fontsize=12, y=1.01)

    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            _plot_models(ax, det, var, reg, 'weekly', 'msss_clim')
            _style_ax(ax, ylabel='MSSS' if ci == 0 else '',
                      ylim=(-0.6, 1.1),
                      title=REGION_LABEL[reg] if ri == 0 else '')
            if ci == 0:
                ax.text(-0.35, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10, fontweight='bold')

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in DET_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig11_msss_regional.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — CRPSS all 5 regions (TP + Z500)
# ══════════════════════════════════════════════════════════════════════════════
def fig12_crpss_regional(prob):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    fig.suptitle('CRPSS vs Climatology — All IMD Regions', fontsize=12, y=1.01)

    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            _plot_models(ax, prob, var, reg, 'weekly', 'crpss_clim', models=PROB_MODELS)
            _style_ax(ax, ylabel='CRPSS' if ci == 0 else '',
                      ylim=(-0.4, 1.1),
                      title=REGION_LABEL[reg] if ri == 0 else '')
            if ci == 0:
                ax.text(-0.35, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10, fontweight='bold')

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in PROB_MODELS if m in MODEL_STYLE]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig12_crpss_regional.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 — Spread-Skill Ratio all regions
# ══════════════════════════════════════════════════════════════════════════════
def fig13_spread_skill(prob):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    fig.suptitle('Spread-Skill Ratio (SSR=1 → perfectly calibrated) — All IMD Regions', fontsize=12, y=1.01)

    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            _plot_models(ax, prob, var, reg, 'weekly', 'ssr', models=PROB_MODELS)
            ax.axhline(1.0, color='red', lw=1.0, ls='-', alpha=0.5, label='perfect')
            _style_ax(ax, ylabel='SSR' if ci == 0 else '',
                      ylim=(0, 3.0),
                      title=REGION_LABEL[reg] if ri == 0 else '')
            if ci == 0:
                ax.text(-0.35, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10, fontweight='bold')

    handles = [plt.Line2D([0], [0], **{k: v for k, v in MODEL_STYLE[m].items()
                                        if k in ('color','marker','lw','ms','ls','label')})
               for m in PROB_MODELS if m in MODEL_STYLE]
    handles.append(plt.Line2D([0], [0], color='red', lw=1.0, ls='-', alpha=0.5, label='SSR=1 (perfect)'))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig13_spread_skill.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
ALL_FIGS = {
    1:  ('fig01_apcc_allIndia',      lambda d, p, b: fig01_apcc_allIndia(d, p)),
    2:  ('fig02_apcc_regional_TP',   lambda d, p, b: fig02_apcc_regional_TP(d)),
    3:  ('fig03_apcc_regional_Z500', lambda d, p, b: fig03_apcc_regional_Z500(d)),
    4:  ('fig04_bias_rmse_ssr',      lambda d, p, b: fig04_bias_rmse_ssr(d, p)),
    5:  ('fig05_msss_allIndia',      lambda d, p, b: fig05_msss_allIndia(d)),
    6:  ('fig06_crpss_allIndia',     lambda d, p, b: fig06_crpss_allIndia(p)),
    7:  ('fig07_bss_events',         lambda d, p, b: fig07_bss_events(b)),
    8:  ('fig08_scatter_r2_mae',     lambda d, p, b: fig08_scatter_r2_mae(d)),
    9:  ('fig09_scorecard',          lambda d, p, b: fig09_scorecard(d, p)),
    10: ('fig10_daily_pcc',          lambda d, p, b: fig10_daily_pcc(d)),
    11: ('fig11_msss_regional',      lambda d, p, b: fig11_msss_regional(d)),
    12: ('fig12_crpss_regional',     lambda d, p, b: fig12_crpss_regional(p)),
    13: ('fig13_spread_skill',       lambda d, p, b: fig13_spread_skill(p)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None,
                        help='Directory containing skill_*.csv files')
    parser.add_argument('--figs', nargs='+', type=int, default=None,
                        help='Which figures to generate (e.g. --figs 1 3 8)')
    args = parser.parse_args()

    data_dir = _find_data_dir(args.data_dir)
    det   = pd.read_csv(os.path.join(data_dir, 'skill_deterministic.csv'))
    prob  = pd.read_csv(os.path.join(data_dir, 'skill_probabilistic.csv'))
    brier = pd.read_csv(os.path.join(data_dir, 'skill_brier.csv'))

    print(f"  det  rows: {len(det)}")
    print(f"  prob rows: {len(prob)}")
    print(f"  brier rows: {len(brier)}")
    print(f"  models: {sorted(det.model.unique())}")
    print(f"  regions: {sorted(det.region.unique())}")

    figs_to_run = sorted(args.figs) if args.figs else sorted(ALL_FIGS.keys())
    print(f"\nGenerating {len(figs_to_run)} figures → {ODIR}\n")

    for n in figs_to_run:
        name, fn = ALL_FIGS[n]
        print(f"[fig{n:02d}] {name}")
        try:
            fn(det, prob, brier)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. All figures in:\n  {ODIR}")


if __name__ == '__main__':
    main()
