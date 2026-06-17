#!/usr/bin/env python3
"""
plots_v3.py  —  Comprehensive S2S figures for V3 (dual clim-basis results).

Key difference from V2 plots: skill_deterministic.csv now has a `clim_basis`
column ('era5' or 'model_own').  This script can show:
  - ERA5-baseline only (fair inter-model comparison)
  - Model-own-clim only (bias-corrected anomaly skill)
  - BOTH on the same axis (solid=era5, dashed=model_own)

Usage:
  python plots_v3.py                         # all figures, both bases
  python plots_v3.py --basis era5            # era5 only
  python plots_v3.py --basis model_own       # model-own only
  python plots_v3.py --figs 1 3 8            # specific figures
  python plots_v3.py --data-dir /path/to/v3  # explicit data dir

Figures saved to  plots_results_pres/
  fig01_apcc_allIndia.png         — PCC horizon, All India, TP + Z500 (both bases)
  fig02_apcc_regional_TP.png      — TP PCC all 5 regions (both bases)
  fig03_apcc_regional_Z500.png    — Z500 PCC all 5 regions
  fig04_bias_rmse.png             — Bias + RMSE, All India (era5 basis)
  fig05_msss_allIndia.png         — MSSS All India (both bases)
  fig06_crpss_allIndia.png        — CRPSS All India (probabilistic, era5 only)
  fig07_bss_events.png            — BSS above/below normal
  fig08_scatter_r2_mae.png        — Scatter fcst vs obs, R², MAE
  fig09_scorecard_era5.png        — Scorecard heatmap, era5 basis
  fig10_scorecard_own.png         — Scorecard heatmap, model_own basis
  fig11_daily_pcc.png             — Daily PCC horizon
  fig12_msss_regional.png         — MSSS all regions (both bases)
  fig13_crpss_regional.png        — CRPSS all regions
  fig14_basis_comparison.png      — Side-by-side era5 vs model_own PCC
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ODIR = os.path.join(HERE, 'plots_results_pres')
os.makedirs(ODIR, exist_ok=True)


def _find_data_dir(cli_dir=None):
    candidates = []
    if cli_dir:
        candidates.append(cli_dir)
    parent = os.path.dirname(HERE)
    candidates.append(parent)
    for d in candidates:
        if os.path.isfile(os.path.join(d, 'skill_deterministic.csv')):
            print(f"[data] reading from {d}")
            return d
    raise FileNotFoundError("skill_deterministic.csv not found")


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

# Solid line = era5 basis, dashed = model_own basis
MODEL_COLOR = {
    'SPIRE':       '#1f77b4',
    'FuXi':        '#ff7f0e',
    'ECMWF':       '#2ca02c',
    'MME':         '#9467bd',
    'Persistence': '#7f7f7f',
}
MODEL_MARKER = {
    'SPIRE': 'o', 'FuXi': 's', 'ECMWF': '^', 'MME': 'D', 'Persistence': 'x',
}
BASIS_LS = {'era5': '-', 'model_own': '--'}
BASIS_LABEL = {'era5': 'ERA5 clim', 'model_own': 'Model-own clim'}

DET_MODELS  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
PROB_MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
WEEKS       = [1, 2, 3, 4, 5, 6]
WEEK_LABELS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VAR_UNIT    = {'TP': 'mm/day', 'Z500': 'gpm'}


def _agg(df, var, model, region, scale, metric, basis=None, weeks=WEEKS):
    sub = df[(df.variable == var) & (df.model == model) &
             (df.region == region) & (df.scale == scale)]
    if basis is not None and 'clim_basis' in sub.columns:
        sub = sub[sub.clim_basis == basis]
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


def _plot_models_both_bases(ax, df, var, region, metric, models=DET_MODELS,
                             bases=('era5', 'model_own')):
    """Plot solid (era5) and dashed (model_own) lines for each model."""
    for basis in bases:
        if 'clim_basis' in df.columns and basis not in df.clim_basis.values:
            continue
        for m in models:
            if m not in MODEL_COLOR:
                continue
            vals = _agg(df, var, m, region, 'weekly', metric, basis=basis)
            if np.all(np.isnan(vals)):
                continue
            label = f'{m} ({BASIS_LABEL[basis]})' if len(bases) > 1 else m
            ax.plot(WEEKS, vals,
                    color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    lw=2.0, ms=5, ls=BASIS_LS[basis], label=label)


def _savefig(fig, name):
    path = os.path.join(ODIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {path}")


def _legend_both_bases(fig, models, bases):
    handles = []
    for basis in bases:
        for m in models:
            if m not in MODEL_COLOR:
                continue
            h = plt.Line2D([0], [0], color=MODEL_COLOR[m],
                           marker=MODEL_MARKER[m], lw=2.0, ms=5,
                           ls=BASIS_LS[basis],
                           label=f'{m} ({BASIS_LABEL[basis]})')
            handles.append(h)
    ncol = min(len(handles), 5)
    fig.legend(handles=handles, loc='lower center', ncol=ncol, fontsize=8,
               bbox_to_anchor=(0.5, -0.06), frameon=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — APCC All India, both bases on same axis
# ══════════════════════════════════════════════════════════════════════════════
def fig01_apcc_allIndia(det, prob, bases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Pattern Correlation (APCC) — All India\n'
                 'Solid = ERA5 clim baseline  |  Dashed = Model-own clim baseline',
                 fontsize=11, y=1.02)
    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        _plot_models_both_bases(ax, det, var, 'All India', 'pcc',
                                models=DET_MODELS, bases=bases)
        _style_ax(ax, ylabel='PCC', ylim=(-0.2, 1.05),
                  title=f'{var} ({VAR_UNIT[var]})')
        ax.axhline(0.5, color='orange', lw=0.8, ls=':', alpha=0.8)
    _legend_both_bases(fig, DET_MODELS, bases)
    fig.tight_layout()
    _savefig(fig, 'fig01_apcc_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Regional APCC TP
# ══════════════════════════════════════════════════════════════════════════════
def fig02_apcc_regional_TP(det, bases):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
    fig.suptitle('TP APCC by IMD Region  (solid=ERA5 | dashed=model-own)',
                 fontsize=11, y=1.02)
    for ai, reg in enumerate(REGIONS):
        ax = axes[ai]
        _plot_models_both_bases(ax, det, 'TP', reg, 'pcc',
                                models=DET_MODELS, bases=bases)
        _style_ax(ax, ylabel='PCC' if ai == 0 else '',
                  ylim=(-0.4, 1.05), title=REGION_LABEL[reg])
        ax.axhline(0.5, color='orange', lw=0.7, ls=':', alpha=0.7)
    _legend_both_bases(fig, DET_MODELS, bases)
    fig.tight_layout()
    _savefig(fig, 'fig02_apcc_regional_TP.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Regional APCC Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig03_apcc_regional_Z500(det, bases):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
    fig.suptitle('Z500 APCC by IMD Region  (solid=ERA5 | dashed=model-own)',
                 fontsize=11, y=1.02)
    for ai, reg in enumerate(REGIONS):
        ax = axes[ai]
        _plot_models_both_bases(ax, det, 'Z500', reg, 'pcc',
                                models=DET_MODELS, bases=bases)
        _style_ax(ax, ylabel='PCC' if ai == 0 else '',
                  ylim=(-0.4, 1.05), title=REGION_LABEL[reg])
        ax.axhline(0.5, color='orange', lw=0.7, ls=':', alpha=0.7)
    _legend_both_bases(fig, DET_MODELS, bases)
    fig.tight_layout()
    _savefig(fig, 'fig03_apcc_regional_Z500.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Bias + RMSE (era5 basis, baseline-independent metrics)
# ══════════════════════════════════════════════════════════════════════════════
def fig04_bias_rmse(det, prob):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('Bias and RMSE — All India (ERA5 baseline; clim-independent)',
                 fontsize=11, y=1.01)
    for ci, var in enumerate(['TP', 'Z500']):
        unit = VAR_UNIT[var]
        for ri, (met, ylab) in enumerate([('bias', 'Bias'), ('rmse', 'RMSE')]):
            ax = axes[ri, ci]
            for m in DET_MODELS:
                vals = _agg(det, var, m, 'All India', 'weekly', met, basis='era5')
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(WEEKS, vals, color=MODEL_COLOR[m],
                        marker=MODEL_MARKER[m], lw=2.0, ms=5, label=m)
            _style_ax(ax, ylabel=f'{ylab} ({unit})',
                      title=f'{var} ({unit})' if ri == 0 else '')
    handles = [plt.Line2D([0],[0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                          lw=2.0, ms=5, label=m) for m in DET_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig04_bias_rmse.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — MSSS All India (both bases, shows how model-own clim changes picture)
# ══════════════════════════════════════════════════════════════════════════════
def fig05_msss_allIndia(det, bases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MSSS vs ERA5 Climatology — All India\n'
                 'Solid = ERA5 clim baseline  |  Dashed = Model-own clim baseline',
                 fontsize=11, y=1.02)
    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        _plot_models_both_bases(ax, det, var, 'All India', 'msss_clim',
                                models=DET_MODELS, bases=bases)
        _style_ax(ax, ylabel='MSSS', ylim=(-0.6, 1.05),
                  title=f'{var} ({VAR_UNIT[var]})')
    _legend_both_bases(fig, DET_MODELS, bases)
    fig.tight_layout()
    _savefig(fig, 'fig05_msss_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — CRPSS All India (probabilistic, era5 only)
# ══════════════════════════════════════════════════════════════════════════════
def fig06_crpss_allIndia(prob):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CRPSS vs Climatology — All India (Gaussian ensemble CRPS)',
                 fontsize=11, y=1.01)
    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in PROB_MODELS:
            vals = _agg(prob, var, m, 'All India', 'weekly', 'crpss_clim')
            if np.all(np.isnan(vals)):
                continue
            ax.plot(WEEKS, vals, color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    lw=2.0, ms=5, label=m)
        _style_ax(ax, ylabel='CRPSS', ylim=(-0.4, 1.05),
                  title=f'{var} ({VAR_UNIT[var]})')
    handles = [plt.Line2D([0],[0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                          lw=2.0, ms=5, label=m) for m in PROB_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig06_crpss_allIndia.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — BSS events
# ══════════════════════════════════════════════════════════════════════════════
def fig07_bss_events(brier):
    events = [('tp_above_normal','TP Above Normal'), ('tp_below_normal','TP Below Normal'),
              ('z500_above_normal','Z500 Above Normal'), ('z500_below_normal','Z500 Below Normal')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('Brier Skill Score — All India', fontsize=11, y=1.01)
    bw = brier[brier.scale == 'weekly']
    for idx, (event, title) in enumerate(events):
        ax = axes[idx // 2, idx % 2]
        var = 'TP' if event.startswith('tp') else 'Z500'
        sub_e = bw[(bw.event == event) & (bw.region == 'All India') & (bw.variable == var)]
        for m in PROB_MODELS:
            vals = np.array([sub_e[(sub_e.model==m)&(sub_e.lead==w)]['briss_clim'].mean()
                             for w in WEEKS])
            if np.all(np.isnan(vals)):
                continue
            ax.plot(WEEKS, vals, color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    lw=2.0, ms=5, label=m)
        _style_ax(ax, ylabel='BSS', ylim=(-0.6, 1.1), title=title)
    handles = [plt.Line2D([0],[0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                          lw=2.0, ms=5, label=m) for m in PROB_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig07_bss_events.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Scatter fcst vs obs (era5 basis, week-coloured)
# ══════════════════════════════════════════════════════════════════════════════
def fig08_scatter_r2_mae(det):
    dw = det[(det.scale=='weekly') & (det.region=='All India') &
             (det.clim_basis=='era5')]
    scat_models = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey='row', sharex='row')
    fig.suptitle('Forecast vs Observed Mean (Weekly, All India, ERA5 baseline)\nR² and MAE annotated',
                 fontsize=11, y=1.01)
    for ri, var in enumerate(['TP', 'Z500']):
        unit = VAR_UNIT[var]
        sub_v = dw[dw.variable == var]
        all_obs = sub_v['obs_mean'].dropna().values
        vmin, vmax = np.nanmin(all_obs), np.nanmax(all_obs)
        pad = (vmax - vmin) * 0.1
        for ci, m in enumerate(scat_models):
            ax = axes[ri, ci]
            sub_m = sub_v[sub_v.model == m].dropna(subset=['fcst_mean', 'obs_mean'])
            x, y  = sub_m['obs_mean'].values, sub_m['fcst_mean'].values
            if len(x) > 2:
                r, _ = pearsonr(x, y)
                r2   = r ** 2
                mae  = np.mean(np.abs(y - x))
                slope, intercept, *_ = stats.linregress(x, y)
                xlim = np.array([vmin - pad, vmax + pad])
                ax.plot(xlim, slope * xlim + intercept, 'r-', lw=1.2, alpha=0.7)
            else:
                r2, mae = np.nan, np.nan
            sc = ax.scatter(x, y, c=sub_m['lead'].values, cmap='viridis',
                            s=35, alpha=0.8, zorder=3, vmin=1, vmax=6,
                            edgecolors=MODEL_COLOR.get(m, '#333'), linewidths=0.4)
            diag = np.array([vmin - pad, vmax + pad])
            ax.plot(diag, diag, 'k--', lw=0.8, alpha=0.4)
            ax.set_xlim(vmin-pad, vmax+pad); ax.set_ylim(vmin-pad, vmax+pad)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(m, fontsize=10, color=MODEL_COLOR.get(m,'k'))
            ax.set_xlabel(f'Obs ({unit})', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{var} Fcst ({unit})', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(lw=0.3, alpha=0.4)
            if not np.isnan(r2):
                ax.text(0.04, 0.96, f'R²={r2:.2f}\nMAE={mae:.3f}',
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(fc='white', ec='none', alpha=0.8))
        fig.colorbar(sc, ax=axes[ri,-1], fraction=0.046, pad=0.04,
                     label='Forecast Week').set_ticks([1,2,3,4,5,6])
    fig.tight_layout()
    _savefig(fig, 'fig08_scatter_r2_mae.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9/10 — Scorecard heatmaps (one per basis)
# ══════════════════════════════════════════════════════════════════════════════
def _scorecard(det, prob, basis, fname, title_suffix):
    metrics = [
        ('det',  'pcc',        'PCC',    (-0.2, 1.0),  'RdYlGn'),
        ('det',  'bias',       'Bias',    None,          'RdBu_r'),
        ('det',  'msss_clim',  'MSSS',   (-0.5, 1.0),  'RdYlGn'),
        ('det',  'rmse',       'RMSE',    None,          'YlOrRd_r'),
        ('prob', 'crpss_clim', 'CRPSS',  (-0.3, 1.0),  'RdYlGn'),
    ]
    vars_  = ['TP', 'Z500']
    mods   = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    nrows  = len(vars_) * len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, nrows * 1.5 + 1.5))
    if nrows == 1:
        axes = [axes]
    fig.suptitle(f'Scorecard — All India  ({title_suffix})', fontsize=11, y=1.01)
    row = 0
    for var in vars_:
        for src_key, met, mlab, vlim, cmap in metrics:
            ax = axes[row]
            src = det if src_key == 'det' else prob
            if src_key == 'det' and 'clim_basis' in src.columns:
                src = src[src.clim_basis == basis]
            data = np.array([[_agg(src, var, m, 'All India', 'weekly', met)[w-1]
                               for w in WEEKS] for m in mods])
            kw = (dict(vmin=vlim[0], vmax=vlim[1]) if vlim
                  else dict(norm=TwoSlopeNorm(vcenter=0)) if met == 'bias'
                  else dict(vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data))))
            im = ax.imshow(data, aspect='auto', cmap=cmap, **kw)
            ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods, fontsize=8)
            ax.set_xticks(range(6)); ax.set_xticklabels(WEEK_LABELS, fontsize=8)
            ax.set_title(f'{var} — {mlab}  ({VAR_UNIT[var]})', fontsize=9, pad=2)
            for i in range(len(mods)):
                for j in range(6):
                    v = data[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
            row += 1
    fig.tight_layout()
    _savefig(fig, fname)


def fig09_scorecard_era5(det, prob):
    _scorecard(det, prob, 'era5', 'fig09_scorecard_era5.png', 'ERA5 clim baseline')


def fig10_scorecard_own(det, prob):
    _scorecard(det, prob, 'model_own', 'fig10_scorecard_own.png', 'Model-own clim baseline')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Daily PCC
# ══════════════════════════════════════════════════════════════════════════════
def fig11_daily_pcc(det):
    DAYS = list(range(1, 43))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle('Daily PCC — All India (ERA5 baseline, mean over 13 inits)',
                 fontsize=11, y=1.01)
    week_bounds = [0.5, 7.5, 14.5, 21.5, 28.5, 35.5, 42.5]
    week_colors = ['#f0f0f0', '#e0e0e0']
    d_era5 = det[det.clim_basis == 'era5'] if 'clim_basis' in det.columns else det
    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for i in range(6):
            ax.axvspan(week_bounds[i], week_bounds[i+1],
                       alpha=0.3, color=week_colors[i % 2], zorder=0)
            ax.text((week_bounds[i]+week_bounds[i+1])/2, 1.02, f'W{i+1}',
                    ha='center', fontsize=7, transform=ax.get_xaxis_transform())
        for m in DET_MODELS:
            sub = d_era5[(d_era5.variable==var) & (d_era5.model==m) &
                         (d_era5.region=='All India') & (d_era5.scale=='daily')]
            vals = np.array([sub[sub.lead==d]['pcc'].mean() for d in DAYS])
            ax.plot(DAYS, vals, color=MODEL_COLOR[m],
                    lw=1.8 if m != 'Persistence' else 1.2,
                    ls='-' if m != 'Persistence' else '--',
                    alpha=0.9, label=m)
        ax.axhline(0.5, color='orange', lw=0.8, ls=':', alpha=0.8)
        ax.axhline(0,   color='#aaaaaa', lw=0.8, ls='--')
        ax.set_xlim(1, 42); ax.set_ylim(-0.3, 1.05)
        ax.set_xlabel('Forecast Lead Day', fontsize=9)
        ax.set_ylabel('PCC', fontsize=9)
        ax.set_xticks([1, 7, 14, 21, 28, 35, 42])
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', lw=0.4, alpha=0.5)
        ax.set_title(f'{var} ({VAR_UNIT[var]})', fontsize=10)
    handles = [plt.Line2D([0],[0], color=MODEL_COLOR[m], lw=1.8,
                          ls='-' if m != 'Persistence' else '--', label=m)
               for m in DET_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig11_daily_pcc.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12 — MSSS regional (both bases)
# ══════════════════════════════════════════════════════════════════════════════
def fig12_msss_regional(det, bases):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    fig.suptitle('MSSS vs ERA5 Clim — All Regions  (solid=ERA5 | dashed=model-own)',
                 fontsize=11, y=1.01)
    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            _plot_models_both_bases(ax, det, var, reg, 'msss_clim',
                                    models=DET_MODELS, bases=bases)
            _style_ax(ax, ylabel='MSSS' if ci==0 else '', ylim=(-0.7, 1.1),
                      title=REGION_LABEL[reg] if ri==0 else '')
            if ci == 0:
                ax.text(-0.35, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10, fontweight='bold')
    _legend_both_bases(fig, DET_MODELS, bases)
    fig.tight_layout()
    _savefig(fig, 'fig12_msss_regional.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 13 — CRPSS regional
# ══════════════════════════════════════════════════════════════════════════════
def fig13_crpss_regional(prob):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    fig.suptitle('CRPSS vs Climatology — All IMD Regions', fontsize=11, y=1.01)
    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            for m in PROB_MODELS:
                vals = _agg(prob, var, m, reg, 'weekly', 'crpss_clim')
                if not np.all(np.isnan(vals)):
                    ax.plot(WEEKS, vals, color=MODEL_COLOR[m],
                            marker=MODEL_MARKER[m], lw=2.0, ms=5, label=m)
            _style_ax(ax, ylabel='CRPSS' if ci==0 else '', ylim=(-0.5, 1.1),
                      title=REGION_LABEL[reg] if ri==0 else '')
            if ci == 0:
                ax.text(-0.35, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=10, fontweight='bold')
    handles = [plt.Line2D([0],[0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                          lw=2.0, ms=5, label=m) for m in PROB_MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig13_crpss_regional.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 14 — Side-by-side PCC era5 vs model_own (FuXi + ECMWF only)
# ══════════════════════════════════════════════════════════════════════════════
def fig14_basis_comparison(det):
    """Direct comparison: how much does model-own clim change the PCC?"""
    models_own = ['FuXi', 'ECMWF']    # only these have model-own clim
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('ERA5 clim vs Model-own clim baseline — PCC impact\n'
                 'Gap = how much systematic model bias inflates/deflates PCC',
                 fontsize=11, y=1.02)
    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(['All India', 'central_india']):
            ax = axes[ri, ci]
            for m in models_own:
                for basis in ('era5', 'model_own'):
                    vals = _agg(det, var, m, reg, 'weekly', 'pcc', basis=basis)
                    if np.all(np.isnan(vals)):
                        continue
                    ax.plot(WEEKS, vals, color=MODEL_COLOR[m],
                            marker=MODEL_MARKER[m], lw=2.0, ms=5,
                            ls=BASIS_LS[basis],
                            label=f'{m} ({BASIS_LABEL[basis]})')
            _style_ax(ax, ylabel='PCC', ylim=(-0.3, 1.05),
                      title=f'{var} — {REGION_LABEL.get(reg, reg)}')
    handles = []
    for m in models_own:
        for basis in ('era5', 'model_own'):
            handles.append(plt.Line2D([0],[0], color=MODEL_COLOR[m],
                                      marker=MODEL_MARKER[m], lw=2.0, ms=5,
                                      ls=BASIS_LS[basis],
                                      label=f'{m} ({BASIS_LABEL[basis]})'))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    fig.tight_layout()
    _savefig(fig, 'fig14_basis_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
ALL_FIGS = {
    1:  ('fig01_apcc_allIndia',     lambda d,p,b,br: fig01_apcc_allIndia(d, p, b)),
    2:  ('fig02_apcc_regional_TP',  lambda d,p,b,br: fig02_apcc_regional_TP(d, b)),
    3:  ('fig03_apcc_regional_Z500',lambda d,p,b,br: fig03_apcc_regional_Z500(d, b)),
    4:  ('fig04_bias_rmse',         lambda d,p,b,br: fig04_bias_rmse(d, p)),
    5:  ('fig05_msss_allIndia',     lambda d,p,b,br: fig05_msss_allIndia(d, b)),
    6:  ('fig06_crpss_allIndia',    lambda d,p,b,br: fig06_crpss_allIndia(p)),
    7:  ('fig07_bss_events',        lambda d,p,b,br: fig07_bss_events(br)),
    8:  ('fig08_scatter_r2_mae',    lambda d,p,b,br: fig08_scatter_r2_mae(d)),
    9:  ('fig09_scorecard_era5',    lambda d,p,b,br: fig09_scorecard_era5(d, p)),
    10: ('fig10_scorecard_own',     lambda d,p,b,br: fig10_scorecard_own(d, p)),
    11: ('fig11_daily_pcc',         lambda d,p,b,br: fig11_daily_pcc(d)),
    12: ('fig12_msss_regional',     lambda d,p,b,br: fig12_msss_regional(d, b)),
    13: ('fig13_crpss_regional',    lambda d,p,b,br: fig13_crpss_regional(p)),
    14: ('fig14_basis_comparison',  lambda d,p,b,br: fig14_basis_comparison(d)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--figs', nargs='+', type=int, default=None)
    ap.add_argument('--basis', choices=['era5', 'model_own', 'both'], default='both')
    args = ap.parse_args()

    data_dir = _find_data_dir(args.data_dir)
    det   = pd.read_csv(os.path.join(data_dir, 'skill_deterministic.csv'))
    prob  = pd.read_csv(os.path.join(data_dir, 'skill_probabilistic.csv'))
    brier = pd.read_csv(os.path.join(data_dir, 'skill_brier.csv'))

    print(f"  det  rows: {len(det)}"
          + (f"  (clim_basis: {sorted(det.clim_basis.unique())})"
             if 'clim_basis' in det.columns else ""))
    print(f"  prob rows: {len(prob)}")
    print(f"  brier rows: {len(brier)}")
    print(f"  models: {sorted(det.model.unique())}")

    bases = (['era5'] if args.basis == 'era5'
             else ['model_own'] if args.basis == 'model_own'
             else ['era5', 'model_own'])

    figs_to_run = sorted(args.figs) if args.figs else sorted(ALL_FIGS.keys())
    print(f"\nGenerating {len(figs_to_run)} figures → {ODIR}\n")
    for n in figs_to_run:
        name, fn = ALL_FIGS[n]
        print(f"[fig{n:02d}] {name}")
        try:
            fn(det, prob, bases, brier)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
    print(f"\nDone. Figures in:\n  {ODIR}")


if __name__ == '__main__':
    main()
