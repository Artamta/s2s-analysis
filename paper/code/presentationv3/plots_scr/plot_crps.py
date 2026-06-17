#!/usr/bin/env python3
"""
plot_crps.py
Probabilistic skill: CRPS, CRPSS, and spread–skill ratio.

Outputs (plots_results_pres/):
  crps_absolute_allIndia.png   — Absolute CRPS + clim/pers reference, All India
  crpss_allIndia.png           — CRPSS vs climatology, All India
  crpss_vs_pers_allIndia.png   — CRPSS vs both climatology and persistence
  crpss_regional_TP.png        — CRPSS by region, TP
  crpss_regional_Z500.png      — CRPSS by region, Z500
  spread_skill_allIndia.png    — Spread–skill ratio (ensemble spread / RMSE)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    PROB_MODELS, WEEKS, WLABELS, VAR_UNITS, VAR_LONG,
                    REGIONS, REGION_LABEL, style_week_axis, model_legend_handles)

set_style()


def load_prob():
    df = pd.read_csv(os.path.join(DATADIR, 'skill_probabilistic.csv'))
    return df[(df.scale == 'weekly') & (df.clim_basis == 'era5')]


def _agg(df, var, model, region, metric, weeks=WEEKS):
    sub = df[(df.variable == var) & (df.model == model) & (df.region == region)]
    means = np.array([sub[sub.lead == w][metric].mean() for w in weeks])
    stds  = np.array([sub[sub.lead == w][metric].std()  for w in weeks])
    return means, stds


def _ref_agg(df, var, region, metric, weeks=WEEKS):
    """Reference values from any model (climatology / persistence are model-independent)."""
    sub = df[(df.variable == var) & (df.region == region) & (df.model == PROB_MODELS[0])]
    return np.array([sub[sub.lead == w][metric].mean() for w in weeks])


def fig_crps_absolute(df):
    """Absolute CRPS values + climatology and persistence references."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Absolute CRPS — All India  (lower = better)\n'
                 'ERA5 anomaly basis  |  Mean across 13 initialisations',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        unit = VAR_UNITS[var]

        # Reference lines
        crps_clim = _ref_agg(df, var, 'All India', 'crps_clim')
        crps_pers = _ref_agg(df, var, 'All India', 'crps_pers')
        ax.plot(WEEKS, crps_clim, color='#555', lw=1.5, ls=':', label='Climatology')
        ax.plot(WEEKS, crps_pers, color='#888', lw=1.5, ls='--', label='Persistence')

        for m in PROB_MODELS:
            means, stds = _agg(df, var, m, 'All India', 'crps')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ax.plot(WEEKS, means, color=c, marker=mk, lw=2.0, ms=7, label=m, zorder=3)

        style_week_axis(ax, ylabel=f'CRPS ({unit})',
                        title=f'{VAR_LONG[var]}  ({unit})')
        ax.set_ylim(bottom=0)

    handles = (model_legend_handles(PROB_MODELS) +
               [plt.Line2D([0],[0], color='#555', lw=1.5, ls=':', label='Climatology'),
                plt.Line2D([0],[0], color='#888', lw=1.5, ls='--', label='Persistence')])
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, 'crps_absolute_allIndia.png')


def fig_crpss(df):
    """CRPSS vs climatology."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('CRPSS vs Climatology — All India\n'
                 'ERA5 anomaly basis  |  > 0 = better than climatology',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in PROB_MODELS:
            means, stds = _agg(df, var, m, 'All India', 'crpss_clim')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ax.plot(WEEKS, means, color=c, marker=mk, lw=2.0, ms=7, label=m, zorder=3)

        style_week_axis(ax, ylabel='CRPSS', ylim=(-0.45, 1.05),
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')

    handles = model_legend_handles(PROB_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, 'crpss_allIndia.png')


def fig_crpss_dual_ref(df):
    """CRPSS vs both climatology (solid) and persistence (dashed)."""
    ref_ls  = {'crpss_clim': '-',  'crpss_pers': '--'}
    ref_lab = {'crpss_clim': 'vs Clim', 'crpss_pers': 'vs Pers'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('CRPSS — All India  (solid = vs Clim | dashed = vs Persistence)',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in PROB_MODELS:
            for metric, ls in ref_ls.items():
                means, _ = _agg(df, var, m, 'All India', metric)
                if np.all(np.isnan(means)):
                    continue
                label = f'{m} ({ref_lab[metric]})' if ci == 0 else None
                ax.plot(WEEKS, means, color=MODEL_COLORS[m],
                        marker=MODEL_MARKERS[m], lw=2.0, ms=6, ls=ls,
                        label=label, zorder=3)

        style_week_axis(ax, ylabel='CRPSS', ylim=(-0.6, 1.05),
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')

    handles = [plt.Line2D([0],[0], color=MODEL_COLORS[m],
                          marker=MODEL_MARKERS[m], lw=2.0, ms=6,
                          ls=ls, label=f'{m} ({ref_lab[metric]})')
               for m in PROB_MODELS
               for metric, ls in ref_ls.items()]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, 'crpss_vs_pers_allIndia.png')


def fig_crpss_regional(df, var):
    """CRPSS by region for one variable."""
    unit = VAR_UNITS[var]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — CRPSS by IMD Region  (ERA5 basis)',
                 fontsize=12, y=1.03)

    for ci, reg in enumerate(REGIONS):
        ax = axes[ci]
        for m in PROB_MODELS:
            means, stds = _agg(df, var, m, reg, 'crpss_clim')
            if np.all(np.isnan(means)):
                continue
            ax.plot(WEEKS, means, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                    lw=1.8, ms=5, zorder=3)
        style_week_axis(ax, ylabel='CRPSS' if ci == 0 else '',
                        ylim=(-0.5, 1.05), title=REGION_LABEL[reg])

    handles = model_legend_handles(PROB_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, f'crpss_regional_{var}.png')


def fig_spread_skill(df):
    """Spread–skill ratio (SSR = ensemble spread / RMSE); perfect = 1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Spread–Skill Ratio (SSR = Ensemble Spread / RMSE) — All India\n'
                 'ERA5 basis  |  SSR = 1 (dashed) = perfect ensemble calibration',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in PROB_MODELS:
            sub = df[(df.variable == var) & (df.model == m) &
                     (df.region == 'All India')]
            spread = np.array([sub[sub.lead == w]['spread'].mean() for w in WEEKS])
            rmse   = np.array([sub[sub.lead == w]['rmse'].mean()   for w in WEEKS])
            ssr    = spread / np.where(rmse > 0, rmse, np.nan)
            if np.all(np.isnan(ssr)):
                continue
            ax.plot(WEEKS, ssr, color=MODEL_COLORS[m],
                    marker=MODEL_MARKERS[m], lw=2.0, ms=7, label=m, zorder=3)

        ax.axhline(1.0, color='#444', lw=1.2, ls='--', alpha=0.8, label='SSR = 1 (perfect)')
        ax.axhline(0.5, color='darkorange', lw=0.9, ls=':', alpha=0.7, label='SSR = 0.5')
        style_week_axis(ax, ylabel='Spread / RMSE (SSR)',
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')
        ax.set_ylim(bottom=0)

    handles = model_legend_handles(PROB_MODELS)
    handles += [plt.Line2D([0],[0], color='#444', lw=1.2, ls='--', label='SSR = 1 (perfect)'),
                plt.Line2D([0],[0], color='darkorange', lw=0.9, ls=':', label='SSR = 0.5')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, 'spread_skill_allIndia.png')


def main():
    print("Loading probabilistic skill (weekly, era5)...")
    df = load_prob()
    print(f"  rows: {len(df)}")

    print("\nAbsolute CRPS...")
    fig_crps_absolute(df)

    print("\nCRPSS vs climatology...")
    fig_crpss(df)

    print("\nCRPSS dual reference (clim + pers)...")
    fig_crpss_dual_ref(df)

    for var in ['TP', 'Z500']:
        print(f"\n[{var}] Regional CRPSS...")
        fig_crpss_regional(df, var)

    print("\nSpread–skill ratio...")
    fig_spread_skill(df)

    print("\nDone.")


if __name__ == '__main__':
    main()
