#!/usr/bin/env python3
"""
plot_pcc.py
Anomaly Pattern Correlation Coefficient (PCC / ACC) vs forecast week.

Outputs (plots_results_pres/):
  pcc_allIndia.png             — All-India PCC, both variables, all models
  pcc_regional_TP.png          — 5-region panel for TP
  pcc_regional_Z500.png        — 5-region panel for Z500
  pcc_daily.png                — Daily PCC horizon (lead 1–42), All India
  pcc_basis_comparison.png     — ERA5 vs model-own baseline, FuXi & ECMWF
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    DET_MODELS, WEEKS, WLABELS, WLONG, VAR_UNITS, VAR_LONG,
                    REGIONS, REGION_LABEL, style_week_axis, model_legend_handles)

set_style()


def load_det():
    return pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))


def _pcc_mean_std(df, var, model, region, scale, basis, weeks=WEEKS):
    sub = df[(df.variable == var) & (df.model == model) &
             (df.region == region) & (df.scale == scale) &
             (df.clim_basis == basis)]
    means = np.array([sub[sub.lead == w]['pcc'].mean() for w in weeks])
    stds  = np.array([sub[sub.lead == w]['pcc'].std()  for w in weeks])
    return means, stds


def fig_allIndia(df):
    """Side-by-side TP and Z500, all models, weekly PCC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Anomaly Pattern Correlation (PCC) — All India\n'
                 'ERA5 anomaly basis  |  Weekly aggregation  |  Mean across 13 initialisations',
                 fontsize=12, y=1.03)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            means, stds = _pcc_mean_std(df, var, m, 'All India', 'weekly', 'era5')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            lw = 1.5 if m == 'Persistence' else 2.2
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(WEEKS, means, color=c, marker=mk, lw=lw, ms=7, ls=ls,
                    label=m, zorder=3)

        style_week_axis(ax, ylabel='PCC', ylim=(-0.25, 1.05),
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')
        ax.axhline(0.5, color='darkorange', lw=1.0, ls=':', alpha=0.8,
                   label='PCC = 0.5')

    handles = model_legend_handles(DET_MODELS)
    handles += [plt.Line2D([0], [0], color='darkorange', lw=1.0, ls=':',
                           label='PCC = 0.5')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, 'pcc_allIndia.png')


def fig_regional(df, var):
    """1×5 regional panel for one variable."""
    unit = VAR_UNITS[var]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — PCC by IMD Region  (ERA5 basis)\n'
                 'Mean across 13 initialisations',
                 fontsize=12, y=1.03)

    for ci, reg in enumerate(REGIONS):
        ax = axes[ci]
        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            means, stds = _pcc_mean_std(df, var, m, reg, 'weekly', 'era5')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(WEEKS, means, color=c, marker=mk, lw=1.8, ms=5, ls=ls,
                    zorder=3)

        ax.axhline(0.5, color='darkorange', lw=0.8, ls=':', alpha=0.7)
        style_week_axis(ax, ylabel='PCC' if ci == 0 else '',
                        ylim=(-0.4, 1.05), title=REGION_LABEL[reg])

    handles = model_legend_handles(DET_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, f'pcc_regional_{var}.png')


def fig_daily(df):
    """Daily PCC vs lead day (1–42) with week shading bands."""
    DAYS = list(range(1, 43))
    sub_era5 = df[(df.clim_basis == 'era5') & (df.scale == 'daily')]

    week_bounds = [0.5, 7.5, 14.5, 21.5, 28.5, 35.5, 42.5]
    band_colors = ['#f5f5f5', '#e8e8e8']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Daily PCC Horizon — All India  (ERA5 anomaly basis)\n'
                 'Mean across 13 weekly initialisations',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for i in range(6):
            ax.axvspan(week_bounds[i], week_bounds[i+1],
                       alpha=0.4, color=band_colors[i % 2], zorder=0)
            ax.text((week_bounds[i] + week_bounds[i+1]) / 2, 1.03,
                    f'W{i+1}', ha='center', fontsize=8,
                    transform=ax.get_xaxis_transform())

        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            sub = sub_era5[(sub_era5.variable == var) & (sub_era5.model == m) &
                           (sub_era5.region == 'All India')]
            vals = np.array([sub[sub.lead == d]['pcc'].mean() for d in DAYS])
            if np.all(np.isnan(vals)):
                continue
            c = MODEL_COLORS[m]
            lw = 1.4 if m == 'Persistence' else 2.0
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(DAYS, vals, color=c, lw=lw, ls=ls, alpha=0.9, label=m)

        ax.axhline(0.5, color='darkorange', lw=0.9, ls=':', alpha=0.8,
                   label='PCC = 0.5')
        ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=1)
        ax.set_xlim(1, 42); ax.set_ylim(-0.3, 1.05)
        ax.set_xlabel('Forecast Lead Day', fontsize=10)
        ax.set_ylabel('PCC', fontsize=10)
        ax.set_xticks([1, 7, 14, 21, 28, 35, 42])
        ax.set_xticklabels(['D1', 'D7', 'D14', 'D21', 'D28', 'D35', 'D42'])
        ax.set_title(f'{VAR_LONG[var]}  ({VAR_UNITS[var]})', fontsize=11)

    handles = model_legend_handles(DET_MODELS)
    handles += [plt.Line2D([0], [0], color='darkorange', lw=1.0, ls=':', label='PCC = 0.5')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, 'pcc_daily.png')


def fig_basis_comparison(df):
    """ERA5 vs model-own baseline — FuXi and ECMWF only."""
    models_own = ['FuXi', 'ECMWF']
    basis_ls  = {'era5': '-', 'model_own': '--'}
    basis_lab = {'era5': 'ERA5 clim', 'model_own': 'Model-own clim'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('PCC: ERA5 Clim vs Model-Own Clim Baseline\n'
                 'Solid = ERA5 basis  |  Dashed = Model-own basis',
                 fontsize=12, y=1.02)

    for ri, var in enumerate(['TP', 'Z500']):
        for ci, reg in enumerate(['All India', 'central_india']):
            ax = axes[ri, ci]
            for m in models_own:
                for basis in ('era5', 'model_own'):
                    sub = df[(df.variable == var) & (df.model == m) &
                             (df.region == reg) & (df.scale == 'weekly') &
                             (df.clim_basis == basis)]
                    vals = np.array([sub[sub.lead == w]['pcc'].mean()
                                     for w in WEEKS])
                    if np.all(np.isnan(vals)):
                        continue
                    ax.plot(WEEKS, vals, color=MODEL_COLORS[m],
                            marker=MODEL_MARKERS[m], lw=2.0, ms=6,
                            ls=basis_ls[basis],
                            label=f'{m} ({basis_lab[basis]})')

            style_week_axis(ax, ylabel='PCC', ylim=(-0.3, 1.05),
                            title=f'{var} — {REGION_LABEL.get(reg, reg)}')

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m],
                          marker=MODEL_MARKERS[m], lw=2.0, ms=6,
                          ls=basis_ls[b],
                          label=f'{m} ({basis_lab[b]})')
               for m in models_own for b in ('era5', 'model_own')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    fig.tight_layout()
    savefig(fig, 'pcc_basis_comparison.png')


def main():
    print("Loading deterministic skill...")
    df = load_det()
    print(f"  rows: {len(df)}")

    print("\nAll-India PCC (both vars)...")
    fig_allIndia(df)

    for var in ['TP', 'Z500']:
        print(f"[{var}] Regional PCC...")
        fig_regional(df, var)

    print("\nDaily PCC horizon...")
    fig_daily(df)

    print("\nBasis comparison (FuXi, ECMWF)...")
    fig_basis_comparison(df)

    print("\nDone.")


if __name__ == '__main__':
    main()
