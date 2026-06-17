#!/usr/bin/env python3
"""
plot_rmse.py
Root Mean Square Error (RMSE) and Normalised RMSE vs forecast week.

Outputs (plots_results_pres/):
  rmse_allIndia.png            — All-India RMSE, both variables, all models
  rmse_regional_TP.png         — 5-region panel for TP
  rmse_regional_Z500.png       — 5-region panel for Z500
  rmse_normalised_allIndia.png — NRMSE (RMSE / obs σ) = spread-to-signal ratio
  rmse_scorecard.png           — Heatmap model × week
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    DET_MODELS, WEEKS, WLABELS, VAR_UNITS, VAR_LONG,
                    REGIONS, REGION_LABEL, style_week_axis, model_legend_handles)

set_style()


def load_det():
    df = pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))
    return df[(df.scale == 'weekly') & (df.clim_basis == 'era5')]


def _stats(df, var, model, region, metric, weeks=WEEKS):
    sub = df[(df.variable == var) & (df.model == model) & (df.region == region)]
    means = np.array([sub[sub.lead == w][metric].mean() for w in weeks])
    stds  = np.array([sub[sub.lead == w][metric].std()  for w in weeks])
    return means, stds


def fig_allIndia(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('RMSE — All India  (ERA5 anomaly basis)\n'
                 'Shading = ±1 std across 13 initialisations',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        unit = VAR_UNITS[var]
        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            means, stds = _stats(df, var, m, 'All India', 'rmse')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(WEEKS, means, color=c, marker=mk, lw=2.0, ms=7, ls=ls,
                    label=m, zorder=3)
            ax.fill_between(WEEKS, np.maximum(0, means - stds), means + stds,
                            color=c, alpha=0.10, zorder=2)

        style_week_axis(ax, ylabel=f'RMSE ({unit})',
                        title=f'{VAR_LONG[var]}  ({unit})')
        ax.set_ylim(bottom=0)

    handles = model_legend_handles(DET_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, 'rmse_allIndia.png')


def fig_regional(df, var):
    unit = VAR_UNITS[var]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — RMSE by IMD Region  (ERA5 basis)\n'
                 'Shading = ±1 std',
                 fontsize=12, y=1.03)

    for ci, reg in enumerate(REGIONS):
        ax = axes[ci]
        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            means, stds = _stats(df, var, m, reg, 'rmse')
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(WEEKS, means, color=c, marker=mk, lw=1.8, ms=5, ls=ls,
                    zorder=3)
            ax.fill_between(WEEKS, np.maximum(0, means - stds), means + stds,
                            color=c, alpha=0.10, zorder=2)
        style_week_axis(ax, ylabel=f'RMSE ({unit})' if ci == 0 else '',
                        title=REGION_LABEL[reg])
        ax.set_ylim(bottom=0)

    handles = model_legend_handles(DET_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, f'rmse_regional_{var}.png')


def fig_normalised(df):
    """Normalised RMSE = RMSE / obs_std (skill relative to obs variability)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Normalised RMSE (RMSE / Obs σ) — All India  (ERA5 basis)\n'
                 'Values < 1 = smaller error than obs spread; < 0.5 = good skill',
                 fontsize=12, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        ax = axes[ci]
        for m in DET_MODELS:
            if m not in MODEL_COLORS:
                continue
            sub = df[(df.variable == var) & (df.model == m) &
                     (df.region == 'All India')]
            rmse_m = np.array([sub[sub.lead == w]['rmse'].mean()     for w in WEEKS])
            obs_std = np.array([sub[sub.lead == w]['obs_std'].mean() for w in WEEKS])
            nrmse = rmse_m / np.where(obs_std > 0, obs_std, np.nan)
            if np.all(np.isnan(nrmse)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ls = '--' if m == 'Persistence' else '-'
            ax.plot(WEEKS, nrmse, color=c, marker=mk, lw=2.0, ms=7, ls=ls,
                    label=m, zorder=3)

        ax.axhline(1.0, color='#888', lw=1.0, ls=':', alpha=0.8, label='NRMSE = 1')
        ax.axhline(0.5, color='darkorange', lw=0.9, ls=':', alpha=0.7, label='NRMSE = 0.5')
        style_week_axis(ax, ylabel='NRMSE (RMSE / Obs σ)',
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')
        ax.set_ylim(bottom=0)

    handles = model_legend_handles(DET_MODELS)
    handles += [
        plt.Line2D([0], [0], color='#888', lw=1.0, ls=':', label='NRMSE = 1'),
        plt.Line2D([0], [0], color='darkorange', lw=0.9, ls=':', label='NRMSE = 0.5'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.tight_layout()
    savefig(fig, 'rmse_normalised_allIndia.png')


def fig_scorecard(df):
    """Heatmap RMSE: model × week for both variables."""
    SCORE_MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('RMSE Scorecard — All India  (ERA5 basis)', fontsize=13, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        unit = VAR_UNITS[var]
        ax = axes[ci]
        data = np.array([[_stats(df, var, m, 'All India', 'rmse')[0][w - 1]
                          for w in WEEKS] for m in SCORE_MODELS])
        vmax = np.nanmax(data) * 1.02
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)

        ax.set_xticks(range(6)); ax.set_xticklabels(WLABELS)
        ax.set_yticks(range(len(SCORE_MODELS))); ax.set_yticklabels(SCORE_MODELS)
        ax.set_xlabel('Forecast Week')
        ax.set_title(f'{VAR_LONG[var]}  ({unit})', fontsize=11)

        for i in range(len(SCORE_MODELS)):
            for j in range(len(WEEKS)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color='white' if v > 0.6 * vmax else 'black')

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f'RMSE ({unit})', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'rmse_scorecard.png')


def main():
    print("Loading deterministic skill (weekly, era5)...")
    df = load_det()
    print(f"  rows: {len(df)}")

    print("\nAll-India RMSE...")
    fig_allIndia(df)

    for var in ['TP', 'Z500']:
        print(f"[{var}] Regional RMSE...")
        fig_regional(df, var)

    print("\nNormalised RMSE...")
    fig_normalised(df)

    print("\nRMSE scorecard...")
    fig_scorecard(df)

    print("\nDone.")


if __name__ == '__main__':
    main()
