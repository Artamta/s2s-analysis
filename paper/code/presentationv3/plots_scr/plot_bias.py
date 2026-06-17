#!/usr/bin/env python3
"""
plot_bias.py
Weekly mean bias (forecast minus observed) with spread across initialisations.

Outputs (plots_results_pres/):
  bias_allIndia_TP.png          — All-India bias by week, all models, TP
  bias_allIndia_Z500.png        — same for Z500
  bias_regional_TP.png          — 5-region panel, TP bias
  bias_regional_Z500.png        — 5-region panel, Z500 bias
  bias_scorecard.png            — Heatmap: model × week, both variables
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    DET_MODELS, WEEKS, WLABELS, VAR_UNITS, VAR_LONG,
                    REGIONS, REGION_LABEL, style_week_axis, model_legend_handles)

set_style()

MODELS_BIAS = ['SPIRE', 'FuXi', 'ECMWF', 'MME']


def load_det():
    df = pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))
    return df[(df.scale == 'weekly') & (df.clim_basis == 'era5')]


def _get_bias_stats(df, var, model, region, weeks=WEEKS):
    """Return (mean_bias, std_bias) arrays per week."""
    sub = df[(df.variable == var) & (df.model == model) & (df.region == region)]
    means = np.array([sub[sub.lead == w]['bias'].mean() for w in weeks])
    stds  = np.array([sub[sub.lead == w]['bias'].std()  for w in weeks])
    return means, stds


def fig_allIndia(df, var):
    unit = VAR_UNITS[var]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — Weekly Mean Bias (Forecast − Observed)\n'
                 f'All India  |  ERA5 anomaly basis  |  Shading = ±1 std across 13 inits',
                 fontsize=12)

    for m in MODELS_BIAS:
        means, stds = _get_bias_stats(df, var, m, 'All India')
        c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
        ax.plot(WEEKS, means, color=c, marker=mk, lw=2.0, ms=7, label=m, zorder=3)
        ax.fill_between(WEEKS, means - stds, means + stds,
                        color=c, alpha=0.12, zorder=2)

    style_week_axis(ax, ylabel=f'Bias ({unit})', title='')
    ax.axhline(0, color='#555', lw=1.2, ls='-', zorder=1)

    handles = model_legend_handles(MODELS_BIAS)
    ax.legend(handles=handles, loc='best', frameon=True, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, f'bias_allIndia_{var}.png')


def fig_regional(df, var):
    unit = VAR_UNITS[var]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — Weekly Bias by IMD Region\n'
                 f'Shading = ±1 std  |  ERA5 anomaly basis',
                 fontsize=12, y=1.03)

    for ci, reg in enumerate(REGIONS):
        ax = axes[ci]
        for m in MODELS_BIAS:
            means, stds = _get_bias_stats(df, var, m, reg)
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ax.plot(WEEKS, means, color=c, marker=mk, lw=1.8, ms=5, label=m, zorder=3)
            ax.fill_between(WEEKS, means - stds, means + stds,
                            color=c, alpha=0.12, zorder=2)
        style_week_axis(ax, ylabel=f'Bias ({unit})' if ci == 0 else '',
                        title=REGION_LABEL[reg])
        ax.axhline(0, color='#555', lw=1.0, ls='-')

    handles = model_legend_handles(MODELS_BIAS)
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.07), frameon=True)
    fig.tight_layout()
    savefig(fig, f'bias_regional_{var}.png')


def fig_scorecard(df):
    """Heatmap: rows=models, cols=weeks, panels=variable."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Mean Bias Scorecard — All India  (ERA5 basis)',
                 fontsize=13, y=1.02)

    for ci, var in enumerate(['TP', 'Z500']):
        unit = VAR_UNITS[var]
        ax = axes[ci]
        data = np.array([[_get_bias_stats(df, var, m, 'All India')[0][w - 1]
                          for w in WEEKS] for m in MODELS_BIAS])

        absmax = np.nanmax(np.abs(data)) * 1.05
        norm = TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax)
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', norm=norm)

        ax.set_xticks(range(6)); ax.set_xticklabels(WLABELS)
        ax.set_yticks(range(len(MODELS_BIAS))); ax.set_yticklabels(MODELS_BIAS)
        ax.set_xlabel('Forecast Week')
        ax.set_title(f'{VAR_LONG[var]}  ({unit})', fontsize=11)

        for i, m in enumerate(MODELS_BIAS):
            for j, w in enumerate(WEEKS):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if abs(v) > 0.5 * absmax else 'black')

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f'Bias ({unit})', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'bias_scorecard.png')


def main():
    print("Loading deterministic skill (weekly, era5)...")
    df = load_det()
    print(f"  rows: {len(df)}")

    for var in ['TP', 'Z500']:
        print(f"\n[{var}] All-India bias plot...")
        fig_allIndia(df, var)

        print(f"[{var}] Regional bias plot...")
        fig_regional(df, var)

    print("\nBias scorecard...")
    fig_scorecard(df)
    print("\nDone.")


if __name__ == '__main__':
    main()
