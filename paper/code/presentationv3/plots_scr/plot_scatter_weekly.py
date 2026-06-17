#!/usr/bin/env python3
"""
plot_scatter_weekly.py
Weekly scatter plots: forecast mean vs observed mean for each model × variable.

Outputs (plots_results_pres/):
  scatter_weekly_TP.png   — 5-model × 6-week grid for Total Precipitation
  scatter_weekly_Z500.png — same for 500-hPa Geopotential Height
  scatter_wk12_compare_TP.png   — W1 vs W2 side-by-side, all models
  scatter_wk12_compare_Z500.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import pearsonr
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    DET_MODELS, WEEKS, WLONG, VAR_UNITS, VAR_LONG)

set_style()


def load_det():
    path = os.path.join(DATADIR, 'skill_deterministic.csv')
    df = pd.read_csv(path, parse_dates=['init_date'])
    df = df[(df.scale == 'weekly') & (df.clim_basis == 'era5')]
    return df


def _scatter_panel(ax, x, y, init_dates, model, title=''):
    """Draw one scatter panel: obs vs fcst with stats."""
    if len(x) < 2:
        ax.set_visible(False)
        return

    # Sort by init date for colour mapping
    order = np.argsort(init_dates)
    x, y, dates = x[order], y[order], init_dates[order]

    sc = ax.scatter(x, y, c=range(len(dates)), cmap='plasma',
                    s=60, zorder=4, edgecolors='white', linewidths=0.5,
                    vmin=0, vmax=len(dates) - 1)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.12
    rng = np.array([lo - pad, hi + pad])

    # 1:1 perfect forecast line
    ax.plot(rng, rng, 'k--', lw=1.2, alpha=0.5, zorder=2, label='1:1')

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    ax.plot(rng, slope * rng + intercept, color=MODEL_COLORS.get(model, '#333'),
            lw=1.8, alpha=0.85, zorder=3)

    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)

    ax.set_xlim(rng); ax.set_ylim(rng)
    ax.set_aspect('equal', adjustable='box')
    if title:
        ax.set_title(title, fontsize=10, pad=3)

    # Stats annotation
    txt = f'PCC={r:.2f}\nRMSE={rmse:.3f}\nBias={bias:+.3f}\nN={len(x)}'
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.85))

    return sc


def fig_model_by_week(df, var):
    """5-models × 6-weeks scatter grid for one variable."""
    models = [m for m in DET_MODELS if m != 'Persistence']
    unit = VAR_UNITS[var]
    nrow, ncol = len(models), 6

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 3.5),
                             squeeze=False)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — Weekly Forecast vs Observed Mean\n'
                 f'Coloured by initialisation date  (ERA5 anomaly basis)',
                 fontsize=13, y=1.01)

    sub_v = df[df.variable == var]

    for ri, model in enumerate(models):
        sub_m = sub_v[sub_v.model == model]
        for ci, wk in enumerate(WEEKS):
            ax = axes[ri, ci]
            sub_w = sub_m[sub_m.lead == wk].dropna(subset=['fcst_mean', 'obs_mean'])
            x = sub_w['obs_mean'].values
            y = sub_w['fcst_mean'].values
            dates = sub_w['init_date'].values

            sc = _scatter_panel(ax, x, y, dates, model,
                                title=f'{WLONG[ci]}' if ri == 0 else '')

            if ci == 0:
                ax.set_ylabel(f'{model}\nFcst ({unit})', fontsize=9,
                              color=MODEL_COLORS.get(model, 'k'))
            else:
                ax.set_ylabel('')
            ax.set_xlabel(f'Obs ({unit})', fontsize=8)

        # Colorbar for this row (rightmost axis)
        if sc is not None:
            cax = axes[ri, -1].inset_axes([1.04, 0, 0.06, 1])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_ticks([0, 6, 12])
            cb.set_ticklabels(['Jan 1', 'Feb 12', 'Mar 26'])
            cb.ax.tick_params(labelsize=7)
            cb.set_label('Init date', fontsize=7)

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    savefig(fig, f'scatter_weekly_{var}.png')


def fig_compare_weeks(df, var, week_pair=(1, 2)):
    """All models side-by-side for two key forecast weeks."""
    models = [m for m in DET_MODELS if m != 'Persistence']
    unit = VAR_UNITS[var]
    sub_v = df[df.variable == var]

    nrow, ncol = len(models), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3.8),
                             squeeze=False)
    fig.suptitle(f'{VAR_LONG[var]}  ({unit}) — Model Comparison W{week_pair[0]} vs W{week_pair[1]}\n'
                 f'Scatter: weekly domain mean (13 initialisations)',
                 fontsize=12, y=1.01)

    for ri, model in enumerate(models):
        sub_m = sub_v[sub_v.model == model]
        for ci, wk in enumerate(week_pair):
            ax = axes[ri, ci]
            sub_w = sub_m[sub_m.lead == wk].dropna(subset=['fcst_mean', 'obs_mean'])
            x = sub_w['obs_mean'].values
            y = sub_w['fcst_mean'].values
            dates = sub_w['init_date'].values
            title = f'Week {wk}' if ri == 0 else ''
            _scatter_panel(ax, x, y, dates, model, title=title)

            if ci == 0:
                ax.set_ylabel(f'{model}\nFcst ({unit})', fontsize=9,
                              color=MODEL_COLORS.get(model, 'k'))
            ax.set_xlabel(f'Obs ({unit})', fontsize=8)

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    savefig(fig, f'scatter_wk{week_pair[0]}{week_pair[1]}_compare_{var}.png')


def main():
    print("Loading deterministic skill (weekly, era5)...")
    df = load_det()
    print(f"  rows: {len(df)}")

    for var in ['TP', 'Z500']:
        print(f"\n[{var}] Model × Week scatter grid...")
        fig_model_by_week(df, var)

        print(f"[{var}] Model comparison W1 vs W2...")
        fig_compare_weeks(df, var, week_pair=(1, 2))

        print(f"[{var}] Model comparison W2 vs W3...")
        fig_compare_weeks(df, var, week_pair=(2, 3))

    print("\nDone.")


if __name__ == '__main__':
    main()
