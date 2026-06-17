#!/usr/bin/env python3
"""
plot_summary.py
High-impact MEETING summary figures — the headline story on one slide.

Outputs (plots_results_pres/):
  SUMMARY_headline.png    — 4-panel: TP-PCC, Z500-PCC, TP-CRPSS, Z500-CRPSS
                            with the key takeaways annotated
  SUMMARY_scorecard.png   — Master scorecard: every model × week, PCC+CRPSS,
                            both variables, on one figure
  SUMMARY_skill_horizon.png — "Useful skill horizon" bar chart: how many weeks
                            each model beats PCC=0.5 / CRPSS=0
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
                    WEEKS, WLABELS, VAR_UNITS, VAR_LONG, style_week_axis,
                    model_legend_handles)

set_style()

DET_MODELS  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
PROB_MODELS = ['SPIRE', 'FuXi', 'ECMWF']


def load():
    det = pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))
    det = det[(det.scale == 'weekly') & (det.region == 'All India') &
              (det.clim_basis == 'era5')]
    prob = pd.read_csv(os.path.join(DATADIR, 'skill_probabilistic.csv'))
    prob = prob[(prob.scale == 'weekly') & (prob.region == 'All India') &
                (prob.clim_basis == 'era5')]
    return det, prob


def _series(df, var, model, metric):
    s = df[(df.variable == var) & (df.model == model)]
    return np.array([s[s.lead == w][metric].mean() for w in WEEKS])


# ══════════════════════════════════════════════════════════════════════════════
def fig_headline(det, prob):
    """The one-slide story: deterministic + probabilistic skill, both vars."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle('SPIRE S2S Hindcast Skill — JFM 2026, All India\n'
                 'Deterministic (top) and Probabilistic (bottom) vs ERA5',
                 fontsize=14, fontweight='bold', y=0.99)

    panels = [
        (axes[0, 0], det,  'TP',   'pcc',        'PCC',   (-0.15, 1.02), DET_MODELS,  0.5),
        (axes[0, 1], det,  'Z500', 'pcc',        'PCC',   (-0.35, 1.02), DET_MODELS,  0.5),
        (axes[1, 0], prob, 'TP',   'crpss_clim', 'CRPSS', (-0.25, 0.75), PROB_MODELS, 0.0),
        (axes[1, 1], prob, 'Z500', 'crpss_clim', 'CRPSS', (-0.45, 1.02), PROB_MODELS, 0.0),
    ]

    for ax, src, var, metric, ylab, ylim, mods, thresh in panels:
        for m in mods:
            if m not in MODEL_COLORS:
                continue
            vals = _series(src, var, m, metric)
            if np.all(np.isnan(vals)):
                continue
            lw = 1.6 if m == 'Persistence' else 2.6
            ls = '--' if m == 'Persistence' else '-'
            ms = 5 if m == 'Persistence' else 8
            ax.plot(WEEKS, vals, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                    lw=lw, ms=ms, ls=ls, label=m, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.6)

        # Skill threshold line
        ax.axhline(thresh, color='darkorange', lw=1.2, ls=':', alpha=0.85, zorder=1)
        tlabel = 'PCC = 0.5 (useful)' if metric == 'pcc' else 'CRPSS = 0 (= clim)'
        ax.text(6.05, thresh, tlabel, fontsize=8, color='darkorange',
                va='center', ha='left')

        style_week_axis(ax, ylabel=ylab, ylim=ylim,
                        title=f'{VAR_LONG[var]}  ({VAR_UNITS[var]})')

    # One shared legend
    handles = model_legend_handles(DET_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.tight_layout(rect=[0, 0.03, 0.93, 0.96])
    savefig(fig, 'SUMMARY_headline.png')


# ══════════════════════════════════════════════════════════════════════════════
def fig_scorecard(det, prob):
    """Master scorecard: rows = model, cols = week, blocks = metric × variable."""
    mods = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    blocks = [
        ('TP',   'pcc',        det,  'PCC',   'RdYlGn', (-0.2, 1.0)),
        ('Z500', 'pcc',        det,  'PCC',   'RdYlGn', (-0.2, 1.0)),
        ('TP',   'crpss_clim', prob, 'CRPSS', 'RdYlGn', (-0.3, 0.7)),
        ('Z500', 'crpss_clim', prob, 'CRPSS', 'RdYlGn', (-0.3, 1.0)),
    ]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(9, 11))
    fig.suptitle('Skill Scorecard — All India, JFM 2026 (vs ERA5)\n'
                 'Green = skilful · Red = no skill',
                 fontsize=13, fontweight='bold', y=1.0)

    for ax, (var, metric, src, mlab, cmap, vlim) in zip(axes, blocks):
        data = np.array([_series(src, var, m, metric) for m in mods])
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods, fontsize=10)
        ax.set_xticks(range(6)); ax.set_xticklabels(WLABELS, fontsize=10)
        ax.set_title(f'{var}  —  {mlab}', fontsize=11, fontweight='bold', pad=4)
        for i in range(len(mods)):
            for j in range(6):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=9,
                            color='black' if abs(v) < 0.6 else 'white',
                            fontweight='bold')
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, 'SUMMARY_scorecard.png')


# ══════════════════════════════════════════════════════════════════════════════
def fig_skill_horizon(det, prob):
    """Bar chart: # of weeks each model stays skilful (PCC>=0.5 / CRPSS>=0)."""
    mods = ['SPIRE', 'FuXi', 'ECMWF', 'MME']

    def last_skilful_week(vals, thresh):
        """Highest contiguous week index (1-based) staying >= thresh from W1."""
        n = 0
        for v in vals:
            if np.isnan(v) or v < thresh:
                break
            n += 1
        return n

    metrics = [
        ('TP PCC≥0.5',     det,  'TP',   'pcc',        0.5),
        ('Z500 PCC≥0.5',   det,  'Z500', 'pcc',        0.5),
        ('TP CRPSS≥0',     prob, 'TP',   'crpss_clim', 0.0),
        ('Z500 CRPSS≥0',   prob, 'Z500', 'crpss_clim', 0.0),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    width = 0.2

    for mi, m in enumerate(mods):
        horizons = []
        for _, src, var, metric, thr in metrics:
            vals = _series(src, var, m, metric)
            horizons.append(last_skilful_week(vals, thr))
        bars = ax.bar(x + (mi - 1.5) * width, horizons, width,
                      color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.8)
        for b, h in zip(bars, horizons):
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.08, f'{h}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color=MODEL_COLORS[m])

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11)
    ax.set_ylabel('Useful skill horizon  (weeks from W1)', fontsize=12)
    ax.set_ylim(0, 6.6)
    ax.set_yticks(range(7))
    ax.set_title('Forecast Skill Horizon by Model — All India, JFM 2026\n'
                 'Contiguous weeks from W1 staying above the skill threshold',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              frameon=True)
    fig.tight_layout()
    savefig(fig, 'SUMMARY_skill_horizon.png')


def main():
    print("Loading skill tables (weekly, All India, era5)...")
    det, prob = load()
    print(f"  det rows: {len(det)}  |  prob rows: {len(prob)}")

    print("\n[1/3] Headline 4-panel story...")
    fig_headline(det, prob)
    print("[2/3] Master scorecard...")
    fig_scorecard(det, prob)
    print("[3/3] Skill horizon bars...")
    fig_skill_horizon(det, prob)
    print("\nDone.")


if __name__ == '__main__':
    main()
