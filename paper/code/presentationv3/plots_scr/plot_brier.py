#!/usr/bin/env python3
"""
plot_brier.py
Brier Skill Score (BSS) for categorical events vs forecast week.

Events: tp_above_normal, tp_below_normal, tp_gt_1mm, tp_gt_10mm,
        z500_above_normal, z500_below_normal

Outputs (plots_results_pres/):
  bss_allIndia_TP.png        — BSS for TP events, All India
  bss_allIndia_Z500.png      — BSS for Z500 events, All India
  bss_regional_above.png     — BSS above-normal across 5 regions
  bss_scorecard.png          — Heatmap: model × week for all events
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
                    PROB_MODELS, WEEKS, WLABELS, VAR_UNITS, VAR_LONG,
                    REGIONS, REGION_LABEL, style_week_axis, model_legend_handles)

set_style()

# Only models present in Brier CSV
BRIER_MODELS = ['SPIRE', 'FuXi', 'ECMWF']

TP_EVENTS   = ['tp_above_normal', 'tp_below_normal', 'tp_gt_1mm', 'tp_gt_10mm']
Z500_EVENTS = ['z500_above_normal', 'z500_below_normal']

EVENT_LABEL = {
    'tp_above_normal':   'TP > P66 (above normal)',
    'tp_below_normal':   'TP < P33 (below normal)',
    'tp_gt_1mm':         'TP > 1 mm day⁻¹',
    'tp_gt_10mm':        'TP > 10 mm day⁻¹',
    'z500_above_normal': 'Z500 > P66 (above normal)',
    'z500_below_normal': 'Z500 < P33 (below normal)',
}


def load_brier():
    df = pd.read_csv(os.path.join(DATADIR, 'skill_brier.csv'))
    return df[(df.scale == 'weekly') & (df.region == 'All India')]


def _bss(df, event, model, region='All India', weeks=WEEKS):
    sub = df[(df.event == event) & (df.model == model) & (df.region == region)]
    means = np.array([sub[sub.lead == w]['briss_clim'].mean() for w in weeks])
    stds  = np.array([sub[sub.lead == w]['briss_clim'].std()  for w in weeks])
    return means, stds


def fig_events(df, var, events):
    """One figure with len(events) panels for a variable."""
    ncol = 2
    nrow = (len(events) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.5, nrow * 4.2),
                             squeeze=False)
    fig.suptitle(f'{VAR_LONG[var]} — Brier Skill Score vs Climatology\n'
                 f'All India  |  ERA5 basis  |  Mean across 13 initialisations',
                 fontsize=12, y=1.01)

    for idx, event in enumerate(events):
        ax = axes[idx // ncol, idx % ncol]
        for m in BRIER_MODELS:
            means, stds = _bss(df, event, m)
            if np.all(np.isnan(means)):
                continue
            c = MODEL_COLORS[m]; mk = MODEL_MARKERS[m]
            ax.plot(WEEKS, means, color=c, marker=mk, lw=2.0, ms=7, label=m, zorder=3)
        style_week_axis(ax, ylabel='BSS', ylim=(-0.6, 1.1),
                        title=EVENT_LABEL.get(event, event))

    # Hide any unused panels
    for idx in range(len(events), nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    handles = model_legend_handles(BRIER_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout()
    savefig(fig, f'bss_allIndia_{var}.png')


def fig_regional_above(df_full):
    """BSS for above-normal events across all 5 regions."""
    events = [('TP', 'tp_above_normal'), ('Z500', 'z500_above_normal')]
    df_reg = pd.read_csv(os.path.join(DATADIR, 'skill_brier.csv'))
    df_reg = df_reg[df_reg.scale == 'weekly']

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    fig.suptitle('BSS (Above-Normal) by IMD Region  (ERA5 basis)',
                 fontsize=12, y=1.02)

    for ri, (var, event) in enumerate(events):
        for ci, reg in enumerate(REGIONS):
            ax = axes[ri, ci]
            for m in BRIER_MODELS:
                sub = df_reg[(df_reg.event == event) & (df_reg.model == m) &
                             (df_reg.region == reg)]
                means = np.array([sub[sub.lead == w]['briss_clim'].mean()
                                  for w in WEEKS])
                if np.all(np.isnan(means)):
                    continue
                ax.plot(WEEKS, means, color=MODEL_COLORS[m],
                        marker=MODEL_MARKERS[m], lw=1.8, ms=5, zorder=3)

            style_week_axis(ax, ylabel='BSS' if ci == 0 else '',
                            ylim=(-0.6, 1.1),
                            title=REGION_LABEL[reg] if ri == 0 else '')
            if ci == 0:
                ax.text(-0.42, 0.5, var, transform=ax.transAxes,
                        rotation=90, va='center', fontsize=11, fontweight='bold')

    handles = model_legend_handles(BRIER_MODELS)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout()
    savefig(fig, 'bss_regional_above.png')


def fig_scorecard(df_full):
    """Heatmap: model × week, one panel per event."""
    all_events = TP_EVENTS + Z500_EVENTS
    ncol = 3
    nrow = (len(all_events) + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 3.5),
                             squeeze=False)
    fig.suptitle('Brier Skill Score Scorecard — All India  (ERA5 basis)',
                 fontsize=13, y=1.02)

    for idx, event in enumerate(all_events):
        ax = axes[idx // ncol, idx % ncol]
        data = np.array([[_bss(df_full, event, m)[0][w - 1] for w in WEEKS]
                         for m in BRIER_MODELS])
        norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', norm=norm)

        ax.set_xticks(range(6)); ax.set_xticklabels(WLABELS, fontsize=8)
        ax.set_yticks(range(len(BRIER_MODELS))); ax.set_yticklabels(BRIER_MODELS, fontsize=8)
        ax.set_title(EVENT_LABEL.get(event, event), fontsize=9, pad=3)

        for i in range(len(BRIER_MODELS)):
            for j in range(len(WEEKS)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=7.5,
                            color='white' if abs(v) > 0.6 else 'black')

        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=7)

    for idx in range(len(all_events), nrow * ncol):
        axes[idx // ncol, idx % ncol].set_visible(False)

    fig.tight_layout()
    savefig(fig, 'bss_scorecard.png')


def main():
    print("Loading Brier skill (weekly)...")
    df_ai = load_brier()       # All India
    print(f"  All-India rows: {len(df_ai)}")

    print("\nTP event BSS...")
    fig_events(df_ai, 'TP', TP_EVENTS)

    print("\nZ500 event BSS...")
    fig_events(df_ai, 'Z500', Z500_EVENTS)

    print("\nRegional above-normal BSS...")
    fig_regional_above(df_ai)

    print("\nBSS scorecard...")
    fig_scorecard(df_ai)

    print("\nDone.")


if __name__ == '__main__':
    main()
