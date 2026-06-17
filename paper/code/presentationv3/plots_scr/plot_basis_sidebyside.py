#!/usr/bin/env python3
"""
plot_basis_sidebyside.py
Side-by-side PCC: Model-own climatology basis (LEFT) vs ERA5 climatology basis (RIGHT).

Only FuXi, ECMWF, MME have a model-own hindcast climatology, so those are the
three models shown in BOTH panels (a fair like-for-like comparison).
SPIRE/Persistence are excluded — they exist only on the ERA5 basis.

The gap between a model's LEFT and RIGHT line = how much its systematic
climatological bias inflates (or deflates) the apparent ERA5-basis skill.

Outputs (plots_results_pres/):
  basis_sidebyside_pcc.png   — 2 rows (TP, Z500) × 2 cols (model-own | ERA5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    WEEKS, VAR_UNITS, VAR_LONG, style_week_axis)

set_style()

# Models that have BOTH bases (the only fair comparison set)
MODELS = ['FuXi', 'ECMWF', 'MME']

BASIS_TITLE = {
    'model_own': 'Model-own climatology',
    'era5':      'ERA5 climatology',
}
BASIS_ORDER = ['model_own', 'era5']     # left, right


def load():
    d = pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))
    return d[(d.scale == 'weekly') & (d.region == 'All India')]


def _series(df, var, model, basis):
    s = df[(df.variable == var) & (df.model == model) & (df.clim_basis == basis)]
    return np.array([s[s.lead == w]['pcc'].mean() for w in WEEKS])


def main():
    print("Loading deterministic skill (weekly, All India)...")
    df = load()
    print(f"  rows: {len(df)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharey='row', sharex=True)
    fig.suptitle('Pattern Correlation (PCC) — Climatology-Basis Comparison\n'
                 'All India, JFM 2026  |  FuXi · ECMWF · MME  '
                 '(only models with a model-own hindcast climatology)',
                 fontsize=13, fontweight='bold', y=1.0)

    ylims = {'TP': (-0.15, 1.02), 'Z500': (-0.4, 1.02)}

    for ri, var in enumerate(['TP', 'Z500']):
        for ci, basis in enumerate(BASIS_ORDER):
            ax = axes[ri, ci]
            for m in MODELS:
                vals = _series(df, var, m, basis)
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(WEEKS, vals, color=MODEL_COLORS[m],
                        marker=MODEL_MARKERS[m], lw=2.6, ms=8, label=m,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=3)

            ax.axhline(0.5, color='darkorange', lw=1.2, ls=':', alpha=0.85, zorder=1)

            title = (f'{VAR_LONG[var]}  ({VAR_UNITS[var]})\n{BASIS_TITLE[basis]}'
                     if ri == 0 else BASIS_TITLE[basis])
            style_week_axis(ax, ylabel='PCC' if ci == 0 else '',
                            ylim=ylims[var], title=title)
            if ri == 0:                      # x-axis shared; label only bottom row
                ax.set_xlabel('')

        # Variable label on the far left
        axes[ri, 0].text(-0.18, 0.5, var, transform=axes[ri, 0].transAxes,
                         rotation=90, va='center', ha='center',
                         fontsize=13, fontweight='bold')

    # Shared legend
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                          lw=2.6, ms=8, label=m) for m in MODELS]
    handles += [plt.Line2D([0], [0], color='darkorange', lw=1.2, ls=':',
                           label='PCC = 0.5 (useful skill)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    savefig(fig, 'basis_sidebyside_pcc.png')
    print("\nDone.")


if __name__ == '__main__':
    main()
