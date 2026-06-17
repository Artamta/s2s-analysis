#!/usr/bin/env python3
"""
make_mixed_basis.py — "fairest" mixed-basis comparison:
  SPIRE  scored vs ERA5 climatology       (it has no own hindcast clim)
  FuXi   scored vs ITS OWN climatology
  ECMWF  scored vs ITS OWN climatology
All India · weekly · ACC / RMSE / Bias.  One line per model.

Output (this folder):  mixed_acc.png  mixed_rmse.png  mixed_bias.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(os.path.dirname(HERE), 'skill_deterministic.csv')

# model -> climatology basis used for it
BASIS = {'SPIRE': 'era5', 'FuXi': 'model_own', 'ECMWF': 'model_own'}
COLORS = {'SPIRE': '#1a6faf', 'FuXi': '#e05c2a', 'ECMWF': '#2a9d54'}
MARKERS = {'SPIRE': 'o', 'FuXi': 's', 'ECMWF': '^'}
LABEL = {'SPIRE': 'SPIRE (ERA5 clim)', 'FuXi': 'FuXi (own clim)',
         'ECMWF': 'ECMWF (own clim)'}
WEEKS = [1, 2, 3, 4, 5, 6]
WLABELS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VARS = {'TP': 'Total Precipitation (mm/day)', 'Z500': 'Z500 (gpm)'}

plt.rcParams.update({'font.size': 13, 'axes.spines.top': False,
                     'axes.spines.right': False})


def load():
    return pd.read_csv(CSV).query("scale=='weekly' and region=='All India'")


def series(df, var, model, metric):
    s = df[(df.variable == var) & (df.model == model) &
           (df.clim_basis == BASIS[model])]
    return [s[s.lead == w][metric].mean() for w in WEEKS]


def make(df, metric, ylabel, title, fname, hline=None, useful='above'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, var in zip(axes, ['TP', 'Z500']):
        # shade the "useful skill" zone (ACC > 0.5)
        if hline is not None and useful == 'above':
            ax.axhspan(hline, 1.05, color='#2a9d54', alpha=0.06, zorder=0)
        for m in ['SPIRE', 'FuXi', 'ECMWF']:
            y = series(df, var, m, metric)
            lw = 3.4 if m == 'SPIRE' else 2.4          # emphasise SPIRE
            ax.plot(WEEKS, y, color=COLORS[m], marker=MARKERS[m],
                    lw=lw, ms=11, label=LABEL[m], zorder=4,
                    markeredgecolor='white', markeredgewidth=1.0)
        if hline is not None:
            ax.axhline(hline, color='#555', ls='--', lw=1.5, zorder=1)
            if useful == 'above':
                ax.text(6.05, hline, 'useful\nskill', fontsize=10, color='#555',
                        va='center', ha='left')
        ax.set_title(VARS[var], fontsize=15, fontweight='bold', pad=8)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WLABELS, fontsize=12)
        ax.set_xlabel('Forecast lead', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)
        ax.margins(x=0.03)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, -0.01), fontsize=12, frameon=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    df = load()
    t_acc = ('How well each model predicts the weather pattern (ACC) — All India\n'
             'Higher = better · SPIRE vs ERA5 climatology · FuXi/ECMWF vs their own climatology')
    t_rmse = ('Forecast error (RMSE) — All India   (lower = better)\n'
              'SPIRE vs ERA5 climatology · FuXi/ECMWF vs their own climatology')
    t_bias = ('Mean bias (forecast − observed) — All India   (closer to 0 = better)\n'
              'SPIRE vs ERA5 climatology · FuXi/ECMWF vs their own climatology')
    make(df, 'pcc',  'ACC',  t_acc,  'mixed_acc.png',  hline=0.5)
    make(df, 'rmse', 'RMSE', t_rmse, 'mixed_rmse.png')
    make(df, 'bias', 'Bias', t_bias, 'mixed_bias.png', hline=0.0, useful='none')
    print('Done.')


if __name__ == '__main__':
    main()
