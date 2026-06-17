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


def make(df, metric, ylabel, title, fname, hline=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, ['TP', 'Z500']):
        for m in ['SPIRE', 'FuXi', 'ECMWF']:
            ax.plot(WEEKS, series(df, var, m, metric), color=COLORS[m],
                    marker=MARKERS[m], lw=2.5, ms=9, label=LABEL[m])
        if hline is not None:
            ax.axhline(hline, color='gray', ls='--', lw=1.3)
        ax.set_title(VARS[var]); ax.set_xticks(WEEKS); ax.set_xticklabels(WLABELS)
        ax.set_xlabel('Forecast Week'); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(1.05, -0.13),
                   frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    df = load()
    sub = 'All India · SPIRE vs ERA5 clim · FuXi/ECMWF vs own clim'
    make(df, 'pcc',  'ACC',  f'ACC — {sub}', 'mixed_acc.png',  hline=0.5)
    make(df, 'rmse', 'RMSE', f'RMSE — {sub}', 'mixed_rmse.png')
    make(df, 'bias', 'Bias', f'Mean Bias — {sub}', 'mixed_bias.png', hline=0.0)
    print('Done.')


if __name__ == '__main__':
    main()
