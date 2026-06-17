#!/usr/bin/env python3
"""
make_simple.py — THREE simple figures: ACC, RMSE, Bias.
All India · weekly · ERA5 basis · models SPIRE/FuXi/ECMWF/MME.
One line per model, value vs forecast week. Nothing else.

Output (this folder):  acc.png  rmse.png  bias.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(os.path.dirname(HERE), 'skill_deterministic.csv')

MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
COLORS = {'SPIRE': '#1a6faf', 'FuXi': '#e05c2a', 'ECMWF': '#2a9d54', 'MME': '#9b59b6'}
MARKERS = {'SPIRE': 'o', 'FuXi': 's', 'ECMWF': '^', 'MME': 'D'}
WEEKS  = [1, 2, 3, 4, 5, 6]
WLABELS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VARS = {'TP': 'Total Precipitation (mm/day)', 'Z500': 'Z500 (gpm)'}

plt.rcParams.update({'font.size': 13, 'axes.spines.top': False,
                     'axes.spines.right': False})


def load():
    df = pd.read_csv(CSV)
    return df[(df.scale == 'weekly') & (df.region == 'All India') &
              (df.clim_basis == 'era5')]


def series(df, var, model, metric):
    s = df[(df.variable == var) & (df.model == model)]
    return [s[s.lead == w][metric].mean() for w in WEEKS]


def make(df, metric, ylabel, title, fname, hline=None, ndigits=2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, ['TP', 'Z500']):
        for m in MODELS:
            ax.plot(WEEKS, series(df, var, m, metric), color=COLORS[m],
                    marker=MARKERS[m], lw=2.5, ms=9, label=m)
        if hline is not None:
            ax.axhline(hline, color='gray', ls='--', lw=1.3)
        ax.set_title(VARS[var], fontsize=13)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WLABELS)
        ax.set_xlabel('Forecast Week')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axes[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(1.05, -0.13),
                   frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    df = load()
    make(df, 'pcc',  'ACC',  'ACC (Anomaly Correlation) — All India', 'acc.png',
         hline=0.5)
    make(df, 'rmse', 'RMSE', 'RMSE — All India', 'rmse.png')
    make(df, 'bias', 'Bias', 'Mean Bias — All India', 'bias.png', hline=0.0)
    print('Done.')


if __name__ == '__main__':
    main()
