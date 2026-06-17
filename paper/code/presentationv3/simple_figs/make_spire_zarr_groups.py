#!/usr/bin/env python3
"""
make_spire_zarr_groups.py — SPIRE scored from the two zarr groups, overlaid.
Reads the precomputed spire_diag/spire_paths.csv (no slow zarr reload).

  mean_stddev group : absolute ensemble mean, minus our ERA5 climatology
  anomalies group   : SPIRE's own pre-computed anomaly value

Output (this folder):  spire_acc.png  spire_rmse.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(os.path.dirname(HERE), 'spire_diag', 'spire_paths.csv')

WEEKS = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6']
WLAB  = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
VARS  = ['TP', 'Z500']
VARLONG = {'TP': 'Total Precipitation (mm/day)', 'Z500': 'Z500 (gpm)'}

LAB_STD  = 'mean_stddev group (absolute mean)'
LAB_ANOM = 'anomalies group (SPIRE anomaly value)'
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False,
                     'axes.spines.right': False})


def series(df, var, col):
    s = df[df['var'] == var]
    return [s[s.week == wn][col].mean() for wn in WEEKS]


def make(df, colA, colB, ylabel, title, fname, hline=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, var in zip(axes, VARS):
        ax.plot(range(1, 7), series(df, var, colA), 'o-', color='#1a6faf',
                lw=2.5, ms=9, label=LAB_STD)
        ax.plot(range(1, 7), series(df, var, colB), 's--', color='#e05c2a',
                lw=2.5, ms=8, label=LAB_ANOM)
        if hline is not None:
            ax.axhline(hline, color='gray', ls=':', lw=1.2)
        ax.set_title(VARLONG[var]); ax.set_xticks(range(1, 7))
        ax.set_xticklabels(WLAB); ax.set_xlabel('Forecast Week')
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    if not os.path.isfile(CSV):
        raise SystemExit(f"missing {CSV} — run spire_diag/spire_meanstd_vs_anom.py first")
    df = pd.read_csv(CSV)
    make(df, 'accA', 'accB', 'ACC',
         'SPIRE ACC — mean_stddev vs anomalies group (≈ identical for TP)',
         'spire_acc.png', hline=0.5)
    make(df, 'rmseA', 'rmseB', 'RMSE',
         'SPIRE RMSE — mean_stddev vs anomalies group '
         '(anomalies path inflated)', 'spire_rmse.png')
    print('Done.')


if __name__ == '__main__':
    main()
