#!/usr/bin/env python3
"""
plot_bars_allindia.py
All-India BAR CHARTS for presentation — grouped bars (model × week), one
clean figure per metric.  Bars are the most legible format for a slide.

Outputs (plots_results_pres/):
  bar_pcc_TP.png            — PCC by week, grouped by model, TP
  bar_pcc_Z500.png          — PCC by week, Z500
  bar_rmse_TP.png           — RMSE by week, TP
  bar_rmse_Z500.png         — RMSE by week, Z500
  bar_bias_TP.png           — Bias by week (diverging), TP
  bar_bias_Z500.png         — Bias by week (diverging), Z500
  bar_msss_TP.png           — MSSS vs climatology, TP
  bar_msss_Z500.png         — MSSS vs climatology, Z500
  bar_crpss_TP.png          — CRPSS by week, TP
  bar_crpss_Z500.png        — CRPSS by week, Z500
  bar_W1_overview.png       — Week-1 snapshot: PCC/RMSE/CRPSS across all metrics
  bar_skill_horizon.png     — Useful-skill horizon (weeks) per model
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    WEEKS, WLABELS, VAR_UNITS, VAR_LONG)

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


def _matrix(df, var, models, metric):
    """Return (n_models × 6weeks) matrix of weekly-mean metric values."""
    out = np.full((len(models), len(WEEKS)), np.nan)
    for i, m in enumerate(models):
        s = df[(df.variable == var) & (df.model == m)]
        for j, w in enumerate(WEEKS):
            out[i, j] = s[s.lead == w][metric].mean()
    return out


def _grouped_bars(ax, data, models, value_fmt='{:.2f}', annotate=True,
                  label_thresh=0.0, show_labels_for=None):
    """Grouped bar chart: groups = weeks (x), bars = models.
    show_labels_for: list of week-indices to annotate (None = all)."""
    n_mod = len(models)
    x = np.arange(len(WEEKS))
    width = 0.8 / n_mod
    for i, m in enumerate(models):
        offset = (i - (n_mod - 1) / 2) * width
        bars = ax.bar(x + offset, data[i], width,
                      color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.6, zorder=3)
        if annotate:
            for j, b in enumerate(bars):
                v = data[i, j]
                if np.isnan(v):
                    continue
                if show_labels_for is not None and j not in show_labels_for:
                    continue
                va = 'bottom' if v >= 0 else 'top'
                off = 0.012 if v >= 0 else -0.012
                ax.annotate(value_fmt.format(v),
                            (b.get_x() + b.get_width() / 2, v + off * abs(_yrange(ax))),
                            ha='center', va=va, fontsize=6.5, rotation=90,
                            color='#222')
    ax.set_xticks(x)
    ax.set_xticklabels(WLABELS, fontsize=10)


def _yrange(ax):
    lo, hi = ax.get_ylim()
    return hi - lo if hi > lo else 1.0


def bar_metric(det, prob, var, metric, source='det', ylabel='', title='',
               fname='', hline=None, hline_label='', ylim=None,
               diverging=False, fmt='{:.2f}'):
    src = det if source == 'det' else prob
    models = DET_MODELS if source == 'det' else PROB_MODELS
    # MSSS / CRPSS / Persistence handling: persistence has no prob metrics
    data = _matrix(src, var, models, metric)
    # Drop all-NaN model rows (e.g. persistence for prob)
    keep = [i for i in range(len(models)) if not np.all(np.isnan(data[i]))]
    models = [models[i] for i in keep]
    data = data[keep]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.0)

    if ylim:
        ax.set_ylim(ylim)
    _grouped_bars(ax, data, models, value_fmt=fmt)

    if hline is not None:
        ax.axhline(hline, color='darkorange', lw=1.4, ls='--', alpha=0.85, zorder=2)
        ax.text(len(WEEKS) - 0.45, hline, hline_label, fontsize=9,
                color='darkorange', va='bottom', ha='right')
    ax.axhline(0, color='#333', lw=0.9, zorder=2)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Forecast Week', fontsize=12)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.legend(ncol=len(models), fontsize=10, loc='upper center',
              bbox_to_anchor=(0.5, -0.10), frameon=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    savefig(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
def make_per_variable(det, prob):
    for var in ['TP', 'Z500']:
        unit = VAR_UNITS[var]
        long = VAR_LONG[var]

        bar_metric(det, prob, var, 'pcc', 'det',
                   ylabel='PCC',
                   title=f'PCC by Forecast Week — {long} ({unit}), All India',
                   fname=f'bar_pcc_{var}.png',
                   hline=0.5, hline_label='useful (0.5)',
                   ylim=(min(-0.4, 0), 1.05))

        bar_metric(det, prob, var, 'rmse', 'det',
                   ylabel=f'RMSE ({unit})',
                   title=f'RMSE by Forecast Week — {long} ({unit}), All India',
                   fname=f'bar_rmse_{var}.png',
                   fmt='{:.2f}')

        bar_metric(det, prob, var, 'bias', 'det',
                   ylabel=f'Bias ({unit})',
                   title=f'Mean Bias by Forecast Week — {long} ({unit}), All India',
                   fname=f'bar_bias_{var}.png',
                   diverging=True, fmt='{:+.2f}')

        bar_metric(det, prob, var, 'msss_clim', 'det',
                   ylabel='MSSS',
                   title=f'MSSS vs Climatology — {long} ({unit}), All India',
                   fname=f'bar_msss_{var}.png',
                   hline=0.0, hline_label='= climatology',
                   ylim=(-1.0, 1.05), fmt='{:+.2f}')

        bar_metric(det, prob, var, 'crpss_clim', 'prob',
                   ylabel='CRPSS',
                   title=f'CRPSS vs Climatology — {long} ({unit}), All India',
                   fname=f'bar_crpss_{var}.png',
                   hline=0.0, hline_label='= climatology',
                   ylim=(-0.5, 0.8), fmt='{:+.2f}')


# ══════════════════════════════════════════════════════════════════════════════
def make_week1_overview(det, prob):
    """Single slide: Week-1 PCC + CRPSS for both variables, grouped by model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Week-1 Skill Snapshot — All India, JFM 2026',
                 fontsize=13, fontweight='bold', y=1.0)

    # Panel A: PCC (det) for TP & Z500
    ax = axes[0]
    models = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    vars_ = ['TP', 'Z500']
    x = np.arange(len(vars_)); width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [det[(det.variable == v) & (det.model == m) & (det.lead == 1)]['pcc'].mean()
                for v in vars_]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.6, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2, v + 0.01),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(0.5, color='darkorange', lw=1.4, ls='--', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel('PCC', fontsize=12)
    ax.set_title('Deterministic — PCC', fontsize=12)
    ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=9, ncol=2)

    # Panel B: CRPSS (prob) for TP & Z500
    ax = axes[1]
    models = ['SPIRE', 'FuXi', 'ECMWF']
    for i, m in enumerate(models):
        vals = [prob[(prob.variable == v) & (prob.model == m) & (prob.lead == 1)]['crpss_clim'].mean()
                for v in vars_]
        offset = (i - (len(models) - 1) / 2) * (0.8 / len(models))
        bars = ax.bar(x + offset, vals, 0.8 / len(models), color=MODEL_COLORS[m],
                      label=m, edgecolor='white', linewidth=0.6, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2, v + 0.01),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(0.0, color='#333', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=11)
    ax.set_ylim(0, 0.85); ax.set_ylabel('CRPSS', fontsize=12)
    ax.set_title('Probabilistic — CRPSS', fontsize=12)
    ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=9, ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'bar_W1_overview.png')


# ══════════════════════════════════════════════════════════════════════════════
def make_skill_horizon(det, prob):
    """Bar chart: contiguous weeks from W1 a model stays skilful."""
    def horizon(vals, thresh):
        n = 0
        for v in vals:
            if np.isnan(v) or v < thresh:
                break
            n += 1
        return n

    models = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
    metrics = [
        ('TP\nPCC≥0.5',   det,  'TP',   'pcc',        0.5),
        ('Z500\nPCC≥0.5', det,  'Z500', 'pcc',        0.5),
        ('TP\nCRPSS≥0',   prob, 'TP',   'crpss_clim', 0.0),
        ('Z500\nCRPSS≥0', prob, 'Z500', 'crpss_clim', 0.0),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics)); width = 0.2
    for mi, m in enumerate(models):
        hs = []
        for _, src, var, metric, thr in metrics:
            s = src[(src.variable == var) & (src.model == m)]
            vals = np.array([s[s.lead == w][metric].mean() for w in WEEKS])
            hs.append(horizon(vals, thr))
        offset = (mi - 1.5) * width
        bars = ax.bar(x + offset, hs, width, color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.8, zorder=3)
        for b, h in zip(bars, hs):
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.08, str(h),
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color=MODEL_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=11)
    ax.set_ylim(0, 6.6); ax.set_yticks(range(7))
    ax.set_ylabel('Useful skill horizon (weeks from W1)', fontsize=12)
    ax.set_title('Forecast Skill Horizon by Model — All India, JFM 2026',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=4, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, 'bar_skill_horizon.png')


def main():
    print("Loading skill tables (weekly, All India, era5)...")
    det, prob = load()
    print(f"  det rows: {len(det)}  |  prob rows: {len(prob)}")

    print("\nPer-variable bar charts (PCC/RMSE/Bias/MSSS/CRPSS)...")
    make_per_variable(det, prob)

    print("Week-1 overview...")
    make_week1_overview(det, prob)

    print("Skill horizon...")
    make_skill_horizon(det, prob)

    print("\nDone.")


if __name__ == '__main__':
    main()
