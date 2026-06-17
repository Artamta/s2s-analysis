#!/usr/bin/env python3
"""
plot_bars_allindia.py
Simple, slide-readable BAR CHARTS — All India + 4 IMD homogeneous regions.

Design for presentation clarity:
  - 4 real models only (SPIRE, FuXi, ECMWF, MME). Persistence is a baseline,
    not a model, so it is dropped from the bars (it lives in the line plots).
  - Horizontal value labels, large fonts, generous spacing.
  - One STANDALONE All-India figure per metric (cleanest, for the title slide).
  - One ALL-REGIONS figure per metric (2x3 panels: All India + 4 sub-regions).

Outputs (plots_results_pres/):
  All-India standalone:
    bar_pcc_{TP,Z500}.png   bar_rmse_{TP,Z500}.png   bar_bias_{TP,Z500}.png
    bar_msss_{TP,Z500}.png  bar_crpss_{TP,Z500}.png
  All regions (2x3 panels):
    bar_regions_pcc_{TP,Z500}.png   bar_regions_rmse_{TP,Z500}.png
    bar_regions_bias_{TP,Z500}.png  bar_regions_msss_{TP,Z500}.png
    bar_regions_crpss_{TP,Z500}.png
  Story slides:
    bar_W1_overview.png     bar_skill_horizon.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _style import (set_style, savefig, DATADIR, MODEL_COLORS, MODEL_MARKERS,
                    WEEKS, WLABELS, VAR_UNITS, VAR_LONG, REGIONS, REGION_LABEL)

set_style()

# Real models only — Persistence dropped for clarity
DET_MODELS  = ['SPIRE', 'FuXi', 'ECMWF', 'MME']
PROB_MODELS = ['SPIRE', 'FuXi', 'ECMWF']


def load():
    det = pd.read_csv(os.path.join(DATADIR, 'skill_deterministic.csv'))
    det = det[(det.scale == 'weekly') & (det.clim_basis == 'era5')]
    prob = pd.read_csv(os.path.join(DATADIR, 'skill_probabilistic.csv'))
    prob = prob[(prob.scale == 'weekly') & (prob.clim_basis == 'era5')]
    return det, prob


def _matrix(df, var, region, models, metric):
    out = np.full((len(models), len(WEEKS)), np.nan)
    for i, m in enumerate(models):
        s = df[(df.variable == var) & (df.model == m) & (df.region == region)]
        for j, w in enumerate(WEEKS):
            out[i, j] = s[s.lead == w][metric].mean()
    return out


def _draw_bars(ax, data, models, fmt='{:.2f}', annotate=True, label_fs=8):
    """Grouped bars: x-groups = weeks, bars = models. Horizontal labels."""
    n = len(models)
    x = np.arange(len(WEEKS))
    width = 0.8 / n
    lo, hi = ax.get_ylim()
    span = (hi - lo) if hi > lo else 1.0
    for i, m in enumerate(models):
        off = (i - (n - 1) / 2) * width
        bars = ax.bar(x + off, data[i], width, color=MODEL_COLORS[m],
                      label=m, edgecolor='white', linewidth=0.7, zorder=3)
        if annotate:
            for j, b in enumerate(bars):
                v = data[i, j]
                if np.isnan(v):
                    continue
                va = 'bottom' if v >= 0 else 'top'
                yoff = 0.015 * span if v >= 0 else -0.015 * span
                ax.text(b.get_x() + b.get_width() / 2, v + yoff, fmt.format(v),
                        ha='center', va=va, fontsize=label_fs, color='#222')
    ax.set_xticks(x)
    ax.set_xticklabels(WLABELS)
    ax.grid(axis='y', alpha=0.3, zorder=0)


# ── metric registry ────────────────────────────────────────────────────────────
METRICS = {
    'pcc':   dict(src='det',  ylabel='PCC',  short='PCC',
                  hline=0.5, hline_label='useful (0.5)', ylim=(-0.45, 1.08),
                  fmt='{:.2f}', title='PCC'),
    'rmse':  dict(src='det',  ylabel='RMSE', short='RMSE',
                  hline=None, hline_label='', ylim=None,
                  fmt='{:.1f}', title='RMSE'),
    # NOTE: bias is plotted as LINES (drift over lead), not bars — see
    # bias_lines_allindia / bias_lines_regions. Kept out of the bar registry.
    'msss_clim': dict(src='det', ylabel='MSSS', short='MSSS',
                  hline=0.0, hline_label='= climatology', ylim=(-1.0, 1.08),
                  fmt='{:+.2f}', title='MSSS vs Climatology'),
    'crpss_clim': dict(src='prob', ylabel='CRPSS', short='CRPSS',
                  hline=0.0, hline_label='= climatology', ylim=(-0.5, 0.85),
                  fmt='{:+.2f}', title='CRPSS vs Climatology'),
}


def _unit_label(meta, var):
    if meta['short'] in ('RMSE', 'Bias'):
        return f"{meta['ylabel']} ({VAR_UNITS[var]})"
    return meta['ylabel']


# ── ALL-INDIA standalone ───────────────────────────────────────────────────────
def bar_allindia(det, prob, metric):
    meta = METRICS[metric]
    src = det if meta['src'] == 'det' else prob
    models = DET_MODELS if meta['src'] == 'det' else PROB_MODELS
    for var in ['TP', 'Z500']:
        data = _matrix(src, var, 'All India', models, metric)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(f'{meta["title"]} — {VAR_LONG[var]} ({VAR_UNITS[var]})\n'
                     f'All India · JFM 2026 · vs ERA5',
                     fontsize=14, fontweight='bold', y=1.0)
        if meta['ylim']:
            ax.set_ylim(meta['ylim'])
        else:
            _autoylim(ax, data)
        _draw_bars(ax, data, models, fmt=meta['fmt'], label_fs=9)
        if meta['hline'] is not None:
            ax.axhline(meta['hline'], color='darkorange', lw=1.6, ls='--',
                       alpha=0.9, zorder=2)
            ax.text(len(WEEKS) - 0.45, meta['hline'], meta['hline_label'],
                    fontsize=10, color='darkorange', va='bottom', ha='right')
        ax.axhline(0, color='#333', lw=1.0, zorder=2)
        ax.set_ylabel(_unit_label(meta, var), fontsize=13)
        ax.set_xlabel('Forecast Week', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(ncol=len(models), fontsize=11, loc='upper center',
                  bbox_to_anchor=(0.5, -0.11), frameon=True)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        savefig(fig, f'bar_{metric.replace("_clim","")}_{var}.png')


def _autoylim(ax, data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    lo, hi = finite.min(), finite.max()
    pad = (hi - lo) * 0.18 + 1e-9
    ax.set_ylim(min(0, lo - pad), max(0, hi + pad * 1.4))


# ── BIAS as LINE plots (drift over lead reads better than up/down bars) ─────────
def _bias_lines(ax, det, var, region, models):
    drew = False
    for m in models:
        vals = _matrix(det, var, region, [m], 'bias')[0]
        if np.all(np.isnan(vals)):
            continue
        ax.plot(WEEKS, vals, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                lw=2.4, ms=8, label=m, markeredgecolor='white',
                markeredgewidth=0.7, zorder=3)
        drew = True
    ax.axhline(0, color='#333', lw=1.2, zorder=2)
    ax.set_xticks(WEEKS); ax.set_xticklabels(WLABELS)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    return drew


def bias_lines_allindia(det):
    for var in ['TP', 'Z500']:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(f'Mean Bias Drift — {VAR_LONG[var]} ({VAR_UNITS[var]})\n'
                     f'All India · JFM 2026 · vs ERA5  '
                     f'(above 0 = too high/warm, below 0 = too low/cold)',
                     fontsize=13, fontweight='bold', y=1.0)
        data = _matrix(det, var, 'All India', DET_MODELS, 'bias')
        _autoylim(ax, data)
        _bias_lines(ax, det, var, 'All India', DET_MODELS)
        ax.set_ylabel(f'Bias ({VAR_UNITS[var]})', fontsize=13)
        ax.set_xlabel('Forecast Week', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(ncol=len(DET_MODELS), fontsize=11, loc='upper center',
                  bbox_to_anchor=(0.5, -0.11), frameon=True)
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        savefig(fig, f'bias_lines_{var}.png')


def bias_lines_regions(det):
    for var in ['TP', 'Z500']:
        all_data = [_matrix(det, var, r, DET_MODELS, 'bias') for r in REGIONS]
        stacked = np.concatenate([d[np.isfinite(d)] for d in all_data])
        lo, hi = stacked.min(), stacked.max()
        pad = (hi - lo) * 0.12 + 1e-9
        ylim = (lo - pad, hi + pad)

        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
        fig.suptitle(f'Mean Bias Drift — {VAR_LONG[var]} ({VAR_UNITS[var]})\n'
                     f'All India + IMD Homogeneous Regions · JFM 2026 · vs ERA5',
                     fontsize=15, fontweight='bold', y=1.0)
        axes_flat = axes.flatten()
        for pi, reg in enumerate(REGIONS):
            ax = axes_flat[pi]
            ax.set_ylim(ylim)
            _bias_lines(ax, det, var, reg, DET_MODELS)
            ax.set_title(REGION_LABEL[reg], fontsize=13, fontweight='bold', pad=4)
            if pi % 3 == 0:
                ax.set_ylabel(f'Bias ({VAR_UNITS[var]})', fontsize=12)
            if pi >= 3:
                ax.set_xlabel('Forecast Week', fontsize=12)
            ax.tick_params(labelsize=10)
        axes_flat[5].axis('off')
        handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m],
                              marker=MODEL_MARKERS[m], lw=2.4, ms=9, label=m)
                   for m in DET_MODELS]
        handles.append(plt.Line2D([0], [0], color='#333', lw=1.2, label='zero bias'))
        axes_flat[5].legend(handles, [m for m in DET_MODELS] + ['zero bias'],
                            loc='center', fontsize=14, frameon=True,
                            title='Models', title_fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        savefig(fig, f'bias_lines_regions_{var}.png')


# ── ALL REGIONS (2x3 panels) ───────────────────────────────────────────────────
def bar_regions(det, prob, metric):
    meta = METRICS[metric]
    src = det if meta['src'] == 'det' else prob
    models = DET_MODELS if meta['src'] == 'det' else PROB_MODELS
    panel_regions = REGIONS                     # All India + 4 sub-regions
    for var in ['TP', 'Z500']:
        # shared y-limits across panels for fair visual comparison
        all_data = [_matrix(src, var, r, models, metric) for r in panel_regions]
        if meta['ylim']:
            ylim = meta['ylim']
        else:
            stacked = np.concatenate([d[np.isfinite(d)] for d in all_data])
            lo, hi = stacked.min(), stacked.max()
            pad = (hi - lo) * 0.18 + 1e-9
            ylim = (min(0, lo - pad), max(0, hi + pad * 1.4))

        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
        fig.suptitle(f'{meta["title"]} — {VAR_LONG[var]} ({VAR_UNITS[var]})\n'
                     f'All India + IMD Homogeneous Regions · JFM 2026 · vs ERA5',
                     fontsize=15, fontweight='bold', y=1.0)
        axes_flat = axes.flatten()
        for pi, reg in enumerate(panel_regions):
            ax = axes_flat[pi]
            ax.set_ylim(ylim)
            _draw_bars(ax, all_data[pi], models, fmt=meta['fmt'], label_fs=7)
            if meta['hline'] is not None:
                ax.axhline(meta['hline'], color='darkorange', lw=1.4, ls='--',
                           alpha=0.9, zorder=2)
            ax.axhline(0, color='#333', lw=0.9, zorder=2)
            ax.set_title(REGION_LABEL[reg], fontsize=13, fontweight='bold', pad=4)
            if pi % 3 == 0:
                ax.set_ylabel(_unit_label(meta, var), fontsize=12)
            if pi >= 3:
                ax.set_xlabel('Forecast Week', fontsize=12)
            ax.tick_params(labelsize=10)
        # 6th panel = legend
        axes_flat[5].axis('off')
        handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models]
        labels = list(models)
        if meta['hline'] is not None:
            handles.append(plt.Line2D([0], [0], color='darkorange', lw=1.6, ls='--'))
            labels.append(meta['hline_label'].strip())
        axes_flat[5].legend(handles, labels, loc='center', fontsize=14,
                            frameon=True, title='Models', title_fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        savefig(fig, f'bar_regions_{metric.replace("_clim","")}_{var}.png')


# ── STORY SLIDES ───────────────────────────────────────────────────────────────
def make_week1_overview(det, prob):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Week-1 Skill Snapshot — All India, JFM 2026',
                 fontsize=14, fontweight='bold', y=1.0)
    vars_ = ['TP', 'Z500']
    x = np.arange(len(vars_))

    # PCC
    ax = axes[0]; models = DET_MODELS; w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [det[(det.variable == v) & (det.model == m) &
                    (det.region == 'All India') & (det.lead == 1)]['pcc'].mean()
                for v in vars_]
        off = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.7, zorder=3)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(0.5, color='darkorange', lw=1.6, ls='--', alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=12)
    ax.set_ylim(0, 1.08); ax.set_ylabel('PCC', fontsize=13)
    ax.set_title('Deterministic — PCC', fontsize=13); ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10, ncol=2)

    # CRPSS
    ax = axes[1]; models = PROB_MODELS; w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [prob[(prob.variable == v) & (prob.model == m) &
                     (prob.region == 'All India') & (prob.lead == 1)]['crpss_clim'].mean()
                for v in vars_]
        off = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.7, zorder=3)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(0.0, color='#333', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=12)
    ax.set_ylim(0, 0.85); ax.set_ylabel('CRPSS', fontsize=13)
    ax.set_title('Probabilistic — CRPSS', fontsize=13); ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10, ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, 'bar_W1_overview.png')


def make_skill_horizon(det, prob):
    def horizon(vals, thresh):
        n = 0
        for v in vals:
            if np.isnan(v) or v < thresh:
                break
            n += 1
        return n
    models = DET_MODELS
    metrics = [
        ('TP PCC≥0.5',   det,  'TP',   'pcc',        0.5),
        ('Z500 PCC≥0.5', det,  'Z500', 'pcc',        0.5),
        ('TP CRPSS≥0',   prob, 'TP',   'crpss_clim', 0.0),
        ('Z500 CRPSS≥0', prob, 'Z500', 'crpss_clim', 0.0),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics)); width = 0.2
    for mi, m in enumerate(models):
        hs = []
        for _, src, var, metric, thr in metrics:
            s = src[(src.variable == var) & (src.model == m) &
                    (src.region == 'All India')]
            vals = np.array([s[s.lead == w][metric].mean() for w in WEEKS])
            hs.append(horizon(vals, thr))
        off = (mi - 1.5) * width
        bars = ax.bar(x + off, hs, width, color=MODEL_COLORS[m], label=m,
                      edgecolor='white', linewidth=0.8, zorder=3)
        for b, h in zip(bars, hs):
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.08, str(h),
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        color=MODEL_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([mm[0] for mm in metrics], fontsize=12)
    ax.set_ylim(0, 6.6); ax.set_yticks(range(7))
    ax.set_ylabel('Useful skill horizon (weeks from W1)', fontsize=13)
    ax.set_title('Forecast Skill Horizon by Model — All India, JFM 2026',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=4, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, 'bar_skill_horizon.png')


def main():
    print("Loading skill tables (weekly, era5)...")
    det, prob = load()
    print(f"  det rows: {len(det)}  |  prob rows: {len(prob)}")

    print("\nAll-India standalone bar charts...")
    for metric in METRICS:
        bar_allindia(det, prob, metric)

    print("\nAll-regions (2x3) bar charts...")
    for metric in METRICS:
        bar_regions(det, prob, metric)

    print("\nBias drift line plots (All India + regions)...")
    bias_lines_allindia(det)
    bias_lines_regions(det)

    print("\nStory slides...")
    make_week1_overview(det, prob)
    make_skill_horizon(det, prob)

    print("\nDone.")


if __name__ == '__main__':
    main()
