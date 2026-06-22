#!/usr/bin/env python3
"""
make_plots.py  —  Skill figures for the V2 (model-own climatology) verification.

Regions: All India + 4 IMD homogeneous regions only.
Reads:   skill_deterministic.csv, skill_probabilistic.csv  (same directory)
Writes:  plots/  (created if absent)

Figures
-------
  fig1_pcc_weekly_allIndia.png   — PCC & CRPSS weekly, All India, TP + Z500
  fig2_pcc_regional_TP.png       — TP PCC weekly, all 5 regions, 1 panel per region
  fig3_pcc_regional_Z500.png     — Z500 PCC weekly, all 5 regions
  fig4_rmse_bias_ssr.png         — RMSE / Bias / SSR weekly, All India
  fig5_msss_crpss_regional.png   — MSSS + CRPSS weekly, all 5 regions, TP + Z500
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE  = os.path.dirname(os.path.abspath(__file__))
ODIR  = os.path.join(HERE, 'plots')
os.makedirs(ODIR, exist_ok=True)

det  = pd.read_csv(os.path.join(HERE, 'skill_deterministic.csv'))
prob = pd.read_csv(os.path.join(HERE, 'skill_probabilistic.csv'))

# ── region display names ──────────────────────────────────────────────────────
REGION_LABEL = {
    'All India':            'All India',
    'northwest_india':      'NW India',
    'central_india':        'Central India',
    'south_peninsula':      'South Peninsula',
    'east_northeast_india': 'East & NE India',
}
REGIONS_ORDER = ['All India', 'northwest_india', 'central_india',
                 'south_peninsula', 'east_northeast_india']

# ── model style ───────────────────────────────────────────────────────────────
MODEL_STYLE = {
    'SPIRE':       dict(color='#1f77b4', marker='o', lw=2.0, ms=6, label='SPIRE'),
    'FuXi':        dict(color='#ff7f0e', marker='s', lw=2.0, ms=6, label='FuXi'),
    'ECMWF':       dict(color='#2ca02c', marker='^', lw=2.0, ms=6, label='ECMWF'),
    'MME':         dict(color='#9467bd', marker='D', lw=2.0, ms=6, label='MME'),
    'Persistence': dict(color='#7f7f7f', marker='x', lw=1.5, ms=6,
                        ls='--', label='Persistence'),
}

WEEKS = [1, 2, 3, 4, 5, 6]


def _agg(df, var, model, region, scale, metric):
    sub = df[(df.variable == var) & (df.model == model) &
             (df.region == region) & (df.scale == scale)]
    return [sub[sub.lead == w][metric].mean() for w in WEEKS]


def _hline(ax, y=0, **kw):
    ax.axhline(y, color='#aaaaaa', lw=0.8, ls='--', **kw)


def _style_ax(ax, ylabel='', ylim=None, title=''):
    ax.set_xlabel('Forecast Week', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(WEEKS)
    ax.tick_params(labelsize=8)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    _hline(ax)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — PCC & CRPSS weekly, All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig1_pcc_crpss_allIndia():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey='row')
    fig.suptitle('V2 Skill — All India, Weekly (model-own climatology)', fontsize=12, y=1.01)

    col_titles = ['Pattern Correlation (PCC)', 'CRPSS vs Climatology']
    row_titles = ['TP (mm/day)', 'Z500 (gpm)']
    vars_ = ['TP', 'Z500']
    metrics_det  = 'pcc'
    metrics_prob = 'crpss_clim'

    det_models  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
    prob_models = ['SPIRE', 'FuXi', 'ECMWF']

    for ri, var in enumerate(vars_):
        for ci, (df_use, models, metric) in enumerate([
                (det,  det_models,  metrics_det),
                (prob, prob_models, metrics_prob)]):
            ax = axes[ri, ci]
            for m in models:
                vals = _agg(df_use, var, m, 'All India', 'weekly', metric)
                st = MODEL_STYLE[m]
                ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                        lw=st.get('lw', 2), ms=st.get('ms', 5),
                        ls=st.get('ls', '-'), label=st['label'])
            ylim = (-0.4, 1.05) if ci == 0 else (-1.1, 1.05)
            _style_ax(ax, ylabel=col_titles[ci] if ri == 0 else '',
                      ylim=ylim,
                      title=f'{row_titles[ri]} — {col_titles[ci]}')

    # shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig1_pcc_crpss_allIndia.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — TP PCC weekly, all 5 IMD regions
# ══════════════════════════════════════════════════════════════════════════════
def fig2_pcc_regional_TP():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    fig.suptitle('TP Pattern Correlation — IMD Homogeneous Regions (Weekly)', fontsize=11)

    models = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
    for ax, rg in zip(axes, REGIONS_ORDER):
        for m in models:
            vals = _agg(det, 'TP', m, rg, 'weekly', 'pcc')
            st = MODEL_STYLE[m]
            ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                    lw=st.get('lw', 2), ms=st.get('ms', 5),
                    ls=st.get('ls', '-'), label=st['label'])
        _style_ax(ax, ylabel='PCC' if rg == REGIONS_ORDER[0] else '',
                  ylim=(-0.6, 1.05), title=REGION_LABEL[rg])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig2_pcc_regional_TP.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Z500 PCC weekly, all 5 IMD regions
# ══════════════════════════════════════════════════════════════════════════════
def fig3_pcc_regional_Z500():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    fig.suptitle('Z500 Pattern Correlation — IMD Homogeneous Regions (Weekly)', fontsize=11)

    models = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
    for ax, rg in zip(axes, REGIONS_ORDER):
        for m in models:
            vals = _agg(det, 'Z500', m, rg, 'weekly', 'pcc')
            st = MODEL_STYLE[m]
            ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                    lw=st.get('lw', 2), ms=st.get('ms', 5),
                    ls=st.get('ls', '-'), label=st['label'])
        _style_ax(ax, ylabel='PCC' if rg == REGIONS_ORDER[0] else '',
                  ylim=(-0.6, 1.05), title=REGION_LABEL[rg])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig3_pcc_regional_Z500.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — RMSE / Bias / SSR, All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig4_rmse_bias_ssr():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('V2 Error & Calibration — All India, Weekly', fontsize=11)

    vars_  = ['TP', 'Z500']
    units  = {'TP': 'mm/day', 'Z500': 'gpm'}
    cols   = [('rmse', 'RMSE'), ('bias', 'Bias (fcst − obs)'), ('ssr', 'Spread-Skill Ratio')]
    base_models = ['SPIRE', 'FuXi', 'ECMWF']
    all_models  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']

    for ri, var in enumerate(vars_):
        for ci, (metric, col_title) in enumerate(cols):
            ax = axes[ri, ci]
            models = base_models if metric == 'ssr' else all_models
            df_use = prob if metric == 'ssr' else det
            for m in models:
                vals = _agg(df_use, var, m, 'All India', 'weekly', metric)
                st = MODEL_STYLE[m]
                ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                        lw=st.get('lw', 2), ms=st.get('ms', 5),
                        ls=st.get('ls', '-'), label=st['label'])
            if metric == 'bias':
                _hline(ax)
            if metric == 'ssr':
                ax.axhline(1.0, color='#e04040', lw=1.0, ls='--', label='Perfect')
            ylabel = f'{col_title}\n({units[var]})'
            if metric == 'ssr':
                ylabel = 'SSR (ideal = 1)'
            _style_ax(ax, ylabel=ylabel,
                      title=f'{var} — {col_title}' if ri == 0 else '')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # deduplicate
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc='lower center', ncol=6,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig4_rmse_bias_ssr.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — MSSS + CRPSS, all 5 regions, TP + Z500  (2-var × 5-region grid)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_msss_crpss_regional():
    vars_   = ['TP', 'Z500']
    metrics = [('msss_clim', det,  ['SPIRE', 'FuXi', 'ECMWF', 'MME'], 'MSSS vs Clim'),
               ('crpss_clim', prob, ['SPIRE', 'FuXi', 'ECMWF'],        'CRPSS vs Clim')]

    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharey='row')
    fig.suptitle('MSSS & CRPSS — IMD Homogeneous Regions (Weekly)', fontsize=12)

    row = 0
    for var in vars_:
        for metric, df_use, models, mname in metrics:
            for ci, rg in enumerate(REGIONS_ORDER):
                ax = axes[row, ci]
                for m in models:
                    vals = _agg(df_use, var, m, rg, 'weekly', metric)
                    st = MODEL_STYLE[m]
                    ax.plot(WEEKS, vals, color=st['color'], marker=st['marker'],
                            lw=st.get('lw', 2), ms=st.get('ms', 5),
                            ls=st.get('ls', '-'), label=st['label'])
                ylim = (-2.5, 1.1) if var == 'Z500' and metric == 'crpss_clim' else (-1.5, 1.1)
                title = REGION_LABEL[rg] if row == 0 else ''
                ylabel = f'{var}\n{mname}' if ci == 0 else ''
                _style_ax(ax, ylabel=ylabel, ylim=ylim, title=title)
            row += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig5_msss_crpss_regional.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Daily PCC (lead day 1-42), All India, TP + Z500
# ══════════════════════════════════════════════════════════════════════════════
def fig6_daily_pcc():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Daily PCC vs Lead Day — All India', fontsize=11)

    vars_  = ['TP', 'Z500']
    models = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
    days   = list(range(1, 43))

    for ax, var in zip(axes, vars_):
        for m in models:
            sub = det[(det.variable == var) & (det.model == m) &
                      (det.region == 'All India') & (det.scale == 'daily')]
            vals = [sub[sub.lead == d]['pcc'].mean() for d in days]
            st = MODEL_STYLE[m]
            ax.plot(days, vals, color=st['color'], lw=st.get('lw', 1.5),
                    ls=st.get('ls', '-'), label=st['label'], alpha=0.85)
        # week dividers
        for w in range(7, 43, 7):
            ax.axvline(w, color='#cccccc', lw=0.7, ls=':')
        ax.set_xlabel('Lead Day', fontsize=9)
        ax.set_ylabel('PCC', fontsize=9)
        ax.set_xlim(1, 42)
        ax.set_ylim(-0.5, 1.05)
        ax.set_xticks(range(7, 43, 7))
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', lw=0.4, alpha=0.5)
        _hline(ax)
        ax.set_title(var, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out = os.path.join(ODIR, 'fig6_daily_pcc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE {out}')


if __name__ == '__main__':
    fig1_pcc_crpss_allIndia()
    fig2_pcc_regional_TP()
    fig3_pcc_regional_Z500()
    fig4_rmse_bias_ssr()
    fig5_msss_crpss_regional()
    fig6_daily_pcc()
    print('ALL DONE')
