#!/usr/bin/env python3
"""
summarize_csv.py — Compact human-readable summary of the V3 result CSVs.
Writes CSV_SUMMARY.md (and prints to stdout).

Covers: skill_deterministic.csv, skill_probabilistic.csv, skill_brier.csv
Headline view = All India, weekly, ERA5 basis (the fair-comparison numbers),
plus the model-own basis where available, plus dataset coverage.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'CSV_SUMMARY.md')
WEEKS = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6']
WCOL  = [1, 2, 3, 4, 5, 6]

det  = pd.read_csv(os.path.join(HERE, 'skill_deterministic.csv'))
prob = pd.read_csv(os.path.join(HERE, 'skill_probabilistic.csv'))
brier = pd.read_csv(os.path.join(HERE, 'skill_brier.csv'))

lines = []
def w(s=''): lines.append(s)


def wk_table(df, metric, models, region='All India', var='TP', basis='era5',
             scale='weekly', fmt='{:+.2f}'):
    """Markdown table: rows=models, cols=W1..W6 of weekly-mean metric."""
    sub = df[(df.region == region) & (df.variable == var) & (df.scale == scale)]
    if 'clim_basis' in sub.columns:
        sub = sub[sub.clim_basis == basis]
    hdr = '| Model | ' + ' | '.join(f'W{i}' for i in WCOL) + ' |'
    sep = '|' + '---|' * (len(WCOL) + 1)
    rows = [hdr, sep]
    for m in models:
        sm = sub[sub.model == m]
        vals = [sm[sm.lead == wkn][metric].mean() for wkn in WCOL]
        cells = [fmt.format(v) if not np.isnan(v) else '–' for v in vals]
        rows.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def brier_table(event, models, var, region='All India'):
    sub = brier[(brier.scale == 'weekly') & (brier.region == region) &
                (brier.event == event) & (brier.variable == var)]
    hdr = '| Model | ' + ' | '.join(f'W{i}' for i in WCOL) + ' |'
    sep = '|' + '---|' * (len(WCOL) + 1)
    rows = [hdr, sep]
    for m in models:
        sm = sub[sub.model == m]
        vals = [sm[sm.lead == wkn]['briss_clim'].mean() for wkn in WCOL]
        cells = [f'{v:+.2f}' if not np.isnan(v) else '–' for v in vals]
        rows.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


# ── header / coverage ──────────────────────────────────────────────────────────
w('# V3 Skill CSV Summary')
w()
w('All India · weekly · JFM 2026 · 13 initialisations.  '
  'ERA5 basis = every model vs ERA5 climatology (fair comparison).')
w()
w('## Dataset coverage')
w()
w('| File | Rows | Models | Variables | Regions | Bases |')
w('|---|---|---|---|---|---|')
for name, df in [('skill_deterministic.csv', det),
                 ('skill_probabilistic.csv', prob),
                 ('skill_brier.csv', brier)]:
    bases = ','.join(sorted(df.clim_basis.unique())) if 'clim_basis' in df.columns else '–'
    w(f'| {name} | {len(df):,} | {len(df.model.unique())} | '
      f'{len(df.variable.unique())} | {len(df.region.unique())} | {bases} |')
w()

DET_M  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
PROB_M = ['SPIRE', 'FuXi', 'ECMWF']
OWN_M  = ['FuXi', 'ECMWF', 'MME']

# ── deterministic ──────────────────────────────────────────────────────────────
w('## Deterministic (ERA5 basis, All India)')
for var in ['TP', 'Z500']:
    w(); w(f'### {var} — PCC (anomaly correlation)')
    w(wk_table(det, 'pcc', DET_M, var=var, fmt='{:.2f}'))
    w(); w(f'### {var} — RMSE')
    w(wk_table(det, 'rmse', ['SPIRE', 'FuXi', 'ECMWF', 'MME'], var=var, fmt='{:.2f}'))
    w(); w(f'### {var} — Bias')
    w(wk_table(det, 'bias', ['SPIRE', 'FuXi', 'ECMWF', 'MME'], var=var, fmt='{:+.2f}'))
    w(); w(f'### {var} — MSSS vs climatology')
    w(wk_table(det, 'msss_clim', ['SPIRE', 'FuXi', 'ECMWF', 'MME'], var=var))

# ── model-own basis ────────────────────────────────────────────────────────────
w(); w('## Deterministic (MODEL-OWN basis — FuXi/ECMWF/MME only)')
for var in ['TP', 'Z500']:
    w(); w(f'### {var} — PCC (each model vs its own climatology)')
    w(wk_table(det, 'pcc', OWN_M, var=var, basis='model_own', fmt='{:.2f}'))

# ── probabilistic ──────────────────────────────────────────────────────────────
w(); w('## Probabilistic (ERA5 basis, All India)')
for var in ['TP', 'Z500']:
    w(); w(f'### {var} — CRPSS vs climatology')
    w(wk_table(prob, 'crpss_clim', PROB_M, var=var))
    w(); w(f'### {var} — Spread-skill ratio (SSR, 1=calibrated)')
    w(wk_table(prob, 'ssr', PROB_M, var=var, fmt='{:.2f}'))

# ── brier ──────────────────────────────────────────────────────────────────────
w(); w('## Brier Skill Score (vs climatology, All India)')
for var, events in [('TP', ['tp_above_normal', 'tp_below_normal']),
                    ('Z500', ['z500_above_normal', 'z500_below_normal'])]:
    for ev in events:
        w(); w(f'### {ev}')
        w(brier_table(ev, PROB_M, var))

# ── one-line takeaways ─────────────────────────────────────────────────────────
def w1(df, metric, m, var, basis='era5'):
    s = df[(df.region == 'All India') & (df.variable == var) & (df.scale == 'weekly') &
           (df.model == m) & (df.lead == 1)]
    if 'clim_basis' in df.columns:
        s = s[s.clim_basis == basis]
    return s[metric].mean()

w(); w('## Headline takeaways (Week 1, All India)')
w(f'- **TP PCC:** SPIRE {w1(det,"pcc","SPIRE","TP"):.2f}, '
  f'FuXi {w1(det,"pcc","FuXi","TP"):.2f}, ECMWF {w1(det,"pcc","ECMWF","TP"):.2f} '
  f'— SPIRE leads precipitation.')
w(f'- **Z500 PCC:** ECMWF {w1(det,"pcc","ECMWF","Z500"):.2f}, '
  f'SPIRE {w1(det,"pcc","SPIRE","Z500"):.2f}, FuXi {w1(det,"pcc","FuXi","Z500"):.2f} '
  f'— ECMWF best short-range; FuXi Z500 collapses after W2.')
w(f'- **TP CRPSS:** SPIRE {w1(prob,"crpss_clim","SPIRE","TP"):+.2f}, '
  f'ECMWF {w1(prob,"crpss_clim","ECMWF","TP"):+.2f}, '
  f'FuXi {w1(prob,"crpss_clim","FuXi","TP"):+.2f}.')

text = '\n'.join(lines) + '\n'
with open(OUT, 'w') as f:
    f.write(text)
print(text)
print(f'\n[written] {OUT}')
