#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fig_mjo_rmm_skill.py
====================
Canonical MJO forecast-skill metrics for the Spire AI-S2S paper:
bivariate RMM correlation (COR) and bivariate RMSE vs forecast lead day,
following Gottschalck et al. (2010) / Lin et al. (2008) / Rashid et al. (2011).

This is THE standard MJO prediction-skill diagram. The "useful forecast
horizon" is the lead at which COR drops below 0.5 (and RMSE exceeds the
sqrt(2) climatological reference).

Definitions (a = observed RMM, b = forecast RMM, summed over N inits at lead τ):
  COR(τ)  = Σ(a1 b1 + a2 b2) / sqrt(Σ(a1²+a2²)) / sqrt(Σ(b1²+b2²))
  RMSE(τ) = sqrt( (1/N) Σ[(a1-b1)² + (a2-b2)²] )

Forecast RMM is built exactly as in fig_mjo_phase_diagram.py:
  WH04 scalar field normalisation + day-1 calibration to the observed RMM scale.

Produces: figures/fig20_mjo_rmm_skill.png
Author: Ayush Raj  |  Created 2026-06-08
"""
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGD = BASE / "figures"; FIGD.mkdir(exist_ok=True)

# ── Build forecast RMM (same method as the corrected phase diagram) ──────────
h = xr.open_dataset(DATA / "equatorial_hovmoller.nc")
e = xr.open_dataset(DATA / "eofs_MJO.nc")
it = pd.DatetimeIndex(h['init_time'].values)
steps = h['step'].values                     # 1..42
e1 = np.concatenate([e['eof1_olr'].values, e['eof1_u850'].values, e['eof1_u200'].values])
e2 = np.concatenate([e['eof2_olr'].values, e['eof2_u850'].values, e['eof2_u200'].values])

olr, u8, u2 = h['spire_olr_eq'].values, h['spire_u850_eq'].values, h['spire_u200_eq'].values
oa = olr - olr.mean(0, keepdims=True)
ua = u8  - u8.mean(0,  keepdims=True)
va = u2  - u2.mean(0,  keepdims=True)
idx = np.arange(0, 720, 2)
oa, ua, va = oa[:, :, idx], ua[:, :, idx], va[:, :, idx]
NF_OLR, NF_U850, NF_U200 = 15.1, 1.81, 4.81
on, un, vn = oa / NF_OLR, ua / NF_U850, va / NF_U200

n_inits, n_steps = on.shape[0], on.shape[1]
p1 = np.full((n_inits, n_steps), np.nan); p2 = np.full((n_inits, n_steps), np.nan)
for i in range(n_inits):
    for s in range(n_steps):
        c = np.concatenate([on[i, s], un[i, s], vn[i, s]])
        if not np.any(np.isnan(c)):
            p1[i, s] = c @ e1; p2[i, s] = c @ e2

# ── Observed RMM ─────────────────────────────────────────────────────────────
cols = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'source']
rmm = pd.read_csv(DATA / "rmm.74toRealtime.txt", skiprows=2, sep=r'\s+', names=cols)
rmm['date'] = pd.to_datetime(rmm[['year', 'month', 'day']]); rmm = rmm.set_index('date')

def obs_rmm(date):
    d = pd.Timestamp(date)
    if d in rmm.index:
        r = rmm.loc[d]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        return float(r['RMM1']), float(r['RMM2'])
    return np.nan, np.nan

# Day-1 calibration to observed scale
o1i = np.array([obs_rmm(d)[0] for d in it]); o2i = np.array([obs_rmm(d)[1] for d in it])
m = np.isfinite(o1i) & np.isfinite(p1[:, 0])
s1 = np.nansum(o1i[m] * p1[m, 0]) / np.nansum(p1[m, 0] ** 2)
s2 = np.nansum(o2i[m] * p2[m, 0]) / np.nansum(p2[m, 0] ** 2)
f1, f2 = p1 * s1, p2 * s2
print(f"calibration s1={s1:.3f} s2={s2:.3f}")

# ── Bivariate COR & RMSE vs lead ─────────────────────────────────────────────
COR = np.full(n_steps, np.nan); RMSE = np.full(n_steps, np.nan); NN = np.zeros(n_steps, int)
for s in range(n_steps):
    a1, a2, b1, b2 = [], [], [], []
    for i in range(n_inits):
        vd = it[i] + pd.Timedelta(int(steps[s]), 'D')
        o1, o2 = obs_rmm(vd)
        if np.isfinite(o1) and np.isfinite(f1[i, s]):
            a1.append(o1); a2.append(o2); b1.append(f1[i, s]); b2.append(f2[i, s])
    a1, a2, b1, b2 = map(np.array, (a1, a2, b1, b2))
    if len(a1) >= 5:
        num = np.sum(a1*b1 + a2*b2)
        den = np.sqrt(np.sum(a1**2 + a2**2)) * np.sqrt(np.sum(b1**2 + b2**2))
        COR[s] = num/den
        RMSE[s] = np.sqrt(np.mean((a1-b1)**2 + (a2-b2)**2))
        NN[s] = len(a1)

lead = steps.astype(int)
# Useful horizon = FIRST lead at which COR drops below 0.5 (a later rebound is
# low-frequency/seasonal contamination, not genuine MJO skill — see caveat).
below = np.where(COR < 0.5)[0]
horizon = int(lead[below[0]] - 1) if len(below) else int(lead[-1])
print(f"First COR<0.5 horizon: {horizon} days   "
      f"(COR day1={COR[0]:.2f}, day14={COR[13]:.2f}, min={np.nanmin(COR):.2f}@d{lead[np.nanargmin(COR)]})")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (axc, axr) = plt.subplots(1, 2, figsize=(12, 4.8))

axc.plot(lead, COR, '-', color='#C0392B', lw=2.2, zorder=3)
axc.plot(lead, COR, 'o', color='#C0392B', ms=4, mec='white', mew=0.5, zorder=4)
axc.axhline(0.5, color='gray', ls='--', lw=1.1)
axc.fill_between(lead, 0.5, 1.0, color='#4caf50', alpha=0.07)
if horizon > 0:
    axc.axvline(horizon, color='#2166AC', ls=':', lw=1.4)
    axc.text(horizon+0.4, 0.08, f'horizon ≈ {horizon} d\n(first COR=0.5)',
             fontsize=9, color='#2166AC', va='bottom')
# Flag the long-lead rebound as contamination, not skill
imin = int(np.nanargmin(COR))
axc.annotate('long-lead rebound =\nlow-freq. contamination',
             xy=(lead[-3], COR[-3]), xytext=(28, 0.20), fontsize=8, color='0.4',
             ha='center', arrowprops=dict(arrowstyle='->', color='0.6', lw=0.8))
axc.text(1, 0.52, 'COR = 0.5 (useful-skill threshold)', fontsize=8.5, color='gray', va='bottom')
axc.set_xlim(1, 42); axc.set_ylim(0, 1.0)
axc.set_xlabel('Forecast lead (days)'); axc.set_ylabel('Bivariate RMM correlation (COR)')
axc.set_title('(a) MJO bivariate correlation', fontweight='bold')
axc.grid(True, alpha=0.3)

axr.plot(lead, RMSE, '-', color='#2166AC', lw=2.2, zorder=3)
axr.plot(lead, RMSE, 's', color='#2166AC', ms=4, mec='white', mew=0.5, zorder=4)
axr.axhline(np.sqrt(2), color='gray', ls='--', lw=1.1)
axr.text(1, np.sqrt(2)+0.02, 'climatological reference (√2)', fontsize=8.5, color='gray', va='bottom')
axr.set_xlim(1, 42); axr.set_ylim(0, max(2.0, np.nanmax(RMSE)*1.1))
axr.set_xlabel('Forecast lead (days)'); axr.set_ylabel('Bivariate RMM RMSE')
axr.set_title('(b) MJO bivariate RMSE', fontweight='bold')
axr.grid(True, alpha=0.3)

for ax in (axc, axr):
    for w in range(1, 7):
        ax.axvline(w*7, ls=':', lw=0.5, color='0.8')

fig.suptitle('Spire AI-S2S | MJO Prediction Skill (RMM bivariate) — JFM 2026, '
             f'{n_inits} inits', fontweight='bold', y=1.0)
fig.tight_layout()
out = FIGD / "fig20_mjo_rmm_skill.png"
fig.savefig(out); plt.close(fig)
print(f"Saved {out}  ({out.stat().st_size/1024:.0f} KB)")
