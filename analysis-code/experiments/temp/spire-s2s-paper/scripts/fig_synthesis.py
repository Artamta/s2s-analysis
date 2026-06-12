#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fig_synthesis.py — Publication-quality verification figures (Fig 6–10)
for Spire JFM 2026 S2S hindcast vs ERA5.

Produces:
  fig06_scatter_gridcell   — Grid-cell scatter (hexbin) for 4 vars × 3 weeks
  fig07_scatter_initmean   — Init-date-mean scatter for T2m-mean/max × 3 weeks
  fig08_scorecard_heatmap  — ACC & RMSE scorecard (4 vars × 6 weeks)
  fig09_anomaly_timeseries — India-mean anomaly time series (4 vars, W1–W6)
  fig10_verification_dashboard — 6-panel synthesis dashboard
"""

import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from scipy import stats

warnings.filterwarnings('ignore')

# ── rcParams ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_FILE = '/home/raj.ayush/s2s/s2s_anlysis/spire_era5/s2s_verification/weekly_anomalies_v2.nc'
OUT_DIR   = '/home/raj.ayush/s2s/s2s_anlysis/spire-s2s-paper/figures/'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────────────
INDIA_LAT = slice(6, 36)
INDIA_LON = slice(68, 98)

VAR_PAIRS = [
    ('spire_t2m_mean_anom', 'era5_t2m_mean_anom', 'T2m Mean', 'K'),
    ('spire_t2m_max_anom',  'era5_t2m_max_anom',  'T2m Max',  'K'),
    ('spire_precip_anom',   'era5_precip_anom',    'Precip',   'mm day$^{-1}$'),
    ('spire_z500_anom',     'era5_z500_anom',      'Z500',     'gpm'),
]

WEEK_SHOW = [0, 2, 5]          # W1, W3, W6 (0-indexed)
WEEK_LABELS = ['W1', 'W3', 'W6']
ALL_WEEK_LABELS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']

BBOX = dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9)

PANEL_LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def _india(da):
    """Subset to India domain."""
    return da.sel(latitude=INDIA_LAT, longitude=INDIA_LON)

def _save(fig, stem):
    """Save as PNG and return file size."""
    png = os.path.join(OUT_DIR, f'{stem}.png')
    fig.savefig(png)
    plt.close(fig)
    return png

def _stat_text(x, y, include_mae=False):
    """Return annotation string with R², RMSE, bias (optionally MAE)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 'N/A'
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2
    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)
    txt = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
    if include_mae:
        mae = np.mean(np.abs(y - x))
        txt += f'\nMAE = {mae:.3f}'
    return txt

# ── Load data ───────────────────────────────────────────────────────────────
print('Loading data …')
t0 = time.time()
ds = xr.open_dataset(DATA_FILE)
print(f'  Loaded in {time.time()-t0:.1f}s  |  dims = {dict(ds.dims)}')

file_sizes = {}   # stem → (png_bytes, pdf_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — Grid-cell scatter (hexbin)
# ═══════════════════════════════════════════════════════════════════════════
print('\n▶ Fig 6: Grid-cell scatter plots …')
t0 = time.time()

fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)

rng = np.random.default_rng(42)
MAX_PTS = 5000

for irow, (sv, ev, vname, unit) in enumerate(VAR_PAIRS):
    for icol, (wi, wl) in enumerate(zip(WEEK_SHOW, WEEK_LABELS)):
        ax = axes[irow, icol]

        # Flatten over India, all inits
        s = _india(ds[sv].isel(week=wi)).values.ravel()
        e = _india(ds[ev].isel(week=wi)).values.ravel()
        mask = np.isfinite(s) & np.isfinite(e)
        s, e = s[mask], e[mask]

        # Subsample
        if len(s) > MAX_PTS:
            idx = rng.choice(len(s), MAX_PTS, replace=False)
            s_sub, e_sub = s[idx], e[idx]
        else:
            s_sub, e_sub = s, e

        # Hexbin
        vmin_plot = min(e_sub.min(), s_sub.min())
        vmax_plot = max(e_sub.max(), s_sub.max())
        pad = 0.05 * (vmax_plot - vmin_plot) if vmax_plot != vmin_plot else 1.0
        lims = (vmin_plot - pad, vmax_plot + pad)

        hb = ax.hexbin(e_sub, s_sub, gridsize=50, mincnt=1,
                        cmap='YlOrBr', linewidths=0.2)

        # 1:1 line
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.6, label='1:1')

        # Regression line
        slope, intercept, *_ = stats.linregress(e_sub, s_sub)
        xfit = np.linspace(lims[0], lims[1], 100)
        ax.plot(xfit, slope * xfit + intercept, 'r-', lw=1.2, label='Reg.')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        # Stats
        txt = _stat_text(e_sub, s_sub)
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left', bbox=BBOX)

        # Panel letter
        letter = PANEL_LETTERS[irow * 3 + icol]
        ax.text(-0.02, 1.06, f'({letter})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')

        # Column title (top row only)
        if irow == 0:
            ax.set_title(wl, fontweight='bold')

        # Row label (first column only)
        if icol == 0:
            ax.set_ylabel(f'{vname}\nSpire anom. [{unit}]')
        else:
            ax.set_ylabel('')

        # X label (bottom row only)
        if irow == 3:
            ax.set_xlabel(f'ERA5 anom. [{unit}]')
        else:
            ax.set_xlabel('')

        ax.tick_params(direction='in')

fig.suptitle('Grid-cell Scatter: Spire vs ERA5 Anomalies (India)', fontsize=13, fontweight='bold', y=1.01)

png = _save(fig, 'fig06_scatter_gridcell')
file_sizes['fig06_scatter_gridcell'] = os.path.getsize(png)
print(f'  Done in {time.time()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7 — Init-date-mean scatter
# ═══════════════════════════════════════════════════════════════════════════
print('\n▶ Fig 7: Init-date-mean scatter …')
t0 = time.time()

fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)

# Color by month (Jan=blue, Feb=green, Mar=red)
init_times = ds['init_time'].values
month_num = np.array([np.datetime64(t, 'M').astype(int) % 12 + 1
                       for t in init_times])  # 1=Jan,2=Feb,3=Mar
month_colors = {1: '#2166ac', 2: '#1b7837', 3: '#b2182b'}
colors = np.array([month_colors.get(m, 'gray') for m in month_num])
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar'}

for irow, (sv, ev, vname, unit) in enumerate(VAR_PAIRS[:2]):  # T2m-mean, T2m-max
    for icol, (wi, wl) in enumerate(zip(WEEK_SHOW, WEEK_LABELS)):
        ax = axes[irow, icol]

        # India-mean per init
        s_all = _india(ds[sv].isel(week=wi)).mean(dim=['latitude', 'longitude']).values
        e_all = _india(ds[ev].isel(week=wi)).mean(dim=['latitude', 'longitude']).values

        for m in [1, 2, 3]:
            mask_m = month_num == m
            ax.scatter(e_all[mask_m], s_all[mask_m], c=month_colors[m],
                       s=40, alpha=0.8, edgecolors='k', linewidths=0.3,
                       label=month_names[m], zorder=3)

        # Limits
        vmin_p = min(e_all.min(), s_all.min())
        vmax_p = max(e_all.max(), s_all.max())
        pad = 0.1 * (vmax_p - vmin_p) if vmax_p != vmin_p else 0.5
        lims = (vmin_p - pad, vmax_p + pad)

        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.6)
        slope, intercept, *_ = stats.linregress(e_all, s_all)
        xfit = np.linspace(lims[0], lims[1], 100)
        ax.plot(xfit, slope * xfit + intercept, 'r-', lw=1.2)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        txt = _stat_text(e_all, s_all, include_mae=True)
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left', bbox=BBOX)

        letter = PANEL_LETTERS[irow * 3 + icol]
        ax.text(-0.02, 1.06, f'({letter})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')

        if irow == 0:
            ax.set_title(wl, fontweight='bold')
        if icol == 0:
            ax.set_ylabel(f'{vname}\nSpire anom. [{unit}]')
        else:
            ax.set_ylabel('')
        if irow == 1:
            ax.set_xlabel(f'ERA5 anom. [{unit}]')
        else:
            ax.set_xlabel('')

        if irow == 0 and icol == 2:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
                      edgecolor='gray', markerscale=0.8)

        ax.tick_params(direction='in')

fig.suptitle('Init-date Mean Scatter: Spire vs ERA5 (India domain)',
             fontsize=13, fontweight='bold', y=1.01)

png = _save(fig, 'fig07_scatter_initmean')
file_sizes['fig07_scatter_initmean'] = os.path.getsize(png)
print(f'  Done in {time.time()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8 — ACC & RMSE scorecard heatmap
# ═══════════════════════════════════════════════════════════════════════════
print('\n▶ Fig 8: Scorecard heatmap …')
t0 = time.time()

n_vars = 4
n_weeks = 6
acc_mat  = np.zeros((n_vars, n_weeks))
rmse_mat = np.zeros((n_vars, n_weeks))

for iv, (sv, ev, vname, unit) in enumerate(VAR_PAIRS):
    for iw in range(n_weeks):
        s = _india(ds[sv].isel(week=iw))   # (init, lat, lon)
        e = _india(ds[ev].isel(week=iw))

        # ACC: pixel-wise Pearson r across 90 inits, then spatial mean
        # Vectorised: correlation for each pixel
        s_np = s.values   # (ninit, nlat, nlon)
        e_np = e.values
        s_mean = s_np.mean(axis=0, keepdims=True)
        e_mean = e_np.mean(axis=0, keepdims=True)
        s_anom = s_np - s_mean
        e_anom = e_np - e_mean
        num = (s_anom * e_anom).sum(axis=0)
        den = np.sqrt((s_anom**2).sum(axis=0) * (e_anom**2).sum(axis=0))
        den = np.where(den == 0, np.nan, den)
        r_map = num / den
        acc_mat[iv, iw] = np.nanmean(r_map)

        # RMSE over India pixels and 90 inits
        diff = s_np - e_np
        rmse_mat[iv, iw] = np.sqrt(np.nanmean(diff ** 2))

var_labels = [vp[2] for vp in VAR_PAIRS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# --- ACC heatmap ---
cmap_acc = LinearSegmentedColormap.from_list(
    'acc_gyr', ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'], N=256)
im1 = ax1.imshow(acc_mat, cmap=cmap_acc, vmin=0, vmax=1, aspect='auto')
for i in range(n_vars):
    for j in range(n_weeks):
        ax1.text(j, i, f'{acc_mat[i,j]:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 color='white' if acc_mat[i,j] < 0.4 else 'black')
ax1.set_xticks(range(n_weeks))
ax1.set_xticklabels(ALL_WEEK_LABELS)
ax1.set_yticks(range(n_vars))
ax1.set_yticklabels(var_labels)
ax1.set_xlabel('Lead Week')
ax1.set_title('(a) Anomaly Correlation (ACC)', fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='ACC')

# --- RMSE heatmap ---
im2 = ax2.imshow(rmse_mat, cmap='YlOrRd', aspect='auto')
for i in range(n_vars):
    for j in range(n_weeks):
        ax2.text(j, i, f'{rmse_mat[i,j]:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 color='white' if rmse_mat[i,j] > rmse_mat.max()*0.7 else 'black')
ax2.set_xticks(range(n_weeks))
ax2.set_xticklabels(ALL_WEEK_LABELS)
ax2.set_yticks(range(n_vars))
ax2.set_yticklabels(var_labels)
ax2.set_xlabel('Lead Week')
ax2.set_title('(b) RMSE', fontweight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='RMSE')

fig.suptitle('Verification Scorecard: ACC & RMSE (India domain, JFM 2026)',
             fontsize=13, fontweight='bold', y=1.03)

png = _save(fig, 'fig08_scorecard_heatmap')
file_sizes['fig08_scorecard_heatmap'] = os.path.getsize(png)
print(f'  Done in {time.time()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9 — India-mean anomaly time series
# ═══════════════════════════════════════════════════════════════════════════
print('\n▶ Fig 9: Anomaly time series …')
t0 = time.time()

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True, constrained_layout=True)

weeks = np.arange(1, 7)

for iv, (sv, ev, vname, unit) in enumerate(VAR_PAIRS):
    ax = axes[iv]

    # India-mean per init and week  → (init, week)
    s_india = _india(ds[sv]).mean(dim=['latitude', 'longitude'])  # (init, week)
    e_india = _india(ds[ev]).mean(dim=['latitude', 'longitude'])

    s_mean = s_india.mean(dim='init_time').values  # (6,)
    e_mean = e_india.mean(dim='init_time').values
    s_std  = s_india.std(dim='init_time').values
    e_std  = e_india.std(dim='init_time').values

    ax.fill_between(weeks, e_mean - e_std, e_mean + e_std,
                    color='#2166ac', alpha=0.15, label='ERA5 ±1σ')
    ax.fill_between(weeks, s_mean - s_std, s_mean + s_std,
                    color='#b2182b', alpha=0.15, label='Spire ±1σ')
    ax.plot(weeks, e_mean, 'o--', color='#2166ac', lw=1.8, ms=5, label='ERA5')
    ax.plot(weeks, s_mean, 's-',  color='#b2182b', lw=1.8, ms=5, label='Spire')

    ax.axhline(0, color='gray', lw=0.6, ls='-', alpha=0.5)
    ax.set_ylabel(f'{vname} [{unit}]')
    ax.text(-0.07, 1.02, f'({PANEL_LETTERS[iv]})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')
    ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis='both', alpha=0.3, ls='--')

axes[-1].set_xlabel('Lead Week')
axes[-1].set_xticks(weeks)
axes[-1].set_xticklabels(ALL_WEEK_LABELS)

fig.suptitle('India-mean Anomaly vs Lead Week (JFM 2026)',
             fontsize=13, fontweight='bold', y=1.01)

png = _save(fig, 'fig09_anomaly_timeseries')
file_sizes['fig09_anomaly_timeseries'] = os.path.getsize(png)
print(f'  Done in {time.time()-t0:.1f}s')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10 — Verification dashboard (6-panel synthesis)
# ═══════════════════════════════════════════════════════════════════════════
print('\n▶ Fig 10: Verification dashboard …')
t0 = time.time()

fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.32)

var_colors = {
    'T2m Mean': '#d62728',
    'T2m Max':  '#ff7f0e',
    'Precip':   '#2ca02c',
    'Z500':     '#1f77b4',
}
var_markers = {
    'T2m Mean': 'o',
    'T2m Max':  's',
    'Precip':   '^',
    'Z500':     'D',
}

weeks_arr = np.arange(1, 7)

# ── (a) ACC vs Lead ──────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for iv, (_, _, vname, _) in enumerate(VAR_PAIRS):
    ax_a.plot(weeks_arr, acc_mat[iv], marker=var_markers[vname], ms=6,
              lw=1.8, color=var_colors[vname], label=vname)
ax_a.set_xlabel('Lead Week')
ax_a.set_ylabel('ACC')
ax_a.set_xticks(weeks_arr)
ax_a.set_xticklabels(ALL_WEEK_LABELS)
ax_a.set_ylim(0, 1.05)
ax_a.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.7)
ax_a.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
ax_a.set_title('(a) ACC vs Lead', fontweight='bold')
ax_a.grid(alpha=0.3, ls='--')

# ── (b) RMSE vs Lead (T2m) ──────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
for iv in [0, 1]:  # T2m mean, T2m max
    vname = VAR_PAIRS[iv][2]
    ax_b.plot(weeks_arr, rmse_mat[iv], marker=var_markers[vname], ms=6,
              lw=1.8, color=var_colors[vname], label=vname)
ax_b.set_xlabel('Lead Week')
ax_b.set_ylabel('RMSE [K]')
ax_b.set_xticks(weeks_arr)
ax_b.set_xticklabels(ALL_WEEK_LABELS)
ax_b.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax_b.set_title('(b) RMSE vs Lead', fontweight='bold')
ax_b.grid(alpha=0.3, ls='--')

# ── (c) Bias vs Lead (T2m) ──────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
for iv in [0, 1]:
    sv, ev, vname, unit = VAR_PAIRS[iv]
    bias_w = np.zeros(n_weeks)
    for iw in range(n_weeks):
        s = _india(ds[sv].isel(week=iw)).values
        e = _india(ds[ev].isel(week=iw)).values
        bias_w[iw] = np.nanmean(s - e)
    ax_c.plot(weeks_arr, bias_w, marker=var_markers[vname], ms=6,
              lw=1.8, color=var_colors[vname], label=vname)
    # Shade where |bias| > 0.5 K
    above = np.abs(bias_w) > 0.5
    for iw in range(n_weeks):
        if above[iw]:
            ax_c.axvspan(weeks_arr[iw] - 0.35, weeks_arr[iw] + 0.35,
                         color=var_colors[vname], alpha=0.12)
ax_c.axhline(0, color='gray', lw=0.8)
ax_c.axhline(0.5, color='gray', ls=':', lw=0.7, alpha=0.5)
ax_c.axhline(-0.5, color='gray', ls=':', lw=0.7, alpha=0.5)
ax_c.set_xlabel('Lead Week')
ax_c.set_ylabel('Bias [K]')
ax_c.set_xticks(weeks_arr)
ax_c.set_xticklabels(ALL_WEEK_LABELS)
ax_c.legend(fontsize=8, loc='best', framealpha=0.9)
ax_c.set_title('(c) Bias vs Lead', fontweight='bold')
ax_c.grid(alpha=0.3, ls='--')

# ── (d) India-mean anomaly vs Lead (T2m-mean) ───────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
sv, ev = VAR_PAIRS[0][0], VAR_PAIRS[0][1]
s_india = _india(ds[sv]).mean(dim=['latitude', 'longitude'])
e_india = _india(ds[ev]).mean(dim=['latitude', 'longitude'])
s_m = s_india.mean(dim='init_time').values
e_m = e_india.mean(dim='init_time').values
s_s = s_india.std(dim='init_time').values
e_s = e_india.std(dim='init_time').values

ax_d.fill_between(weeks_arr, e_m - e_s, e_m + e_s, color='#2166ac', alpha=0.15)
ax_d.fill_between(weeks_arr, s_m - s_s, s_m + s_s, color='#b2182b', alpha=0.15)
ax_d.plot(weeks_arr, e_m, 'o--', color='#2166ac', lw=1.8, ms=5, label='ERA5')
ax_d.plot(weeks_arr, s_m, 's-',  color='#b2182b', lw=1.8, ms=5, label='Spire')
ax_d.axhline(0, color='gray', lw=0.6, alpha=0.5)
ax_d.set_xlabel('Lead Week')
ax_d.set_ylabel('T2m Mean Anom. [K]')
ax_d.set_xticks(weeks_arr)
ax_d.set_xticklabels(ALL_WEEK_LABELS)
ax_d.legend(fontsize=8, framealpha=0.9)
ax_d.set_title('(d) India-mean T2m Anomaly', fontweight='bold')
ax_d.grid(alpha=0.3, ls='--')

# ── (e) Best-case scatter: W1 T2m-mean init scatter ─────────────────────
ax_e = fig.add_subplot(gs[1, 1])
sv, ev = VAR_PAIRS[0][0], VAR_PAIRS[0][1]
s_init = _india(ds[sv].isel(week=0)).mean(dim=['latitude', 'longitude']).values
e_init = _india(ds[ev].isel(week=0)).mean(dim=['latitude', 'longitude']).values

for m in [1, 2, 3]:
    mask_m = month_num == m
    ax_e.scatter(e_init[mask_m], s_init[mask_m], c=month_colors[m],
                 s=40, alpha=0.8, edgecolors='k', linewidths=0.3,
                 label=month_names[m], zorder=3)

vmin_p = min(e_init.min(), s_init.min())
vmax_p = max(e_init.max(), s_init.max())
pad = 0.1 * (vmax_p - vmin_p) if vmax_p != vmin_p else 0.5
lims = (vmin_p - pad, vmax_p + pad)
ax_e.plot(lims, lims, 'k--', lw=0.8, alpha=0.6)
slope, intercept, *_ = stats.linregress(e_init, s_init)
xfit = np.linspace(lims[0], lims[1], 100)
ax_e.plot(xfit, slope * xfit + intercept, 'r-', lw=1.2)
ax_e.set_xlim(lims)
ax_e.set_ylim(lims)
ax_e.set_aspect('equal', adjustable='box')

txt = _stat_text(e_init, s_init, include_mae=True)
ax_e.text(0.04, 0.96, txt, transform=ax_e.transAxes,
          fontsize=7.5, va='top', ha='left', bbox=BBOX)
ax_e.set_xlabel('ERA5 T2m Mean Anom. [K]')
ax_e.set_ylabel('Spire T2m Mean Anom. [K]')
ax_e.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
ax_e.set_title('(e) W1 T2m-mean Init Scatter', fontweight='bold')
ax_e.tick_params(direction='in')

# ── (f) ACC heatmap (mini) ──────────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
im = ax_f.imshow(acc_mat, cmap=cmap_acc, vmin=0, vmax=1, aspect='auto')
for i in range(n_vars):
    for j in range(n_weeks):
        ax_f.text(j, i, f'{acc_mat[i,j]:.2f}', ha='center', va='center',
                  fontsize=9, fontweight='bold',
                  color='white' if acc_mat[i,j] < 0.4 else 'black')
ax_f.set_xticks(range(n_weeks))
ax_f.set_xticklabels(ALL_WEEK_LABELS)
ax_f.set_yticks(range(n_vars))
ax_f.set_yticklabels(var_labels)
ax_f.set_xlabel('Lead Week')
ax_f.set_title('(f) ACC Scorecard', fontweight='bold')
plt.colorbar(im, ax=ax_f, fraction=0.046, pad=0.04, label='ACC',
             ticks=[0, 0.25, 0.5, 0.75, 1.0])

fig.suptitle('Spire JFM 2026 S2S | Verification Skill Dashboard',
             fontsize=14, fontweight='bold', y=1.02)

png = _save(fig, 'fig10_verification_dashboard')
file_sizes['fig10_verification_dashboard'] = os.path.getsize(png)
print(f'  Done in {time.time()-t0:.1f}s')


# ── Summary ─────────────────────────────────────────────────────────────────
ds.close()

print('\n' + '='*55)
print(f'{"Figure":<32} {"PNG size":>10}')
print('-'*55)
for stem, psize in sorted(file_sizes.items()):
    print(f'{stem:<32} {psize/1024:>8.1f} KB')
print('='*55)
print(f'\nAll figures saved to: {OUT_DIR}')
print('Done ✓')
