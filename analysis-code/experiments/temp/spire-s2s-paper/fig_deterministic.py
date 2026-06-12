#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fig_deterministic.py
====================
Generate 5 publication-quality deterministic verification figures from
weekly_anomalies_v2.nc for the Spire AI-S2S paper.

Figures:
  1. ACC vs forecast lead week
  2. RMSE vs forecast lead week
  3. Mean Bias (Spire − ERA5) vs lead
  4. Spatial ACC maps (3 vars × 3 weeks)
  5. Spatial bias maps (2 vars × 3 weeks)

Author : Ayush Raj
Created: 2026-06-06
"""

# ── Imports & rcParams ──────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

import numpy as np
import xarray as xr
import os
import pathlib
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────
DATA_FILE = '/home/raj.ayush/s2s/s2s_anlysis/spire_era5/s2s_verification/weekly_anomalies_v2.nc'
OUT_DIR   = pathlib.Path('/home/raj.ayush/s2s/s2s_anlysis/spire-s2s-paper/figures/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────
print('Loading data …')
ds = xr.open_dataset(DATA_FILE)
print(f'  Dataset loaded: {dict(ds.dims)}')

lat = ds['latitude'].values
lon = ds['longitude'].values
weeks = ds['week'].values          # 1..6
n_init = ds.dims['init_time']      # 90
week_labels = [f'W{w}\n(d{(w-1)*7+1}-{w*7})' for w in weeks]

# ── India mask (6–36 °N, 68–98 °E) ─────────────────────────────────────
india_lat_mask = (lat >= 6) & (lat <= 36)
india_lon_mask = (lon >= 68) & (lon <= 98)

# ── Variable pairs ──────────────────────────────────────────────────────
VAR_PAIRS = {
    'T2m-mean': ('spire_t2m_mean_anom', 'era5_t2m_mean_anom'),
    'T2m-max':  ('spire_t2m_max_anom',  'era5_t2m_max_anom'),
    'Precip':   ('spire_precip_anom',    'era5_precip_anom'),
    'Z500':     ('spire_z500_anom',      'era5_z500_anom'),
}
UNITS = {'T2m-mean': 'K', 'T2m-max': 'K', 'Precip': 'mm day⁻¹', 'Z500': 'gpm'}
COLORS  = {'T2m-mean': '#d62728', 'T2m-max': '#ff7f0e',
           'Precip': '#2ca02c', 'Z500': '#9467bd'}
MARKERS = {'T2m-mean': 'o', 'T2m-max': 's', 'Precip': '^', 'Z500': 'D'}

# ── Helper: pixel-wise Pearson r over init_time ─────────────────────────
def pixelwise_acc(spire, era5):
    """Return (n_week, n_lat, n_lon) ACC array."""
    nw, nlat, nlon = len(weeks), len(lat), len(lon)
    acc = np.full((nw, nlat, nlon), np.nan, dtype=np.float32)
    for w in range(nw):
        s = spire[:, w, :, :]   # (90, lat, lon)
        e = era5[:, w, :, :]
        # Vectorised Pearson r along axis 0
        s_m = s - s.mean(axis=0, keepdims=True)
        e_m = e - e.mean(axis=0, keepdims=True)
        num = (s_m * e_m).sum(axis=0)
        den = np.sqrt((s_m**2).sum(axis=0) * (e_m**2).sum(axis=0))
        den = np.where(den == 0, np.nan, den)
        acc[w] = num / den
    return acc

def india_mean(arr_2d):
    """Mean over India sub-domain of a (lat, lon) array."""
    return np.nanmean(arr_2d[np.ix_(india_lat_mask, india_lon_mask)])

# ── Pre-compute metrics ────────────────────────────────────────────────
print('Computing pixel-wise ACC, RMSE, bias …')

acc_india = {}      # var -> (6,)  India-mean ACC per week
rmse_india = {}     # var -> (6,)
bias_india = {}     # var -> (6,)
acc_maps = {}       # var -> (6, lat, lon)
bias_maps = {}      # var -> (6, lat, lon)

for vname, (svar, evar) in VAR_PAIRS.items():
    s = ds[svar].values   # (90, 6, lat, lon)
    e = ds[evar].values

    # ACC maps
    acc_map = pixelwise_acc(s, e)      # (6, lat, lon)
    acc_maps[vname] = acc_map
    acc_india[vname] = np.array([india_mean(acc_map[w]) for w in range(len(weeks))])

    # RMSE (over India pixels AND inits)
    diff = s - e   # (90, 6, lat, lon)
    sq   = diff ** 2
    rmse_arr = np.zeros(len(weeks))
    bias_arr = np.zeros(len(weeks))
    for w in range(len(weeks)):
        sub = sq[:, w][:, india_lat_mask][:, :, india_lon_mask]
        rmse_arr[w] = np.sqrt(np.mean(sub))
        sub_b = diff[:, w][:, india_lat_mask][:, :, india_lon_mask]
        bias_arr[w] = np.mean(sub_b)
    rmse_india[vname] = rmse_arr
    bias_india[vname] = bias_arr

    # Bias maps  (mean over inits)
    bias_maps[vname] = diff.mean(axis=0)   # (6, lat, lon)

print('  Metrics computed.')

# ── Helper: save figure ────────────────────────────────────────────────
created_files = []

def savefig(fig, stem):
    p = OUT_DIR / f'{stem}.png'
    fig.savefig(str(p))
    created_files.append(p)
    plt.close(fig)
    print(f'  Saved {stem}.png')

# ====================================================================
# FIG 1 — ACC vs forecast lead week
# ====================================================================
print('\n── Figure 1: ACC vs lead ──')
fig, ax = plt.subplots(figsize=(7, 5))

# Skill bands
ax.axhspan(0.5, 1.05, color='#c8e6c9', alpha=0.35, zorder=0)
ax.axhspan(0.3, 0.5,  color='#fff9c4', alpha=0.40, zorder=0)
ax.axhspan(-0.6, 0.3, color='#ffcdd2', alpha=0.30, zorder=0)

ax.axhline(0.5, color='grey', ls='--', lw=0.8, zorder=1)
ax.axhline(0.3, color='grey', ls='--', lw=0.8, zorder=1)
ax.text(6.05, 0.51, 'skillful (0.5)', fontsize=7.5, color='#388e3c', va='bottom')
ax.text(6.05, 0.31, 'marginal (0.3)', fontsize=7.5, color='#f57f17', va='bottom')

x = np.arange(len(weeks))
for vname in VAR_PAIRS:
    ax.plot(x, acc_india[vname], marker=MARKERS[vname], color=COLORS[vname],
            lw=2, ms=7, label=f'{vname} ({UNITS[vname]})', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(week_labels)
ax.set_xlabel('Forecast Lead')
ax.set_ylabel('Anomaly Correlation Coefficient (ACC)')
ax.set_xlim(-0.3, len(weeks) - 0.7)
ax.set_ylim(-0.1, 1.02)
ax.legend(loc='lower left', framealpha=0.9, edgecolor='0.7')
ax.grid(axis='y', alpha=0.3, ls='--')

ax.set_title('Spire AI-S2S | Anomaly Correlation Coefficient vs Forecast Lead',
             fontsize=12, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'India-mean (6°–36°N, 68°–98°E) | JFM 2026 | 90 initializations',
        transform=ax.transAxes, ha='center', fontsize=9, color='0.4')

fig.tight_layout()
savefig(fig, 'fig01_acc_vs_lead')

# ====================================================================
# FIG 2 — RMSE vs forecast lead
# ====================================================================
print('\n── Figure 2: RMSE vs lead ──')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel (a) Temperature RMSE
ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')
for vname in ('T2m-mean', 'T2m-max'):
    ax1.plot(x, rmse_india[vname], marker=MARKERS[vname], color=COLORS[vname],
             lw=2, ms=7, label=vname)
ax1.set_xticks(x); ax1.set_xticklabels(week_labels)
ax1.set_xlabel('Forecast Lead'); ax1.set_ylabel('RMSE (K)')
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='0.7')
ax1.grid(axis='y', alpha=0.3, ls='--')
ax1.set_title('Temperature RMSE', fontweight='bold')

# Panel (b) Precip & Z500 RMSE  (dual y-axis)
ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')
l1, = ax2.plot(x, rmse_india['Precip'], marker=MARKERS['Precip'], color=COLORS['Precip'],
               lw=2, ms=7, label='Precip')
ax2.set_xticks(x); ax2.set_xticklabels(week_labels)
ax2.set_xlabel('Forecast Lead')
ax2.set_ylabel('Precip RMSE (mm day⁻¹)', color=COLORS['Precip'])
ax2.tick_params(axis='y', labelcolor=COLORS['Precip'])
ax2.grid(axis='y', alpha=0.3, ls='--')

ax2b = ax2.twinx()
l2, = ax2b.plot(x, rmse_india['Z500'], marker=MARKERS['Z500'], color=COLORS['Z500'],
                lw=2, ms=7, label='Z500')
ax2b.set_ylabel('Z500 RMSE (gpm)', color=COLORS['Z500'])
ax2b.tick_params(axis='y', labelcolor=COLORS['Z500'])

ax2.legend(handles=[l1, l2], loc='upper left', framealpha=0.9, edgecolor='0.7')
ax2.set_title('Precipitation & Z500 RMSE', fontweight='bold')

fig.suptitle('Spire AI-S2S | RMSE vs Forecast Lead — India-mean | JFM 2026',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig02_rmse_vs_lead')

# ====================================================================
# FIG 3 — Bias vs forecast lead
# ====================================================================
print('\n── Figure 3: Bias vs lead ──')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel (a) Temperature bias
ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')
for vname in ('T2m-mean', 'T2m-max'):
    ax1.plot(x, bias_india[vname], marker=MARKERS[vname], color=COLORS[vname],
             lw=2, ms=7, label=vname)
ax1.axhline(0, color='k', lw=0.8, ls='-')
# Shade positive / negative
ymin, ymax = ax1.get_ylim()
ax1.axhspan(0, max(ymax, 0.5), color='#ffcdd2', alpha=0.15, zorder=0)
ax1.axhspan(min(ymin, -0.5), 0, color='#bbdefb', alpha=0.15, zorder=0)
ax1.set_xticks(x); ax1.set_xticklabels(week_labels)
ax1.set_xlabel('Forecast Lead'); ax1.set_ylabel('Bias (Spire − ERA5) [K]')
ax1.legend(loc='best', framealpha=0.9, edgecolor='0.7')
ax1.grid(axis='y', alpha=0.3, ls='--')
ax1.set_title('Temperature Bias', fontweight='bold')

# Panel (b) Precip & Z500 bias  (dual y-axis)
ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')
l1, = ax2.plot(x, bias_india['Precip'], marker=MARKERS['Precip'], color=COLORS['Precip'],
               lw=2, ms=7, label='Precip')
ax2.axhline(0, color='k', lw=0.8, ls='-')
ax2.set_xticks(x); ax2.set_xticklabels(week_labels)
ax2.set_xlabel('Forecast Lead')
ax2.set_ylabel('Precip Bias (mm day⁻¹)', color=COLORS['Precip'])
ax2.tick_params(axis='y', labelcolor=COLORS['Precip'])
ax2.grid(axis='y', alpha=0.3, ls='--')

ax2b = ax2.twinx()
l2, = ax2b.plot(x, bias_india['Z500'], marker=MARKERS['Z500'], color=COLORS['Z500'],
                lw=2, ms=7, label='Z500')
ax2b.axhline(0, color='k', lw=0.5, ls=':')
ax2b.set_ylabel('Z500 Bias (gpm)', color=COLORS['Z500'])
ax2b.tick_params(axis='y', labelcolor=COLORS['Z500'])

ax2.legend(handles=[l1, l2], loc='best', framealpha=0.9, edgecolor='0.7')
ax2.set_title('Precipitation & Z500 Bias', fontweight='bold')

fig.suptitle('Spire AI-S2S | Mean Bias (Spire − ERA5) vs Forecast Lead — India-mean | JFM 2026',
             fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig03_bias_vs_lead')

# ====================================================================
# FIG 4 — Spatial ACC maps  (3 vars × 3 weeks)
# ====================================================================
print('\n── Figure 4: ACC skill maps ──')
map_vars   = ['T2m-mean', 'Precip', 'Z500']
map_weeks  = [0, 2, 5]   # W1, W3, W6
week_tags  = ['W1 (d1-7)', 'W3 (d15-21)', 'W6 (d36-42)']
panel_labels = list('abcdefghi')

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(3, 3, figsize=(10, 10),
                         subplot_kw={'projection': proj})

borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land',
                                        '50m', edgecolor='0.3', facecolor='none', linewidth=0.5)
coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                          edgecolor='0.3', facecolor='none', linewidth=0.5)

for row, vname in enumerate(map_vars):
    for col, (wi, wtag) in enumerate(zip(map_weeks, week_tags)):
        ax = axes[row, col]
        idx = row * 3 + col
        ax.set_extent([55, 105, 0, 50], crs=proj)
        ax.add_feature(coastline)

        data = acc_maps[vname][wi]
        cf = ax.contourf(lon, lat, data,
                         levels=np.linspace(-0.5, 1.0, 16),
                         cmap='RdYlGn', extend='both', transform=proj)

        # Stippling where |ACC| < 0.3
        stip_mask = np.abs(data) < 0.3
        lon2d, lat2d = np.meshgrid(lon, lat)
        ax.scatter(lon2d[stip_mask], lat2d[stip_mask],
                   s=0.15, c='0.4', marker='.', transform=proj, alpha=0.6)

        # India-mean ACC annotation
        im = india_mean(data)
        ax.text(0.97, 0.03, f'ACC={im:.2f}', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85))

        # Panel label
        ax.text(0.03, 0.97, f'({panel_labels[idx]})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

        # Row / column titles
        if row == 0:
            ax.set_title(wtag, fontsize=11, fontweight='bold', pad=8)
        if col == 0:
            ax.text(-0.18, 0.5, vname, transform=ax.transAxes, rotation=90,
                    fontsize=11, fontweight='bold', va='center', ha='center')

        # Grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--',
                          alpha=0.4, color='grey')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(55, 110, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 55, 10))
        if col != 0:
            gl.left_labels = False
        if row != 2:
            gl.bottom_labels = False

# Colorbar
cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cb.set_label('Anomaly Correlation Coefficient')

fig.suptitle('Spire AI-S2S | Spatial ACC — JFM 2026 (90 inits)',
             fontsize=13, fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.15, wspace=0.15, bottom=0.08, top=0.94, left=0.10)
savefig(fig, 'fig04_acc_skill_maps')

# ====================================================================
# FIG 5 — Spatial bias maps  (2 vars × 3 weeks)
# ====================================================================
print('\n── Figure 5: Bias maps ──')
bias_vars  = ['T2m-mean', 'T2m-max']
panel_labels5 = list('abcdef')

fig, axes = plt.subplots(2, 3, figsize=(10, 7),
                         subplot_kw={'projection': proj})

for row, vname in enumerate(bias_vars):
    for col, (wi, wtag) in enumerate(zip(map_weeks, week_tags)):
        ax = axes[row, col]
        idx = row * 3 + col
        ax.set_extent([55, 105, 0, 50], crs=proj)
        ax.add_feature(coastline)

        data = bias_maps[vname][wi]
        cf = ax.contourf(lon, lat, data,
                         levels=np.linspace(-3, 3, 13),
                         cmap='RdBu_r', extend='both', transform=proj)

        # India-mean bias annotation
        im = india_mean(data)
        ax.text(0.97, 0.03, f'Bias={im:+.2f} K', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85))

        # Panel label
        ax.text(0.03, 0.97, f'({panel_labels5[idx]})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

        if row == 0:
            ax.set_title(wtag, fontsize=11, fontweight='bold', pad=8)
        if col == 0:
            ax.text(-0.18, 0.5, vname, transform=ax.transAxes, rotation=90,
                    fontsize=11, fontweight='bold', va='center', ha='center')

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--',
                          alpha=0.4, color='grey')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(55, 110, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 55, 10))
        if col != 0:
            gl.left_labels = False
        if row != 1:
            gl.bottom_labels = False

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cb.set_label('Bias (Spire − ERA5) [K]')

fig.suptitle('Spire AI-S2S | Mean Bias (Spire − ERA5) — JFM 2026 (90 inits)',
             fontsize=13, fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.15, wspace=0.15, bottom=0.10, top=0.93, left=0.10)
savefig(fig, 'fig05_bias_maps')

# ====================================================================
# Summary table
# ====================================================================
print('\n' + '=' * 65)
print(f'{"Figure":<35} {"Size (KB)":>10}  {"Path"}')
print('-' * 65)
for p in sorted(created_files):
    sz = p.stat().st_size / 1024
    print(f'{p.name:<35} {sz:>8.1f}   {p}')
print('=' * 65)
print(f'\nAll {len(created_files)} files written to {OUT_DIR}')
print('Done ✓')
