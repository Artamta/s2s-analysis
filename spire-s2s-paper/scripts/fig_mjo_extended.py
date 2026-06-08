"""
fig_mjo_extended.py
====================
Publication-quality MJO/OLR/U850/U200 diagnostics for Spire AI-S2S paper.

Produces Figures 11-17:
  11 — OLR/U850/U200 spatial climatology maps (what does Spire forecast?)
  12 — OLR/U850/U200 spread (inter-init variability) vs lead
  13 — Spatial variance maps: OLR/U850/U200 (3x3: var × week)
  14 — OLR Hovmöller (composite and individual init examples)
  15 — U850 Hovmöller
  16 — MJO propagation: equatorial OLR + U850 combined Hovmöller
  17 — Extended scorecard heatmap (all 7 vars × 6 weeks, ACC from original dataset)

Uses:
  data/weekly_anomalies_extended.nc   (Spire OLR/U850/U200)
  data/equatorial_hovmoller.nc        (equatorial Hovmöller)
  ../spire_era5/s2s_verification/weekly_anomalies_v2.nc  (T2m/Precip/Z500)
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Style ─────────────────────────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
ORIG_DATA = BASE.parent / "spire_era5" / "s2s_verification" / "weekly_anomalies_v2.nc"
FIG_DIR  = BASE / "figures" / "mjo"
FIG_DIR.mkdir(parents=True, exist_ok=True)

created_files = []

def savefig(fig, name):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p)
    created_files.append(p)
    plt.close(fig)
    print(f"  Saved {name}.png")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
ds = xr.open_dataset(DATA_DIR / "weekly_anomalies_extended.nc")
ds_orig = xr.open_dataset(ORIG_DATA)

has_hov = (DATA_DIR / "equatorial_hovmoller.nc").exists()
if has_hov:
    ds_hov = xr.open_dataset(DATA_DIR / "equatorial_hovmoller.nc")
    print(f"  Hovmöller loaded: {list(ds_hov.data_vars)}")

lat = ds["latitude"].values
lon = ds["longitude"].values
n_weeks = ds.sizes['week']
n_inits = ds.sizes['init_time']
weeks = ds['week'].values
week_labels = [f'W{w}\n(d{(w-1)*7+1}-{w*7})' for w in weeks]

IND_LAT = (lat >= 6) & (lat <= 36)
IND_LON = (lon >= 68) & (lon <= 98)

def india_mean(field_2d):
    return np.nanmean(field_2d[np.ix_(IND_LAT, IND_LON)])

COLORS = {'OLR': '#E74C3C', 'U850': '#2ECC71', 'U200': '#9B59B6'}
MARKERS = {'OLR': 'o', 'U850': 's', 'U200': '^'}

# Cartopy features
proj = ccrs.PlateCarree()
borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land',
                                        '50m', edgecolor='0.3', facecolor='none', linewidth=0.5)
coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                          edgecolor='0.3', facecolor='none', linewidth=0.5)

# ====================================================================
# FIG 11 — OLR/U850/U200 spatial climatology (init-mean, W1 vs W3 vs W6)
# ====================================================================
print('\n── Figure 11: MJO variable climatology maps ──')

clim_vars = {'OLR': 'olr_wk_climatology', 'U850': 'u850_wk_climatology', 'U200': 'u200_wk_climatology'}
clim_units = {'OLR': 'W m⁻²', 'U850': 'm s⁻¹', 'U200': 'm s⁻¹'}
clim_cmaps = {'OLR': 'YlOrRd', 'U850': 'RdBu_r', 'U200': 'RdBu_r'}
clim_ranges = {'OLR': (100, 350), 'U850': (-10, 10), 'U200': (-60, 60)}

map_weeks = [0, 2, 5]
week_tags = ['W1 (d1-7)', 'W3 (d15-21)', 'W6 (d36-42)']

fig, axes = plt.subplots(3, 3, figsize=(11, 10), subplot_kw={'projection': proj})
panel_labels = list('abcdefghi')

cf_by_row = {}

for row, (vname, var_key) in enumerate(clim_vars.items()):
    data = ds[var_key].values  # (6, lat, lon)
    cmap = clim_cmaps[vname]
    vmin, vmax = clim_ranges[vname]
    
    for col, (wi, wtag) in enumerate(zip(map_weeks, week_tags)):
        ax = axes[row, col]
        ax.set_extent([55, 105, 0, 50], crs=proj)
        ax.add_feature(coastline)
        
        d = data[wi]
        if vname == 'OLR':
            levels = np.linspace(vmin, vmax, 13)
        else:
            levels = np.linspace(vmin, vmax, 11)
        
        cf = ax.contourf(lon, lat, d, levels=levels,
                         cmap=cmap, extend='both', transform=proj)
        cf_by_row[row] = cf
        
        im = india_mean(d)
        ax.text(0.97, 0.03, f'{im:.1f} {clim_units[vname]}',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85))
        
        idx = row * 3 + col
        ax.text(0.03, 0.97, f'({panel_labels[idx]})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')
        
        if row == 0:
            ax.set_title(wtag, fontsize=11, fontweight='bold', pad=8)
        if col == 0:
            ax.text(-0.18, 0.5, vname, transform=ax.transAxes, rotation=90,
                    fontsize=11, fontweight='bold', va='center', ha='center')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--', alpha=0.4, color='grey')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(55, 110, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 55, 10))
        if col != 0: gl.left_labels = False
        if row != 2: gl.bottom_labels = False

fig.suptitle('Spire AI-S2S | Mean Forecast Fields — MJO Variables\nJFM 2026 (90-init mean)',
             fontsize=13, fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.15, wspace=0.15, bottom=0.06, top=0.92, left=0.08, right=0.85)

for row, (vname, var_key) in enumerate(clim_vars.items()):
    pos = axes[row, 2].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(cf_by_row[row], cax=cax, orientation='vertical')
    cb.set_label(f'{vname} ({clim_units[vname]})', fontsize=9)
    cb.ax.tick_params(labelsize=8)

savefig(fig, 'fig11_mjo_climatology_maps')

# ====================================================================
# FIG 12 — Inter-init spread (std dev) vs lead time
# ====================================================================
print('\n── Figure 12: Inter-init spread vs lead ──')

spread_vars = {
    'OLR': 'spire_olr_anom',
    'U850': 'spire_u850_anom',
    'U200': 'spire_u200_anom',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel (a): OLR spread
olr_spread = []
for wi in range(n_weeks):
    std_map = ds['spire_olr_anom'].values[:, wi, :, :].std(axis=0)
    olr_spread.append(india_mean(std_map))
ax1.plot(range(n_weeks), olr_spread, 'o-', color=COLORS['OLR'], lw=2, ms=7, label='OLR')
ax1.set_xticks(range(n_weeks))
ax1.set_xticklabels(week_labels)
ax1.set_xlabel('Forecast Lead')
ax1.set_ylabel('Inter-init Std Dev (W m⁻²)')
ax1.set_title('(a) OLR Forecast Spread', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel (b): U850/U200 spread
u850_spread, u200_spread = [], []
for wi in range(n_weeks):
    u850_spread.append(india_mean(ds['spire_u850_anom'].values[:, wi, :, :].std(axis=0)))
    u200_spread.append(india_mean(ds['spire_u200_anom'].values[:, wi, :, :].std(axis=0)))
ax2.plot(range(n_weeks), u850_spread, 's-', color=COLORS['U850'], lw=2, ms=7, label='U850')
ax2.plot(range(n_weeks), u200_spread, '^-', color=COLORS['U200'], lw=2, ms=7, label='U200')
ax2.set_xticks(range(n_weeks))
ax2.set_xticklabels(week_labels)
ax2.set_xlabel('Forecast Lead')
ax2.set_ylabel('Inter-init Std Dev (m s⁻¹)')
ax2.set_title('(b) Wind Forecast Spread', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle('Spire AI-S2S | Forecast Signal Spread vs Lead — MJO Variables',
             fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'fig12_mjo_spread_vs_lead')

# ====================================================================
# FIG 13 — Spatial variance maps (3 vars × 3 weeks)
# ====================================================================
print('\n── Figure 13: Spatial variance maps ──')

fig, axes = plt.subplots(3, 3, figsize=(11, 10), subplot_kw={'projection': proj})

var_data = {
    'OLR': ds['spire_olr_anom'].values,
    'U850': ds['spire_u850_anom'].values,
    'U200': ds['spire_u200_anom'].values,
}
var_units = {'OLR': 'W m⁻²', 'U850': 'm s⁻¹', 'U200': 'm s⁻¹'}
var_cmaps = {'OLR': 'YlOrRd', 'U850': 'YlOrRd', 'U200': 'YlOrRd'}
var_vmax = {'OLR': 40, 'U850': 4, 'U200': 15}

cf_by_row = {}

for row, vname in enumerate(var_data.keys()):
    data = var_data[vname]  # (90, 6, lat, lon)
    for col, (wi, wtag) in enumerate(zip(map_weeks, week_tags)):
        ax = axes[row, col]
        ax.set_extent([55, 105, 0, 50], crs=proj)
        ax.add_feature(coastline)
        
        # Standard deviation across inits
        std_map = data[:, wi, :, :].std(axis=0)
        vmax = var_vmax[vname]
        cf = ax.contourf(lon, lat, std_map,
                         levels=np.linspace(0, vmax, 11),
                         cmap=var_cmaps[vname], extend='max', transform=proj)
        cf_by_row[row] = cf
        
        im = india_mean(std_map)
        ax.text(0.97, 0.03, f'σ={im:.1f}', transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85))
        
        idx = row * 3 + col
        ax.text(0.03, 0.97, f'({panel_labels[idx]})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')
        
        if row == 0:
            ax.set_title(wtag, fontsize=11, fontweight='bold', pad=8)
        if col == 0:
            ax.text(-0.18, 0.5, vname,
                    transform=ax.transAxes, rotation=90,
                    fontsize=11, fontweight='bold', va='center', ha='center')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--', alpha=0.4, color='grey')
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(55, 110, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 55, 10))
        if col != 0: gl.left_labels = False
        if row != 2: gl.bottom_labels = False

fig.suptitle('Spire AI-S2S | Forecast Variability (σ across 90 inits)\nMJO-Relevant Variables — JFM 2026',
             fontsize=13, fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.15, wspace=0.15, bottom=0.06, top=0.92, left=0.08, right=0.85)

for row, vname in enumerate(var_data.keys()):
    pos = axes[row, 2].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(cf_by_row[row], cax=cax, orientation='vertical')
    cb.set_label(f'{vname} ({var_units[vname]})', fontsize=9)
    cb.ax.tick_params(labelsize=8)

savefig(fig, 'fig13_mjo_variance_maps')

# ====================================================================
# FIG 14 — OLR Hovmöller
# ====================================================================
if has_hov:
    print('\n── Figure 14: OLR Hovmöller ──')
    
    sp_olr = ds_hov['spire_olr_eq'].values  # (90, 42, lon)
    hov_lon = ds_hov['longitude'].values
    n_steps = ds_hov.sizes['step']
    
    # Anomaly from temporal mean
    sp_olr_mean = np.nanmean(sp_olr, axis=(0, 1))
    sp_olr_anom = sp_olr - sp_olr_mean[np.newaxis, np.newaxis, :]
    
    # Indian-Pacific sector
    lon_mask = (hov_lon >= 40) & (hov_lon <= 180)
    hov_lon_sub = hov_lon[lon_mask]
    
    # 3 panels: composite + 2 example inits
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    
    # Composite (all 90 inits)
    composite = np.nanmean(sp_olr_anom[:, :, lon_mask], axis=0)
    cf = axes[0].contourf(hov_lon_sub, np.arange(1, n_steps+1), composite,
                          levels=np.linspace(-20, 20, 11), cmap='RdBu_r', extend='both')
    axes[0].set_title('(a) Composite\n(90-init mean)', fontweight='bold')
    axes[0].set_ylabel('Lead Day')
    
    # Strong MJO init — find init with max equatorial variance
    eq_var = np.nanvar(sp_olr_anom[:, :, lon_mask], axis=(1, 2))
    top_init = np.argsort(eq_var)[-1]
    init_date1 = pd.Timestamp(ds_hov['init_time'].values[top_init]).strftime('%Y-%m-%d')
    axes[1].contourf(hov_lon_sub, np.arange(1, n_steps+1),
                     sp_olr_anom[top_init, :, :][:, lon_mask],
                     levels=np.linspace(-40, 40, 11), cmap='RdBu_r', extend='both')
    axes[1].set_title(f'(b) Active MJO Init\n({init_date1})', fontweight='bold')
    
    # Quiet init — lowest variance
    quiet_init = np.argsort(eq_var)[0]
    init_date2 = pd.Timestamp(ds_hov['init_time'].values[quiet_init]).strftime('%Y-%m-%d')
    cf2 = axes[2].contourf(hov_lon_sub, np.arange(1, n_steps+1),
                           sp_olr_anom[quiet_init, :, :][:, lon_mask],
                           levels=np.linspace(-40, 40, 11), cmap='RdBu_r', extend='both')
    axes[2].set_title(f'(c) Quiet Init\n({init_date2})', fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('Longitude (°E)')
        ax.axvline(55, ls='--', lw=0.8, color='grey', alpha=0.5)
        ax.axvline(105, ls='--', lw=0.8, color='grey', alpha=0.5)
        ax.text(80, n_steps-1, 'India', fontsize=8, ha='center', color='0.4',
                bbox=dict(fc='white', ec='none', alpha=0.7))
        ax.invert_yaxis()
        # Week markers
        for w in range(1, 7):
            ax.axhline(w*7, ls=':', lw=0.5, color='grey', alpha=0.4)
    
    
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(cf, cax=cbar_ax, label='OLR anomaly (W m⁻²)')
    
    fig.suptitle('Spire AI-S2S | Equatorial OLR Hovmöller (15°S-15°N)\nJFM 2026',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(right=0.90, wspace=0.08, top=0.83)
    savefig(fig, 'fig14_hovmoller_olr')
    
    # ====================================================================
    # FIG 15 — U850 Hovmöller
    # ====================================================================
    print('\n── Figure 15: U850 Hovmöller ──')
    
    sp_u850 = ds_hov['spire_u850_eq'].values
    sp_u850_mean = np.nanmean(sp_u850, axis=(0, 1))
    sp_u850_anom = sp_u850 - sp_u850_mean[np.newaxis, np.newaxis, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    
    composite_u = np.nanmean(sp_u850_anom[:, :, lon_mask], axis=0)
    cf = axes[0].contourf(hov_lon_sub, np.arange(1, n_steps+1), composite_u,
                          levels=np.linspace(-3, 3, 13), cmap='RdBu_r', extend='both')
    axes[0].set_title('(a) Composite\n(90-init mean)', fontweight='bold')
    axes[0].set_ylabel('Lead Day')
    
    u_var = np.nanvar(sp_u850_anom[:, :, lon_mask], axis=(1, 2))
    top_u = np.argsort(u_var)[-1]
    init_u1 = pd.Timestamp(ds_hov['init_time'].values[top_u]).strftime('%Y-%m-%d')
    axes[1].contourf(hov_lon_sub, np.arange(1, n_steps+1),
                     sp_u850_anom[top_u, :, :][:, lon_mask],
                     levels=np.linspace(-8, 8, 11), cmap='RdBu_r', extend='both')
    axes[1].set_title(f'(b) Active Init\n({init_u1})', fontweight='bold')
    
    quiet_u = np.argsort(u_var)[0]
    init_u2 = pd.Timestamp(ds_hov['init_time'].values[quiet_u]).strftime('%Y-%m-%d')
    axes[2].contourf(hov_lon_sub, np.arange(1, n_steps+1),
                     sp_u850_anom[quiet_u, :, :][:, lon_mask],
                     levels=np.linspace(-8, 8, 11), cmap='RdBu_r', extend='both')
    axes[2].set_title(f'(c) Quiet Init\n({init_u2})', fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('Longitude (°E)')
        ax.axvline(55, ls='--', lw=0.8, color='grey', alpha=0.5)
        ax.axvline(105, ls='--', lw=0.8, color='grey', alpha=0.5)
        ax.invert_yaxis()
        for w in range(1, 7):
            ax.axhline(w*7, ls=':', lw=0.5, color='grey', alpha=0.4)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='U850 anomaly (m s⁻¹)')
    
    fig.suptitle('Spire AI-S2S | Equatorial U850 Hovmöller (15°S-15°N)\nJFM 2026',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(right=0.90, wspace=0.08, top=0.83)
    savefig(fig, 'fig15_hovmoller_u850')
    
    # ====================================================================
    # FIG 16 — Combined MJO propagation (OLR + U850 overlaid)
    # ====================================================================
    print('\n── Figure 16: Combined MJO propagation ──')
    
    # Inter-init consistency: correlation between pairs of inits
    # For each lead day, how similar are different init forecasts in the equatorial band?
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Compute signal-to-noise ratio per lead day (equatorial band)
    for vname, data, color, marker in [
        ('OLR', sp_olr[:, :, lon_mask], COLORS['OLR'], 'o'),
        ('U850', sp_u850[:, :, lon_mask], COLORS['U850'], 's'),
    ]:
        snr = []
        for s in range(n_steps):
            signal = np.nanstd(np.nanmean(data[:, s, :], axis=1))  # std of init-means
            noise = np.nanmean(np.nanstd(data[:, s, :], axis=0))   # mean of spatial stds
            snr.append(signal / max(noise, 1e-10))
        
        snr_arr = np.array(snr)
        ax.plot(np.arange(1, n_steps+1), snr_arr, color=color, alpha=0.3, lw=0.8)
        # Weekly smoothed
        weekly = [np.mean(snr_arr[i:i+7]) for i in range(0, 42, 7)]
        ax.plot(np.arange(3.5, 42, 7), weekly, marker=marker, color=color,
                lw=2, ms=8, label=vname, zorder=5)
    
    ax.axhline(1.0, ls='--', lw=0.8, color='grey', alpha=0.6, label='SNR = 1')
    ax.set_xlabel('Lead Day')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('Spire AI-S2S | MJO Signal-to-Noise Ratio\nEquatorial band (15°S-15°N, 40°-180°E)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(1, 42)
    ax.grid(True, alpha=0.3)
    for w in range(6):
        ax.axvspan(w*7+1, (w+1)*7, alpha=0.03, color='grey' if w % 2 else 'white')
        ax.text(w*7+4, ax.get_ylim()[0] + 0.02, f'W{w+1}', ha='center', fontsize=8, color='0.5')
    
    fig.tight_layout()
    savefig(fig, 'fig16_mjo_snr')

# ====================================================================
# FIG 17 — Extended scorecard (7 vars × 6 weeks)
# ====================================================================
print('\n── Figure 17: Extended scorecard ──')

# Compute ACC for original variables (T2m/Precip/Z500)
VARS_ORIG = {
    'T2m-mean': ('spire_t2m_mean_anom', 'era5_t2m_mean_anom'),
    'T2m-max':  ('spire_t2m_max_anom',  'era5_t2m_max_anom'),
    'Precip':   ('spire_precip_anom',   'era5_precip_anom'),
    'Z500':     ('spire_z500_anom',     'era5_z500_anom'),
}

lat_o = ds_orig["latitude"].values
lon_o = ds_orig["longitude"].values
IND_LAT_O = (lat_o >= 6) & (lat_o <= 36)
IND_LON_O = (lon_o >= 68) & (lon_o <= 98)

acc_all = {}
for vname, (sp_var, e5_var) in VARS_ORIG.items():
    sp = ds_orig[sp_var].values
    e5 = ds_orig[e5_var].values
    acc_w = []
    for wi in range(n_weeks):
        acc_map = np.full((len(lat_o), len(lon_o)), np.nan)
        for j in range(len(lat_o)):
            for k in range(len(lon_o)):
                s_ = sp[:, wi, j, k]
                e_ = e5[:, wi, j, k]
                mask = ~(np.isnan(s_) | np.isnan(e_))
                if mask.sum() > 10:
                    r, _ = stats.pearsonr(s_[mask], e_[mask])
                    acc_map[j, k] = r
        acc_w.append(np.nanmean(acc_map[np.ix_(IND_LAT_O, IND_LON_O)]))
    acc_all[vname] = np.array(acc_w)
    print(f"  {vname}: W1={acc_w[0]:.3f} W6={acc_w[5]:.3f}")

# For MJO vars: compute inter-init signal variance / total variance as proxy
# (No ERA5 reference, so true ACC not computable)
for vname, var_key in [('OLR', 'spire_olr_anom'), ('U850', 'spire_u850_anom'), ('U200', 'spire_u200_anom')]:
    data = ds[var_key].values  # (90, 6, lat, lon)
    sv = []
    for wi in range(n_weeks):
        wk = data[:, wi, :, :]
        # Normalized variance: what fraction of total spread is signal vs noise
        # Signal = std of init-means (init-to-init difference)
        # Use R² of first 2 PCs as proxy for forecast skill
        wk_india = wk[:, IND_LAT, :][:, :, IND_LON].reshape(n_inits, -1)
        # Simple: inter-init correlation of spatial patterns
        # Take correlation between consecutive init pairs
        corrs = []
        for i in range(0, n_inits-1, 2):
            r, _ = stats.pearsonr(wk_india[i].flatten(), wk_india[i+1].flatten())
            corrs.append(r)
        sv.append(np.nanmean(corrs))
    acc_all[vname] = np.array(sv)
    print(f"  {vname} (proxy): W1={sv[0]:.3f} W6={sv[5]:.3f}")

# Build matrix
all_vars = ['T2m-mean', 'T2m-max', 'Precip', 'Z500', 'OLR', 'U850', 'U200']
acc_matrix = np.array([acc_all[v] for v in all_vars])

fig, ax = plt.subplots(figsize=(8, 5))

cmap = plt.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.3, vmax=1.0)
im = ax.imshow(acc_matrix, cmap=cmap, norm=norm, aspect='auto')

for i in range(len(all_vars)):
    for j in range(n_weeks):
        val = acc_matrix[i, j]
        color = 'white' if (val < 0.1 or val > 0.85) else 'black'
        suffix = '' if i < 4 else '*'
        ax.text(j, i, f'{val:.2f}{suffix}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

ax.set_xticks(range(n_weeks))
ax.set_xticklabels([f'W{w}' for w in weeks])
ax.set_yticks(range(len(all_vars)))
ax.set_yticklabels(all_vars)
ax.set_xlabel('Forecast Lead Week')

ax.axhline(3.5, color='white', lw=3)
ax.text(n_weeks + 0.1, 1.5, 'ACC\n(vs ERA5)', fontsize=8, va='center', ha='left',
        color='0.4', fontstyle='italic')
ax.text(n_weeks + 0.1, 5.0, 'Init-pair\ncorrelation*', fontsize=8, va='center', ha='left',
        color='0.4', fontstyle='italic')

plt.colorbar(im, ax=ax, label='Skill Metric', shrink=0.8, pad=0.22)

ax.set_title('Spire AI-S2S | Extended Verification Scorecard\nIndia-mean — JFM 2026 (90 inits)',
             fontweight='bold', pad=12)
fig.tight_layout()
savefig(fig, 'fig17_extended_scorecard')

# ====================================================================
# Summary
# ====================================================================
print('\n' + '=' * 65)
print(f'{"Figure":<40} {"Size (KB)":>10}  Path')
print('-' * 65)
for p in sorted(created_files):
    sz = p.stat().st_size / 1024
    print(f'{p.name:<40} {sz:>8.1f}   {p}')
print('=' * 65)
print(f'\nAll {len(created_files)} files written to {FIG_DIR}')
print('Done ✓')
