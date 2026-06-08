"""
fig_z500_mjo_composite.py
=========================
Publication-quality Z500 anomaly composite maps conditioned on MJO phase.

Produces Figure 19:
  Z500 anomaly composites at Week 2 (d8-14, ~10-day lag) for the four
  most-populated active MJO phases during JFM 2026.

Layout:
  2 rows × 4 columns
  Top row    — ERA5  Z500 anomaly composite  (verification)
  Bottom row — Spire Z500 anomaly composite  (forecast)
  Columns    — 4 MJO phases (ordered by population, largest first)

Data:
  ../spire_era5/s2s_verification/weekly_anomalies_v2.nc
      spire_z500_anom, era5_z500_anom  (90 inits × 6 weeks × 101 lat × 101 lon)
      Domain: 0-50°N, 55-105°E  (India/South Asia), 0.5° resolution

  data/rmm.74toRealtime.txt
      Observed RMM1/RMM2/phase/amplitude for MJO classification
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
import matplotlib.patheffects as pe
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
ANOM_FILE = BASE.parent / "spire_era5" / "s2s_verification" / "weekly_anomalies_v2.nc"
RMM_FILE = DATA_DIR / "rmm.74toRealtime.txt"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = FIG_DIR / "fig19_z500_mjo_composite.png"

# ── Cartopy features ─────────────────────────────────────────────────────────
proj = ccrs.PlateCarree()
borders = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '50m',
    edgecolor='0.3', facecolor='none', linewidth=0.5,
)
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '50m',
    edgecolor='0.3', facecolor='none', linewidth=0.5,
)

# ── Load anomaly data ────────────────────────────────────────────────────────
print("Loading anomaly data …")
ds = xr.open_dataset(ANOM_FILE)

init_times = pd.DatetimeIndex(ds['init_time'].values)
lat = ds['latitude'].values
lon = ds['longitude'].values
n_inits = ds.sizes['init_time']

# Week 2 = index 1 (weeks are 1-indexed in the data, but stored 0-indexed in array)
WEEK_IDX = 1  # Week 2 → d8-14 (~10-day lag)

spire_z500_w2 = ds['spire_z500_anom'].values[:, WEEK_IDX, :, :]  # (90, 101, 101)
era5_z500_w2 = ds['era5_z500_anom'].values[:, WEEK_IDX, :, :]    # (90, 101, 101)

print(f"  Loaded {n_inits} inits, lat {lat[0]:.1f}-{lat[-1]:.1f}°N, "
      f"lon {lon[0]:.1f}-{lon[-1]:.1f}°E")
print(f"  Week 2 (d8-14) Z500 anomaly: Spire {spire_z500_w2.shape}, ERA5 {era5_z500_w2.shape}")

# ── Load & parse RMM data ────────────────────────────────────────────────────
print("\nLoading RMM index …")
rmm = pd.read_csv(
    RMM_FILE, skiprows=2, sep=r'\s+',
    names=['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'source'],
    dtype={'year': int, 'month': int, 'day': int},
)
rmm['date'] = pd.to_datetime(rmm[['year', 'month', 'day']])
rmm = rmm.set_index('date')

# Match each init date to its RMM phase/amplitude
init_phases = []
init_amplitudes = []
for dt in init_times:
    date_key = dt.normalize()
    if date_key in rmm.index:
        row = rmm.loc[date_key]
        # handle duplicate dates: take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        init_phases.append(int(row['phase']))
        init_amplitudes.append(float(row['amplitude']))
    else:
        init_phases.append(0)
        init_amplitudes.append(0.0)

init_phases = np.array(init_phases)
init_amplitudes = np.array(init_amplitudes)

# Active MJO mask
active_mask = init_amplitudes > 1.0
print(f"  Active MJO inits: {active_mask.sum()} / {n_inits}")

# Phase counts (active only)
phase_counts = {}
for ph in range(1, 9):
    mask = (init_phases == ph) & active_mask
    phase_counts[ph] = mask.sum()
    print(f"  Phase {ph}: {mask.sum()} inits")

# Select top 4 phases by count
top_phases = sorted(phase_counts, key=phase_counts.get, reverse=True)[:4]
top_phases.sort()  # Sort numerically for cleaner display
print(f"\n  Selected phases: {top_phases}")

# ── Compute composites ───────────────────────────────────────────────────────
print("\nComputing composites …")

composites_spire = {}
composites_era5 = {}

for ph in top_phases:
    mask = (init_phases == ph) & active_mask
    n_ph = mask.sum()

    sp_composite = np.nanmean(spire_z500_w2[mask], axis=0)
    e5_composite = np.nanmean(era5_z500_w2[mask], axis=0)

    composites_spire[ph] = sp_composite
    composites_era5[ph] = e5_composite

    sp_range = (np.nanmin(sp_composite), np.nanmax(sp_composite))
    e5_range = (np.nanmin(e5_composite), np.nanmax(e5_composite))
    print(f"  Phase {ph} ({n_ph} inits): "
          f"Spire [{sp_range[0]:.1f}, {sp_range[1]:.1f}] m, "
          f"ERA5 [{e5_range[0]:.1f}, {e5_range[1]:.1f}] m")

# ── Determine symmetric contour levels ───────────────────────────────────────
all_vals = np.concatenate([
    np.concatenate([composites_spire[ph].ravel() for ph in top_phases]),
    np.concatenate([composites_era5[ph].ravel() for ph in top_phases]),
])
vmax_data = np.nanpercentile(np.abs(all_vals), 98)
# Round up to nearest 5 m
vmax = max(5, int(np.ceil(vmax_data / 5)) * 5)
levels = np.linspace(-vmax, vmax, 21)
print(f"\n  Contour range: ±{vmax} m ({len(levels)} levels)")

# ── Pattern correlation (Spire vs ERA5) per phase ────────────────────────────
print("\nPattern correlations (Spire composite vs ERA5 composite):")
for ph in top_phases:
    sp = composites_spire[ph].ravel()
    e5 = composites_era5[ph].ravel()
    valid = ~(np.isnan(sp) | np.isnan(e5))
    if valid.sum() > 10:
        r, pval = stats.pearsonr(sp[valid], e5[valid])
        print(f"  Phase {ph}: r = {r:.3f} (p = {pval:.2e})")
    else:
        r = np.nan
        print(f"  Phase {ph}: insufficient data")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\nPlotting figure …")

n_cols = len(top_phases)
fig, axes = plt.subplots(
    2, n_cols, figsize=(4.0 * n_cols, 8),
    subplot_kw={'projection': proj},
)

# Row labels
row_labels = ['ERA5 (Observed)', 'Spire AI-S2S (Forecast)']
panel_letters = list('abcdefgh')

cmap = plt.cm.RdBu_r

for col, ph in enumerate(top_phases):
    n_ph = phase_counts[ph]

    for row, (data_dict, label) in enumerate([
        (composites_era5, row_labels[0]),
        (composites_spire, row_labels[1]),
    ]):
        ax = axes[row, col]
        ax.set_extent([55, 105, 0, 50], crs=proj)
        ax.add_feature(coastline)
        ax.add_feature(borders)

        composite = data_dict[ph]

        # Filled contours
        cf = ax.contourf(
            lon, lat, composite, levels=levels,
            cmap=cmap, extend='both', transform=proj,
        )

        # Thin contour lines for structure (every 10 m)
        contour_levels = np.arange(-vmax, vmax + 1, 10)
        contour_levels = contour_levels[contour_levels != 0]  # skip zero
        cs = ax.contour(
            lon, lat, composite, levels=contour_levels,
            colors='0.3', linewidths=0.4, transform=proj,
        )

        # Zero contour
        ax.contour(
            lon, lat, composite, levels=[0],
            colors='k', linewidths=0.7, linestyles='--', transform=proj,
        )

        # Panel letter
        idx = row * n_cols + col
        ax.text(
            0.03, 0.97, f'({panel_letters[idx]})',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='left',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
        )

        # Domain-mean absolute anomaly annotation
        mean_abs = np.nanmean(np.abs(composite))
        ax.text(
            0.97, 0.03, f'|Z500\'| = {mean_abs:.1f} m',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85),
        )

        # Column title (top row only)
        if row == 0:
            ax.set_title(
                f'MJO Phase {ph}\n({n_ph} inits)',
                fontsize=12, fontweight='bold', pad=8,
            )

        # Row label (first column only)
        if col == 0:
            ax.text(
                -0.18, 0.5, label,
                transform=ax.transAxes, rotation=90,
                fontsize=11, fontweight='bold', va='center', ha='center',
            )

        # Gridlines
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, linestyle='--',
            alpha=0.4, color='grey',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(55, 110, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 55, 10))
        if col != 0:
            gl.left_labels = False
        if row != 1:
            gl.bottom_labels = False

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    'Spire AI-S2S | Z500 Anomaly Composites by MJO Phase — Week 2 (d8-14)\n'
    'India Domain (0-50°N, 55-105°E) · JFM 2026 · Active MJO (amp > 1)',
    fontsize=13, fontweight='bold', y=0.98,
)

# ── Shared colorbar ──────────────────────────────────────────────────────────
fig.subplots_adjust(
    hspace=0.12, wspace=0.12,
    bottom=0.10, top=0.88, left=0.10, right=0.92,
)

# Colorbar below the figure
cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
cb = fig.colorbar(
    cf, cax=cbar_ax, orientation='horizontal',
    label='Z500 Anomaly (m)',
)
cb.ax.tick_params(labelsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_FILE)
plt.close(fig)
sz_kb = OUT_FILE.stat().st_size / 1024
print(f"\n  Saved: {OUT_FILE}")
print(f"  Size:  {sz_kb:.1f} KB")

# ── Also generate a compact summary panel: pattern correlation bar chart ─────
print("\nGenerating supplemental pattern correlation summary …")

fig2, ax2 = plt.subplots(figsize=(6, 3.5))

corrs = []
for ph in top_phases:
    sp = composites_spire[ph].ravel()
    e5 = composites_era5[ph].ravel()
    valid = ~(np.isnan(sp) | np.isnan(e5))
    if valid.sum() > 10:
        r, _ = stats.pearsonr(sp[valid], e5[valid])
    else:
        r = np.nan
    corrs.append(r)

colors = ['#2196F3' if r > 0.5 else '#FF9800' if r > 0.3 else '#F44336' for r in corrs]
bars = ax2.bar(
    [f'Phase {ph}\n({phase_counts[ph]} inits)' for ph in top_phases],
    corrs, color=colors, edgecolor='0.3', linewidth=0.8, width=0.55,
)

# Value labels
for bar, r in zip(bars, corrs):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f'{r:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
    )

ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(0.5, color='green', lw=1.0, ls='--', alpha=0.5, label='r = 0.5')
ax2.set_ylabel('Pattern Correlation (r)')
ax2.set_ylim(-0.3, 1.05)
ax2.set_title(
    'Spire vs ERA5 | Z500 Composite Pattern Correlation\n'
    'Week 2 (d8-14), Active MJO — JFM 2026',
    fontweight='bold',
)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, axis='y', alpha=0.3)

fig2.tight_layout()
supp_file = FIG_DIR / "fig19b_z500_mjo_pattern_corr.png"
fig2.savefig(supp_file)
plt.close(fig2)
sz_kb2 = supp_file.stat().st_size / 1024
print(f"  Saved: {supp_file}")
print(f"  Size:  {sz_kb2:.1f} KB")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Figure':<45} {'Size (KB)':>10}")
print("-" * 65)
for p in [OUT_FILE, supp_file]:
    sz = p.stat().st_size / 1024
    print(f"{p.name:<45} {sz:>8.1f}")
print("=" * 65)
print("\nDone ✓")
