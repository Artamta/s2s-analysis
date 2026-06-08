#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fig_mjo_phase_diagram.py
========================
Publication-quality MJO phase diagrams (Wheeler–Hendon RMM1 vs RMM2
phase-space plots) for the Spire AI-S2S paper.

Shows Spire AI-S2S 42-day forecast RMM trajectories vs observed RMM for two
selected initialisation dates with active MJO during JFM 2026.

Method:
  1. Load equatorial Hovmöller data (OLR, U850, U200) from Spire forecasts.
  2. Remove the "climatology" (mean across all 90 inits for each lead step).
  3. Interpolate from 0.5° → 1° to match Wheeler–Hendon EOF resolution.
  4. Normalise anomalies by standard deviation of each field.
  5. Project normalised anomalies onto combined EOF1/EOF2 to get RMM1/RMM2.
  6. Plot forecast trajectory vs observed trajectory in RMM phase space.

Author : Ayush Raj
Created: 2026-06-06
"""

# ── Imports & rcParams ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.linewidth': 0.8,
})

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
FIG_DIR  = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HOV_FILE  = DATA_DIR / "equatorial_hovmoller.nc"
EOF_FILE  = DATA_DIR / "eofs_MJO.nc"
RMM_FILE  = DATA_DIR / "rmm.74toRealtime.txt"
OUT_FILE  = FIG_DIR / "fig18_mjo_phase_diagram.png"

# ── Load observed RMM ─────────────────────────────────────────────────────
print("Loading observed RMM …")
rmm_cols = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'source']
rmm_df = pd.read_csv(RMM_FILE, skiprows=2, sep=r'\s+', names=rmm_cols,
                      dtype={'year': int, 'month': int, 'day': int})
rmm_df['date'] = pd.to_datetime(rmm_df[['year', 'month', 'day']])
rmm_df = rmm_df.set_index('date')

# Filter to JFM 2026 + buffer for 42-day trajectories starting in late March
rmm_jfm = rmm_df.loc['2026-01-01':'2026-05-15'].copy()
print(f"  Observed RMM records loaded: {len(rmm_jfm)}")

# ── Load EOF patterns ─────────────────────────────────────────────────────
print("Loading Wheeler–Hendon EOFs …")
ds_eof = xr.open_dataset(EOF_FILE)
eof_lon = ds_eof['longitude'].values  # 0, 1, 2, …, 359 (360 pts, 1° res)

# Build combined EOF vectors (OLR, U850, U200) – length 1080
eof1_combined = np.concatenate([
    ds_eof['eof1_olr'].values,
    ds_eof['eof1_u850'].values,
    ds_eof['eof1_u200'].values,
])
eof2_combined = np.concatenate([
    ds_eof['eof2_olr'].values,
    ds_eof['eof2_u850'].values,
    ds_eof['eof2_u200'].values,
])
print(f"  EOF combined vector length: {len(eof1_combined)}")

# ── Load equatorial Hovmöller data ────────────────────────────────────────
print("Loading equatorial Hovmöller data …")
ds_hov = xr.open_dataset(HOV_FILE)
init_times = pd.DatetimeIndex(ds_hov['init_time'].values)
steps = ds_hov['step'].values           # 1..42
hov_lon = ds_hov['longitude'].values    # 0.0, 0.5, 1.0, …, 359.5 (720 pts)

# Raw fields: (90 inits, 42 steps, 720 lons)
olr_raw  = ds_hov['spire_olr_eq'].values
u850_raw = ds_hov['spire_u850_eq'].values
u200_raw = ds_hov['spire_u200_eq'].values
print(f"  Hovmöller shape: {olr_raw.shape}")

# ── Compute anomalies & project onto EOFs ─────────────────────────────────
print("Computing RMM projections …")

# Step 1: Remove climatology (mean across all 90 inits for each step)
olr_clim  = np.nanmean(olr_raw,  axis=0, keepdims=True)
u850_clim = np.nanmean(u850_raw, axis=0, keepdims=True)
u200_clim = np.nanmean(u200_raw, axis=0, keepdims=True)

olr_anom  = olr_raw  - olr_clim
u850_anom = u850_raw - u850_clim
u200_anom = u200_raw - u200_clim

# Step 2: Interpolate 0.5° → 1° (select every other longitude, starting at 0°)
idx_1deg = np.arange(0, 720, 2)
olr_1deg  = olr_anom[:, :, idx_1deg]
u850_1deg = u850_anom[:, :, idx_1deg]
u200_1deg = u200_anom[:, :, idx_1deg]

# Verify longitude alignment
lons_selected = hov_lon[idx_1deg]
assert np.allclose(lons_selected, eof_lon), "Longitude grids do not align!"

# Step 3: Normalise anomalies per gridpoint
olr_std_gp  = np.nanstd(olr_1deg,  axis=(0, 1), keepdims=True)
u850_std_gp = np.nanstd(u850_1deg, axis=(0, 1), keepdims=True)
u200_std_gp = np.nanstd(u200_1deg, axis=(0, 1), keepdims=True)

olr_std_gp[olr_std_gp   < 1e-6] = 1.0
u850_std_gp[u850_std_gp < 1e-6] = 1.0
u200_std_gp[u200_std_gp < 1e-6] = 1.0

olr_norm  = olr_1deg  / olr_std_gp
u850_norm = u850_1deg / u850_std_gp
u200_norm = u200_1deg / u200_std_gp

print(f"  Per-gridpoint normalisation — OLR std range: "
      f"[{olr_std_gp.min():.3f}, {olr_std_gp.max():.3f}]")

# Step 4: Project onto EOFs → raw RMM1, RMM2 for all (init, step)
n_inits = olr_norm.shape[0]
n_steps = olr_norm.shape[1]

rmm1_raw = np.full((n_inits, n_steps), np.nan)
rmm2_raw = np.full((n_inits, n_steps), np.nan)

for i in range(n_inits):
    for s in range(n_steps):
        combined = np.concatenate([
            olr_norm[i, s, :],
            u850_norm[i, s, :],
            u200_norm[i, s, :],
        ])
        if np.any(np.isnan(combined)):
            continue
        rmm1_raw[i, s] = np.dot(combined, eof1_combined)
        rmm2_raw[i, s] = np.dot(combined, eof2_combined)

# Step 5: Normalise projected RMM to unit-variance scale
rmm1_sigma = np.nanstd(rmm1_raw)
rmm2_sigma = np.nanstd(rmm2_raw)
print(f"  Projection stds — RMM1: {rmm1_sigma:.3f}, RMM2: {rmm2_sigma:.3f}")

rmm1_fcst = rmm1_raw / rmm1_sigma
rmm2_fcst = rmm2_raw / rmm2_sigma

print(f"  Forecast RMM range — RMM1: [{np.nanmin(rmm1_fcst):.2f}, {np.nanmax(rmm1_fcst):.2f}], "
      f"RMM2: [{np.nanmin(rmm2_fcst):.2f}, {np.nanmax(rmm2_fcst):.2f}]")

# ── Select init dates with active MJO ─────────────────────────────────────
print("Selecting initialization dates …")

def find_best_init(month_range, min_amp=1.5):
    """Find init date within month_range with highest observed MJO amplitude."""
    best_amp = 0
    best_idx = None
    for idx, init_date in enumerate(init_times):
        if init_date.month not in month_range:
            continue
        date_str = pd.Timestamp(init_date)
        if date_str in rmm_jfm.index:
            amp = rmm_jfm.loc[date_str, 'amplitude']
            if amp > min_amp and amp > best_amp:
                best_amp = amp
                best_idx = idx
    return best_idx

idx_a = find_best_init([1, 2], min_amp=1.0)
idx_b = find_best_init([3], min_amp=1.0)

if idx_b is None:
    for idx, init_date in enumerate(init_times):
        if init_date.month == 3:
            idx_b = idx

print(f"  Panel (a): init = {init_times[idx_a].strftime('%Y-%m-%d')} "
      f"(amplitude = {rmm_jfm.loc[pd.Timestamp(init_times[idx_a]), 'amplitude']:.2f})")
print(f"  Panel (b): init = {init_times[idx_b].strftime('%Y-%m-%d')} "
      f"(amplitude = {rmm_jfm.loc[pd.Timestamp(init_times[idx_b]), 'amplitude']:.2f})")

# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

# ── Colours & constants ───────────────────────────────────────────────────
OBS_COLOR  = '#888888'          # mid-grey for analysis
FCST_COLOR = '#C0392B'          # elegant dark red for Spire
PHASE_LINE_COLOR = '#B0B0B0'   # light grey for phase boundary lines
CIRCLE_COLOR = '#333333'       # dark for unit circle
LIM = 4.0                      # axis limits

# Build a truncated OrRd colormap so early lead days are still visible
from matplotlib.colors import LinearSegmentedColormap
_base_cmap = plt.get_cmap('OrRd')
_colors = _base_cmap(np.linspace(0.25, 1.0, 256))  # skip the lightest 25%
FCST_CMAP_OBJ = LinearSegmentedColormap.from_list('OrRd_trunc', _colors)
FCST_CMAP_NAME = 'OrRd_trunc'
plt.colormaps.register(cmap=FCST_CMAP_OBJ, name=FCST_CMAP_NAME)

# ── Phase layout (Wheeler-Hendon convention) ──────────────────────────────
# Phase number positions at the boundary intersections, outside unit circle
PHASE_NUM_R = 3.35
PHASE_ANGLES = {
    5: 22.5,   6: 67.5,   7: 112.5,  8: 157.5,
    1: 202.5,  2: 247.5,  3: 292.5,  4: 337.5,
}

# Region text positions (centred in each octant, intermediate radius)
REGION_R = 2.25
REGION_LABELS = {
    5:  "Maritime\nContinent",
    6:  "Western\nPacific",
    7:  "Western\nPacific",
    8:  "West. Hem.\n& Africa",
    1:  "West. Hem.\n& Africa",
    2:  "Indian\nOcean",
    3:  "Indian\nOcean",
    4:  "Maritime\nContinent",
}


def draw_phase_background(ax):
    """Draw the canonical MJO phase diagram background."""
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect('equal')
    ax.set_xlabel('RMM1', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMM2', fontsize=13, fontweight='bold')

    # — Thin axis lines through origin —
    ax.axhline(0, color='#AAAAAA', linewidth=0.6, zorder=1)
    ax.axvline(0, color='#AAAAAA', linewidth=0.6, zorder=1)

    # — Phase boundary lines (8 sectors = 4 diagonal lines) —
    for angle_deg in [22.5, 67.5]:
        a = np.radians(angle_deg)
        x_end = LIM * np.cos(a)
        y_end = LIM * np.sin(a)
        ax.plot([-x_end, x_end], [-y_end, y_end],
                color=PHASE_LINE_COLOR, linewidth=1.0, linestyle='-', zorder=1)
        ax.plot([-y_end, y_end], [x_end, -x_end],
                color=PHASE_LINE_COLOR, linewidth=1.0, linestyle='-', zorder=1)

    # — Unit circle (1-σ amplitude threshold) —
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=CIRCLE_COLOR,
            linewidth=1.5, zorder=2)

    # — Phase number labels (boxed) —
    for phase, angle_deg in PHASE_ANGLES.items():
        a = np.radians(angle_deg)
        x = PHASE_NUM_R * np.cos(a)
        y = PHASE_NUM_R * np.sin(a)
        ax.text(x, y, str(phase), ha='center', va='center',
                fontsize=12, fontweight='bold', color='#222222',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='white',
                          edgecolor='#555555', linewidth=0.8))

    # — Region labels (italic, subtle) —
    for phase, label in REGION_LABELS.items():
        a = np.radians(PHASE_ANGLES[phase])
        x = REGION_R * np.cos(a)
        y = REGION_R * np.sin(a)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontstyle='italic', color='#666666',
                linespacing=1.1)

    # — Ticks —
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.tick_params(direction='in', length=4, width=0.6, colors='#555555')

    # — Subtle grid —
    ax.grid(False)


def get_obs_trajectory(init_date, n_days=43):
    """Get observed RMM trajectory from init_date for n_days."""
    obs_dates = pd.date_range(init_date, periods=n_days, freq='D')
    r1, r2 = [], []
    for d in obs_dates:
        if d in rmm_jfm.index:
            v1, v2 = rmm_jfm.loc[d, 'RMM1'], rmm_jfm.loc[d, 'RMM2']
            if hasattr(v1, '__len__'):
                v1, v2 = float(v1.iloc[0]), float(v2.iloc[0])
            r1.append(float(v1))
            r2.append(float(v2))
        else:
            r1.append(np.nan)
            r2.append(np.nan)
    return np.array(r1), np.array(r2)


def plot_trajectory(ax, rmm1, rmm2, color, linewidth=2.0, alpha=1.0,
                    marker_every=7, marker_size=5, zorder=5, label=None,
                    use_cmap=None, start_marker=True, show_week_labels=True):
    """Plot an MJO trajectory with weekly markers.
    
    If use_cmap is set, colour the trajectory by lead day.
    Otherwise use a solid colour.
    """
    valid = ~(np.isnan(rmm1) | np.isnan(rmm2))
    r1, r2 = rmm1[valid], rmm2[valid]
    n = len(r1)
    if n < 2:
        return

    if use_cmap is not None:
        # ── Gradient-coloured trajectory ──
        cmap = plt.get_cmap(use_cmap)
        norm = plt.Normalize(0, n - 1)

        # Line segments
        points = np.array([r1, r2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidth=linewidth, alpha=alpha, zorder=zorder,
                            capstyle='round', joinstyle='round')
        lc.set_array(np.arange(n - 1))
        ax.add_collection(lc)

        # Weekly markers (day 7, 14, 21, 28, 35, 42) with collision avoidance
        _placed_labels = []  # track (x, y) of placed label positions
        for d in range(marker_every - 1, n, marker_every):
            c = cmap(norm(d))
            ax.plot(r1[d], r2[d], 'o', color=c, markersize=marker_size + 2,
                    markeredgecolor='white', markeredgewidth=1.0,
                    zorder=zorder + 2)
            if show_week_labels:
                wk = (d + 1) // 7
                # Smart label offset: perpendicular to local trajectory
                if d > 0 and d < n - 1:
                    dx = r1[min(d+1, n-1)] - r1[max(d-1, 0)]
                    dy = r2[min(d+1, n-1)] - r2[max(d-1, 0)]
                else:
                    dx, dy = 1, 1
                mag = np.sqrt(dx**2 + dy**2) + 1e-10
                # Perpendicular (outward from origin)
                px, py = -dy / mag, dx / mag
                # Pick direction away from origin
                if px * r1[d] + py * r2[d] < 0:
                    px, py = -px, -py
                ofs = 12  # offset in points
                lx, ly = r1[d] + px * 0.25, r2[d] + py * 0.25  # approx position
                # Check collision with previously placed labels
                too_close = any(np.sqrt((lx - ox)**2 + (ly - oy)**2) < 0.35
                                for ox, oy in _placed_labels)
                if not too_close:
                    ax.annotate(f'W{wk}', (r1[d], r2[d]),
                                textcoords='offset points',
                                xytext=(ofs * px, ofs * py),
                                fontsize=7.5, fontweight='bold', color=c,
                                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                                zorder=zorder + 3)
                    _placed_labels.append((lx, ly))
    else:
        # ── Solid-colour trajectory ──
        ax.plot(r1, r2, '-', color=color, linewidth=linewidth,
                alpha=alpha, zorder=zorder, solid_capstyle='round')

        # Weekly markers (no labels for observed to reduce clutter)
        for d in range(marker_every - 1, n, marker_every):
            ax.plot(r1[d], r2[d], 'o', color=color, markersize=marker_size,
                    markeredgecolor='white', markeredgewidth=0.6,
                    zorder=zorder + 2)

    # Start marker (init day)
    if start_marker:
        ax.plot(r1[0], r2[0], '*', color=color if use_cmap is None else 'black',
                markersize=14, markeredgecolor='white', markeredgewidth=0.8,
                zorder=zorder + 4)


# ═══════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════════════
print("Generating figure …")

fig, axes = plt.subplots(1, 2, figsize=(14, 7.2),
                          gridspec_kw={'wspace': 0.32})

selected_inits = [(idx_a, '(a)'), (idx_b, '(b)')]

for panel_idx, (init_idx, panel_label) in enumerate(selected_inits):
    ax = axes[panel_idx]
    init_date = pd.Timestamp(init_times[init_idx])
    obs_phase = int(rmm_jfm.loc[init_date, 'phase'])
    obs_amp   = float(rmm_jfm.loc[init_date, 'amplitude'])

    # Draw background
    draw_phase_background(ax)

    # ── Observed trajectory (grey, no week labels) ──
    obs_r1, obs_r2 = get_obs_trajectory(init_date, n_days=43)
    plot_trajectory(ax, obs_r1, obs_r2, color=OBS_COLOR,
                    linewidth=3.5, alpha=0.60, marker_size=4.5,
                    zorder=4, label='Observed (analysis)',
                    show_week_labels=False)

    # ── Spire forecast trajectory (coloured by lead) ──
    fcst_r1 = rmm1_fcst[init_idx, :]
    fcst_r2 = rmm2_fcst[init_idx, :]
    plot_trajectory(ax, fcst_r1, fcst_r2, color=FCST_COLOR,
                    linewidth=2.8, alpha=0.95, marker_size=6,
                    zorder=6, label='Spire AI-S2S',
                    use_cmap=FCST_CMAP_NAME, start_marker=True,
                    show_week_labels=True)

    # Panel title with init-date info
    ax.set_title(
        f"{panel_label}  Init: {init_date.strftime('%d %b %Y')}"
        f"  (Phase {obs_phase}, amp = {obs_amp:.1f})",
        fontsize=12, fontweight='bold', pad=12)

# ── Suptitle ──────────────────────────────────────────────────────────────
fig.suptitle('Spire AI-S2S  |  MJO Phase Diagram — JFM 2026',
             fontsize=16, fontweight='bold', y=1.0)

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color=OBS_COLOR, linewidth=3.5, alpha=0.65,
           marker='o', markersize=5, markeredgecolor='white',
           label='Observed (analysis)'),
    Line2D([0], [0], color=FCST_COLOR, linewidth=2.5,
           marker='o', markersize=5, markeredgecolor='white',
           label='Spire AI-S2S forecast'),
    Line2D([0], [0], color='none', marker='*', markersize=12,
           markeredgecolor='white', markerfacecolor='black',
           label='Initialisation date'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           frameon=True, fancybox=True, shadow=False, fontsize=10.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.95,
           edgecolor='#CCCCCC')

# ── Colorbar for forecast lead ────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=FCST_CMAP_OBJ, norm=plt.Normalize(1, 42))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Forecast lead (days)', fontsize=10.5, labelpad=8)
cbar.set_ticks([1, 7, 14, 21, 28, 35, 42])
cbar.ax.tick_params(labelsize=9, length=3)
cbar.outline.set_linewidth(0.6)

# ── Save ──────────────────────────────────────────────────────────────────
fig.savefig(OUT_FILE, dpi=300, bbox_inches='tight', pad_inches=0.2,
            facecolor='white')
plt.close(fig)
print(f"\n✓ Saved: {OUT_FILE}")
print(f"  Size: {OUT_FILE.stat().st_size / 1024:.0f} KB")
