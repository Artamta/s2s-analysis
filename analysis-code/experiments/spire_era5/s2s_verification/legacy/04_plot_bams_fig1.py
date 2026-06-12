"""
04_plot_bams_fig1.py

Publication-quality T2M anomaly spatial maps for a single init date.

Layout: 3 rows × 6 cols  (or however many weeks are in the file)
  Row 1 : Spire T2M anomaly forecast
  Row 2 : ERA5 T2M anomaly observed
  Row 3 : Bias = Spire − ERA5

Auto-selects the init date with strongest India-domain T2M anomaly signal.
Override with INIT_DATE = "YYYY-MM-DD".

Input:  weekly_anomalies.nc
Output: figures/bams_fig1_t2m_anomaly.png
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import string

INPUT_FILE  = "weekly_anomalies.nc"
OUTPUT_DIR  = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bams_fig1_t2m_anomaly.png")

INIT_DATE = None   # set "YYYY-MM-DD" to override auto-selection
CLIM_ANOM = 4.0    # ±°C for anomaly panels
CLIM_BIAS = 2.0    # ±°C for bias panels

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
ds = xr.open_dataset(INPUT_FILE)
lat   = ds["latitude"].values
lon   = ds["longitude"].values
weeks = ds["week"].values
n_weeks = len(weeks)

# ── Select init date ───────────────────────────────────────────────────────────
if INIT_DATE is not None:
    init = np.datetime64(INIT_DATE)
    i_sel = int(np.argmin(np.abs(ds["init_time"].values - init)))
else:
    sp_india = (ds["spire_t2m_anom"]
                  .sel(latitude=slice(8, 35), longitude=slice(68, 98))
                  .values)
    mean_signal = np.abs(sp_india).mean(axis=(1, 2, 3))
    i_sel = int(np.argmax(mean_signal))

init_label = str(ds["init_time"].values[i_sel])[:10]
print(f"Init date: {init_label}")

spire_t2m = ds["spire_t2m_anom"].isel(init_time=i_sel).values   # (n_weeks, lat, lon)
era5_t2m  = ds["era5_t2m_anom"].isel(init_time=i_sel).values
bias_t2m  = spire_t2m - era5_t2m

week_labels = [f"W{int(wk)}: days {(int(wk)-1)*7+1}–{int(wk)*7}" for wk in weeks]

# ── Figure setup ───────────────────────────────────────────────────────────────
col_width = 3.0
fig_width = col_width * n_weeks + 0.8   # +0.8 for row labels
fig_height = 9.5

fig = plt.figure(figsize=(fig_width, fig_height))

# Manual grid to give precise control over spacing
left   = 0.10
right  = 0.90
bottom = 0.10
top    = 0.93
hspace = 0.04
wspace = 0.03

row_height = (top - bottom - 2 * hspace) / 3
col_width_frac = (right - left - (n_weeks - 1) * wspace) / n_weeks

proj = ccrs.PlateCarree()

row_labels = ["(a) Spire forecast", "(b) ERA5 observed", "(c) Bias (Spire − ERA5)"]
data_rows  = [spire_t2m, era5_t2m, bias_t2m]
clims      = [CLIM_ANOM, CLIM_ANOM, CLIM_BIAS]
cmaps      = ["RdBu_r", "RdBu_r", "RdBu_r"]

axes = [[None] * n_weeks for _ in range(3)]

for row in range(3):
    for col in range(n_weeks):
        x0 = left + col * (col_width_frac + wspace)
        y0 = bottom + (2 - row) * (row_height + hspace)
        ax = fig.add_axes([x0, y0, col_width_frac, row_height],
                          projection=proj)
        axes[row][col] = ax

# ── Colormaps ──────────────────────────────────────────────────────────────────
def get_norm(clim):
    return mcolors.TwoSlopeNorm(vmin=-clim, vcenter=0, vmax=clim)

norm_anom = get_norm(CLIM_ANOM)
norm_bias = get_norm(CLIM_BIAS)

# ── Plot ───────────────────────────────────────────────────────────────────────
panel_idx = 0
im_anom = None
im_bias = None

for row in range(3):
    data   = data_rows[row]
    norm   = norm_bias if row == 2 else norm_anom
    cmap   = cmaps[row]

    for col in range(n_weeks):
        ax = axes[row][col]
        field = data[col]

        im = ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm,
                           transform=proj, rasterized=True)
        if row == 2:
            im_bias = im
        else:
            im_anom = im

        # Coastlines only — BORDERS skipped to avoid disputed boundaries
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       linewidth=0.7, edgecolor="k", zorder=4)

        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)

        # Gridlines — only left/bottom edges get labels
        gl = ax.gridlines(crs=proj, linewidth=0.25, color="gray",
                          linestyle=":", zorder=3)
        gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
        gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = (col == 0)
        gl.bottom_labels = (row == 2)
        gl.xlabel_style  = {"size": 6.5, "color": "0.3"}
        gl.ylabel_style  = {"size": 6.5, "color": "0.3"}

        # Column header (week label) on top row only
        if row == 0:
            ax.set_title(week_labels[col], fontsize=8.5, fontweight="bold", pad=3)

        # Panel letter in top-left corner
        letter = string.ascii_lowercase[panel_idx]
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5)
        panel_idx += 1

    # Row label on left of first column
    ax0 = axes[row][0]
    ax0.text(-0.18, 0.5, row_labels[row],
             transform=ax0.transAxes,
             rotation=90, va="center", ha="center",
             fontsize=9, fontweight="bold")

# ── Colorbars ─────────────────────────────────────────────────────────────────
# Anomaly colorbar (spans rows 0–1)
cbar_ax1 = fig.add_axes([0.92, bottom + row_height + hspace, 0.016,
                          2 * row_height + hspace])
cb1 = fig.colorbar(im_anom, cax=cbar_ax1, orientation="vertical", extend="both")
cb1.set_label("T2M anomaly (°C)", fontsize=8.5, labelpad=6)
cb1.ax.tick_params(labelsize=7.5)
cb1.set_ticks([-4, -2, 0, 2, 4])

# Bias colorbar (row 2 only)
cbar_ax2 = fig.add_axes([0.92, bottom, 0.016, row_height])
cb2 = fig.colorbar(im_bias, cax=cbar_ax2, orientation="vertical", extend="both")
cb2.set_label("Bias (°C)", fontsize=8.5, labelpad=6)
cb2.ax.tick_params(labelsize=7.5)
cb2.set_ticks([-2, -1, 0, 1, 2])

# ── Title ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Spire JFM 2026 S2S  |  T2M Weekly Anomaly  |  Init: {init_label}",
    fontsize=11, fontweight="bold", y=0.975,
)

fig.savefig(OUTPUT_FILE, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUTPUT_FILE}")
