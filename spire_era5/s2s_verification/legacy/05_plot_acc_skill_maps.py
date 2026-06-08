"""
05_plot_acc_skill_maps.py

Publication-quality ACC skill maps + India-mean ACC vs lead week.

Figure 1: 2 rows × n_weeks cols
  Row 1 : T2M  ACC  (W1…W6)
  Row 2 : Precip ACC (W1…W6)
  Stippling: grey dots where |ACC| < 0.3 (below ~95% significance, n=90)

Figure 2: India-mean ACC vs lead week line plot for T2M, Precip, Z500.

Input:  skill_metrics.nc
Output: figures/acc_skill_maps.png
        figures/acc_vs_lead.png
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
import string

INPUT_FILE     = "skill_metrics.nc"
OUTPUT_DIR     = "figures"
OUT_MAPS       = os.path.join(OUTPUT_DIR, "acc_skill_maps.png")
OUT_LEAD_CURVE = os.path.join(OUTPUT_DIR, "acc_vs_lead.png")

ACC_SIG_THRESH = 0.3   # |ACC| below this → stipple (not significant)
INDIA_LAT = (8, 35)
INDIA_LON = (68, 98)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
ds = xr.open_dataset(INPUT_FILE)

lat   = ds["latitude"].values
lon   = ds["longitude"].values
weeks = ds["week"].values
n_weeks = len(weeks)

def week_label(wk):
    d0 = (int(wk) - 1) * 7 + 1
    d1 = int(wk) * 7
    return f"W{int(wk)}: d{d0}–{d1}"

week_labels = [week_label(wk) for wk in weeks]

# ── Colormap: deep brown/orange for negative, white at zero, teal/green positive
# Much cleaner than RdYlGn which has an ugly yellow band in the middle
from matplotlib.colors import LinearSegmentedColormap

acc_colors = [
    (0.60, 0.20, 0.00),   # deep burnt orange  → strong negative
    (0.90, 0.55, 0.30),   # light orange
    (1.00, 1.00, 1.00),   # white              → zero / insignificant
    (0.30, 0.75, 0.60),   # teal
    (0.00, 0.45, 0.30),   # deep green         → strong positive
]
cmap_acc = LinearSegmentedColormap.from_list("acc_cmap", acc_colors, N=256)
norm_acc  = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)

# ── Figure 1: ACC maps ─────────────────────────────────────────────────────────
print("Plotting ACC skill maps …")

col_width  = 3.0
fig_width  = col_width * n_weeks + 0.9
fig_height = 6.8

fig = plt.figure(figsize=(fig_width, fig_height))

left   = 0.10
right  = 0.90
bottom = 0.09
top    = 0.91
hspace = 0.04
wspace = 0.03

row_height    = (top - bottom - hspace) / 2
col_width_frac = (right - left - (n_weeks - 1) * wspace) / n_weeks

proj = ccrs.PlateCarree()

row_varnames = ["acc_t2m", "acc_precip"]
row_titles   = ["T2M ACC", "Precip ACC"]

axes = [[None] * n_weeks for _ in range(2)]
panel_idx = 0
im_ref = None

for row in range(2):
    varname = row_varnames[row]

    for col in range(n_weeks):
        wk = weeks[col]
        x0 = left + col * (col_width_frac + wspace)
        y0 = bottom + (1 - row) * (row_height + hspace)
        ax = fig.add_axes([x0, y0, col_width_frac, row_height], projection=proj)
        axes[row][col] = ax

        acc = ds[varname].sel(week=wk).values   # (lat, lon)
        im  = ax.pcolormesh(lon, lat, acc,
                            cmap=cmap_acc, norm=norm_acc,
                            transform=proj, rasterized=True)
        im_ref = im

        # Stippling: sparse black dots where |ACC| < threshold (not significant)
        insig = np.abs(acc) < ACC_SIG_THRESH
        lon2d, lat2d = np.meshgrid(lon, lat)
        step = 4   # every 4th grid point keeps it readable
        mask = insig[::step, ::step]
        ax.scatter(lon2d[::step, ::step][mask], lat2d[::step, ::step][mask],
                   s=1.5, color="k", alpha=0.35,
                   transform=proj, zorder=4)

        # Coastlines only — BORDERS skipped to avoid disputed boundaries
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       linewidth=0.7, edgecolor="k", zorder=5)
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=proj)

        # Gridlines
        gl = ax.gridlines(crs=proj, linewidth=0.25, color="gray",
                          linestyle=":", zorder=3)
        gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
        gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = (col == 0)
        gl.bottom_labels = (row == 1)
        gl.xlabel_style  = {"size": 6.5, "color": "0.3"}
        gl.ylabel_style  = {"size": 6.5, "color": "0.3"}

        # India-mean ACC in panel subtitle
        lat_m = (lat >= INDIA_LAT[0]) & (lat <= INDIA_LAT[1])
        lon_m = (lon >= INDIA_LON[0]) & (lon <= INDIA_LON[1])
        india_acc = float(np.nanmean(acc[np.ix_(lat_m, lon_m)]))

        # Column header on top row
        if row == 0:
            ax.set_title(week_labels[col], fontsize=8.5, fontweight="bold", pad=3)

        # India ACC value at bottom of panel
        ax.text(0.98, 0.03, f"India r={india_acc:.2f}",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=6)

        # Panel letter
        letter = string.ascii_lowercase[panel_idx]
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=6)
        panel_idx += 1

    # Row label
    axes[row][0].text(-0.18, 0.5, row_titles[row],
                      transform=axes[row][0].transAxes,
                      rotation=90, va="center", ha="center",
                      fontsize=9.5, fontweight="bold")

# ── Shared colorbar ────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, bottom, 0.016, top - bottom])
cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical", extend="both")
cb.set_label("Anomaly Correlation Coefficient (ACC)", fontsize=8.5, labelpad=6)
cb.set_ticks([-0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
cb.ax.tick_params(labelsize=7.5)

fig.suptitle(
    "Spire JFM 2026 S2S  |  ACC Skill Maps  |  stippling: |ACC| < 0.3",
    fontsize=11, fontweight="bold", y=0.965,
)

fig.savefig(OUT_MAPS, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_MAPS}")

# ── Figure 2: India-mean ACC vs lead week ─────────────────────────────────────
print("Plotting ACC vs lead week …")

lat_m = (lat >= INDIA_LAT[0]) & (lat <= INDIA_LAT[1])
lon_m = (lon >= INDIA_LON[0]) & (lon <= INDIA_LON[1])

acc_t2m_india    = []
acc_precip_india = []
acc_z500_india   = []

for wk in weeks:
    def india_mean(var):
        arr = ds[var].sel(week=wk).values
        return float(np.nanmean(arr[np.ix_(lat_m, lon_m)]))
    acc_t2m_india.append(india_mean("acc_t2m"))
    acc_precip_india.append(india_mean("acc_precip"))
    acc_z500_india.append(india_mean("acc_z500"))

week_centers = [(int(wk) - 1) * 7 + 4 for wk in weeks]
xtick_labels = [f"W{int(wk)}\n(d{(int(wk)-1)*7+1}–{int(wk)*7})" for wk in weeks]

fig2, ax2 = plt.subplots(figsize=(max(6, 1.2 * n_weeks + 2), 4.2))

ax2.plot(week_centers, acc_t2m_india,    "o-", color="#C0392B",
         lw=2.2, ms=7, markeredgewidth=0.8, markeredgecolor="white", label="T2M", zorder=4)
ax2.plot(week_centers, acc_precip_india, "s-", color="#2980B9",
         lw=2.2, ms=7, markeredgewidth=0.8, markeredgecolor="white", label="Precip", zorder=4)
ax2.plot(week_centers, acc_z500_india,   "^-", color="#27AE60",
         lw=2.2, ms=7, markeredgewidth=0.8, markeredgecolor="white", label="Z500", zorder=4)

ax2.axhline(ACC_SIG_THRESH, color="0.45", linestyle="--", lw=1.2,
            label=f"Significance threshold (r = {ACC_SIG_THRESH})", zorder=3)
ax2.axhline(0.0, color="k", lw=0.8, zorder=3)

ax2.fill_between(week_centers, -1, ACC_SIG_THRESH,
                 color="0.92", zorder=1)

ax2.set_xticks(week_centers)
ax2.set_xticklabels(xtick_labels, fontsize=9)
ax2.set_ylabel("India-mean ACC (r)", fontsize=10.5)
ax2.set_xlabel("Forecast lead", fontsize=10.5)
ax2.set_ylim(-0.35, 1.0)
ax2.set_xlim(week_centers[0] - 3, week_centers[-1] + 3)

ax2.legend(fontsize=9, loc="upper right", framealpha=0.9,
           edgecolor="0.7", frameon=True)
ax2.grid(axis="y", alpha=0.35, lw=0.7, color="0.6")
ax2.spines[["top", "right"]].set_visible(False)

ax2.set_title("Spire JFM 2026 S2S  |  India-mean ACC vs Forecast Lead",
              fontsize=11, fontweight="bold", pad=8)

fig2.tight_layout()
fig2.savefig(OUT_LEAD_CURVE, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved → {OUT_LEAD_CURVE}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── India-mean ACC summary ──")
header = f"{'':8s}  " + "  ".join(f"{'W'+str(int(wk)):>6s}" for wk in weeks)
print(header)
print(f"{'T2M':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_t2m_india))
print(f"{'Precip':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_precip_india))
print(f"{'Z500':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_z500_india))
