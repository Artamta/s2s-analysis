"""
05_plot_acc_skill_maps.py

ACC skill maps for T2M and precipitation (W1–W4) + area-averaged ACC vs lead week.

Layout of main figure: 2 rows × 4 cols
  Row 1 : T2M     ACC  (W1, W2, W3, W4)
  Row 2 : Precip  ACC  (W1, W2, W3, W4)

Colormap: RdYlGn, range -0.3 to 1.0
Stippling: grey dots where |ACC| < 0.3  (below ~95% significance, n=90)

Second figure: India-mean ACC vs lead week for T2M, Precip, Z500.

Input:  skill_metrics.nc   (output of Script 3)
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

INPUT_FILE     = "skill_metrics.nc"
OUTPUT_DIR     = "figures"
OUT_MAPS       = os.path.join(OUTPUT_DIR, "acc_skill_maps.png")
OUT_LEAD_CURVE = os.path.join(OUTPUT_DIR, "acc_vs_lead.png")

# Significance threshold: |ACC| < this → stipple
ACC_SIG_THRESH = 0.3

# India core box for area-mean ACC
INDIA_LAT = (8, 35)
INDIA_LON = (68, 98)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
ds = xr.open_dataset(INPUT_FILE)
print(ds)

lat   = ds["latitude"].values
lon   = ds["longitude"].values
weeks = ds["week"].values
week_labels = ["W1  days 1–7", "W2  days 8–14", "W3  days 15–21", "W4  days 22–28"]

# ── Map decoration helper ──────────────────────────────────────────────────────
def decorate_map(ax, title):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   linewidth=0.6, edgecolor="black", zorder=3)
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m",
                                     facecolor="none"),
        linewidth=0.4, edgecolor="black", zorder=3,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", linestyle=":")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def add_stippling(ax, acc_2d, lat, lon, thresh=ACC_SIG_THRESH):
    """Grey dots where |ACC| < thresh (not statistically significant)."""
    insig = np.abs(acc_2d) < thresh
    lon2d, lat2d = np.meshgrid(lon, lat)
    ax.scatter(
        lon2d[insig], lat2d[insig],
        s=0.6, color="gray", alpha=0.5,
        transform=ccrs.PlateCarree(), zorder=4,
    )

# ── Colormap ───────────────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)

# ── Figure 1: ACC maps ─────────────────────────────────────────────────────────
print("\nPlotting ACC skill maps …")

fig, axes = plt.subplots(
    2, 4,
    figsize=(16, 8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.05, "wspace": 0.05},
)

for col, (wk, wk_label) in enumerate(zip(weeks, week_labels)):
    acc_t2m    = ds["acc_t2m"].sel(week=wk).values     # (lat, lon)
    acc_precip = ds["acc_precip"].sel(week=wk).values

    # India-mean for panel subtitle
    lat_m = (lat >= INDIA_LAT[0]) & (lat <= INDIA_LAT[1])
    lon_m = (lon >= INDIA_LON[0]) & (lon <= INDIA_LON[1])
    india_t2m    = float(np.nanmean(acc_t2m[np.ix_(lat_m, lon_m)]))
    india_precip = float(np.nanmean(acc_precip[np.ix_(lat_m, lon_m)]))

    # Row 0: T2M ACC
    im = axes[0, col].pcolormesh(lon, lat, acc_t2m,
                                  cmap=cmap, norm=norm,
                                  transform=ccrs.PlateCarree())
    add_stippling(axes[0, col], acc_t2m, lat, lon)
    decorate_map(axes[0, col], f"T2M  {wk_label}\nIndia CC={india_t2m:.2f}")

    # Row 1: Precip ACC
    axes[1, col].pcolormesh(lon, lat, acc_precip,
                             cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree())
    add_stippling(axes[1, col], acc_precip, lat, lon)
    decorate_map(axes[1, col], f"Precip  {wk_label}\nIndia CC={india_precip:.2f}")

# Row labels
for row_idx, row_label in enumerate(["T2M ACC", "Precip ACC"]):
    axes[row_idx, 0].text(
        -0.14, 0.5, row_label, transform=axes[row_idx, 0].transAxes,
        rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
    )

# Shared colorbar
cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
cb.set_label("Anomaly Correlation Coefficient (ACC)", fontsize=9)
cb.set_ticks([-0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
cb.ax.tick_params(labelsize=8)

fig.suptitle(
    "Spire JFM 2026 S2S — ACC Skill Maps  (stippling: |ACC| < 0.3)  |  India domain",
    fontsize=12, fontweight="bold", y=0.98,
)

fig.savefig(OUT_MAPS, dpi=200, bbox_inches="tight")
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

week_centers = [4, 11, 18, 25]   # mid-lead-day of each window

fig2, ax2 = plt.subplots(figsize=(7, 4))

ax2.plot(week_centers, acc_t2m_india,    "o-", color="firebrick",   lw=2, label="T2M")
ax2.plot(week_centers, acc_precip_india, "s-", color="royalblue",   lw=2, label="Precip")
ax2.plot(week_centers, acc_z500_india,   "^-", color="forestgreen", lw=2, label="Z500")

ax2.axhline(0.3, color="gray", linestyle="--", lw=1, label="Sig. threshold (0.3)")
ax2.axhline(0.0, color="black", lw=0.5)

ax2.set_xticks(week_centers)
ax2.set_xticklabels(["W1\n(d1–7)", "W2\n(d8–14)", "W3\n(d15–21)", "W4\n(d22–28)"])
ax2.set_ylabel("India-mean ACC", fontsize=10)
ax2.set_title("Spire JFM 2026 S2S — India-mean ACC vs Lead Week", fontsize=11, fontweight="bold")
ax2.set_ylim(-0.3, 1.05)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.4)

fig2.tight_layout()
fig2.savefig(OUT_LEAD_CURVE, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {OUT_LEAD_CURVE}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n── India-mean ACC summary ──")
print(f"{'':8s}  {'W1':>6s}  {'W2':>6s}  {'W3':>6s}  {'W4':>6s}")
print(f"{'T2M':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_t2m_india))
print(f"{'Precip':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_precip_india))
print(f"{'Z500':8s}  " + "  ".join(f"{v:6.3f}" for v in acc_z500_india))
