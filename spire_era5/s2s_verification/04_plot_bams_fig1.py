"""
04_plot_bams_fig1.py

BAMS Fig 1 equivalent — T2M anomaly spatial maps for a single init date.

Layout: 3 rows × 4 cols
  Row 1 : Spire T2M anomaly forecast  (W1, W2, W3, W4)
  Row 2 : ERA5 T2M anomaly observed   (W1, W2, W3, W4)
  Row 3 : Bias = Spire − ERA5         (W1, W2, W3, W4)

By default picks the init date with the largest India-domain T2M anomaly signal
(a proxy for MJO-active cases). Override with INIT_DATE below.

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

INPUT_FILE  = "weekly_anomalies.nc"
OUTPUT_DIR  = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bams_fig1_t2m_anomaly.png")

# Set a specific init date string "YYYY-MM-DD" to override auto-selection,
# or leave as None to auto-pick the most anomalous JFM 2026 case.
INIT_DATE = None

# Colormap limits for T2M anomaly (°C) — symmetric, RdBu_r
CLIM = 4.0   # ±4°C

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE} …")
ds = xr.open_dataset(INPUT_FILE)

lat  = ds["latitude"].values
lon  = ds["longitude"].values
weeks = [1, 2, 3, 4]

# ── Select init date ───────────────────────────────────────────────────────────
if INIT_DATE is not None:
    init = np.datetime64(INIT_DATE)
    i_sel = int(np.argmin(np.abs(ds["init_time"].values - init)))
    print(f"Using specified init date: {str(ds['init_time'].values[i_sel])[:10]}")
else:
    # Auto-select: init date with largest absolute T2M anomaly over India core
    # (8–35°N, 68–98°E), averaged across W1–W4
    lat_mask = (lat >= 8)  & (lat <= 35)
    lon_mask = (lon >= 68) & (lon <= 98)

    signal = (np.abs(ds["spire_t2m_anom"].values)          # (90, 4, lat, lon)
                [:, :, :, :][:, :, np.ix_(lat_mask, lon_mask)[0]]
                [..., lon_mask])
    # shape trickery — just use sel
    sp_india = (ds["spire_t2m_anom"]
                  .sel(latitude=slice(8, 35), longitude=slice(68, 98))
                  .values)                                  # (90, 4, lat, lon)
    mean_signal = np.abs(sp_india).mean(axis=(1, 2, 3))    # (90,)
    i_sel = int(np.argmax(mean_signal))
    selected_date = str(ds["init_time"].values[i_sel])[:10]
    print(f"Auto-selected init date: {selected_date}  (strongest India T2M signal)")

init_label = str(ds["init_time"].values[i_sel])[:10]

# Extract (week, lat, lon) slices for selected init date
spire_t2m = ds["spire_t2m_anom"].isel(init_time=i_sel).values  # (4, lat, lon)
era5_t2m  = ds["era5_t2m_anom"].isel(init_time=i_sel).values
bias_t2m  = spire_t2m - era5_t2m

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

# ── Build figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 4,
    figsize=(16, 12),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.08, "wspace": 0.05},
)

cmap_anom = "RdBu_r"
norm_anom = mcolors.TwoSlopeNorm(vmin=-CLIM, vcenter=0, vmax=CLIM)

cmap_bias = "RdBu_r"
norm_bias = mcolors.TwoSlopeNorm(vmin=-CLIM / 2, vcenter=0, vmax=CLIM / 2)

for col, wk_label in enumerate(week_labels):
    # Row 0: Spire forecast anomaly
    im = axes[0, col].pcolormesh(lon, lat, spire_t2m[col],
                                  cmap=cmap_anom, norm=norm_anom,
                                  transform=ccrs.PlateCarree())
    decorate_map(axes[0, col], f"Spire | {wk_label}")

    # Row 1: ERA5 observed anomaly
    axes[1, col].pcolormesh(lon, lat, era5_t2m[col],
                             cmap=cmap_anom, norm=norm_anom,
                             transform=ccrs.PlateCarree())
    decorate_map(axes[1, col], f"ERA5 obs | {wk_label}")

    # Row 2: Bias
    im_bias = axes[2, col].pcolormesh(lon, lat, bias_t2m[col],
                                       cmap=cmap_bias, norm=norm_bias,
                                       transform=ccrs.PlateCarree())
    decorate_map(axes[2, col], f"Bias (Spire−ERA5) | {wk_label}")

# Row labels on the left
for row_idx, row_label in enumerate(["Spire forecast", "ERA5 observed", "Bias"]):
    axes[row_idx, 0].text(
        -0.12, 0.5, row_label, transform=axes[row_idx, 0].transAxes,
        rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
    )

# Colorbars
cbar_ax1 = fig.add_axes([0.15, 0.04, 0.50, 0.018])
cb1 = fig.colorbar(im, cax=cbar_ax1, orientation="horizontal", extend="both")
cb1.set_label("T2M anomaly (°C)", fontsize=9)
cb1.ax.tick_params(labelsize=8)

cbar_ax2 = fig.add_axes([0.70, 0.04, 0.22, 0.018])
cb2 = fig.colorbar(im_bias, cax=cbar_ax2, orientation="horizontal", extend="both")
cb2.set_label("Bias (°C)", fontsize=9)
cb2.ax.tick_params(labelsize=8)

fig.suptitle(
    f"Spire JFM 2026 S2S — T2M Anomaly  |  Init: {init_label}  |  India domain",
    fontsize=12, fontweight="bold", y=0.98,
)

fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUTPUT_FILE}")
