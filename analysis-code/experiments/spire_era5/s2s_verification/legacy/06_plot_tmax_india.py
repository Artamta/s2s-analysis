"""
06_plot_tmax_india.py

India-domain T2m anomaly maps: Spire forecast vs ERA5 observed.

Reads directly from weekly_anomalies.nc (already computed, no re-fetch).
Spire anomaly: air_temperature_max from anomalies group (vs ERA5 1991-2020 climo).
ERA5 anomaly: weekly mean T2m − WB2 1990-2019 DOY climo (from weekly_anomalies.nc).

Layout: 2 rows × 6 cols
  Row 1: Spire Tmax anomaly forecast (W1–W6)
  Row 2: ERA5 T2m anomaly observed   (W1–W6)

Averaged over all 90 JFM 2026 init dates.

Output: figures/tmax_india_anomaly.png
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
from arraylake import Client

OUTPUT_DIR  = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tmax_india_anomaly.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 0.0,  50.0
LON_MIN, LON_MAX = 55.0, 105.0
WEEKS = {1:(1,7), 2:(8,14), 3:(15,21), 4:(22,28), 5:(29,35), 6:(36,42)}
CLIM  = 8.0   # ±K for colorbar — wider range to show more detail

# ── 1. Load ERA5 T2m anomaly from weekly_anomalies.nc (already computed) ──────
print("Loading ERA5 T2m anomaly from weekly_anomalies.nc …")
ds_nc = xr.open_dataset("weekly_anomalies.nc")
lat = ds_nc["latitude"].values
lon = ds_nc["longitude"].values

# Get init date range for title
init_times = pd.DatetimeIndex(ds_nc["init_time"].values)
init_range = f"{init_times[0].strftime('%Y-%m-%d')} to {init_times[-1].strftime('%Y-%m-%d')}"

# ERA5: average over all 90 inits per week
era5_tmax = {}
for wk in WEEKS:
    arr = ds_nc["era5_t2m_anom"].sel(week=wk).mean("init_time").values  # (lat, lon)
    era5_tmax[wk] = arr
    lm = (lat>=8)&(lat<=35); om = (lon>=68)&(lon<=98)
    print(f"  W{wk} ERA5: India mean = {np.nanmean(arr[np.ix_(lm,om)]):.2f} K")

# ── 2. Load Spire Tmax anomaly from arraylake (correct climo) ─────────────────
print("\nOpening Spire anomalies group …")
client  = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")
ds_anom = xr.open_zarr(session.store, group="anomalies")

ds_anom = (ds_anom
           .isel(latitude=slice(None, None, -1))
           .sel(latitude=slice(LAT_MIN, LAT_MAX),
                longitude=slice(LON_MIN, LON_MAX)))

spire_lat = ds_anom["latitude"].values
spire_lon = ds_anom["longitude"].values
n_inits   = ds_anom.sizes["reference_time"]

spire_tmax = {}
print("Computing Spire Tmax weekly means (averaged over 90 inits) …")
for wk, (d0, d1) in WEEKS.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    arr = (ds_anom["air_temperature_max"]
           .sel(step=steps).mean("step")
           .mean("reference_time")
           .compute().values)          # (lat, lon)  K anomaly vs ERA5 1991-2020
    spire_tmax[wk] = arr
    lm = (spire_lat>=8)&(spire_lat<=35); om = (spire_lon>=68)&(spire_lon<=98)
    print(f"  W{wk} Spire: India mean = {np.nanmean(arr[np.ix_(lm,om)]):.2f} K")

# ── 3. Plot ────────────────────────────────────────────────────────────────────
print("\nPlotting …")
n_weeks  = len(WEEKS)
weeks    = list(WEEKS.keys())
col_w    = 3.0
fig_w    = col_w * n_weeks + 0.9
fig_h    = 5.5

fig = plt.figure(figsize=(fig_w, fig_h))

left     = 0.12
right    = 0.90
bottom   = 0.10
top      = 0.91
hspace   = 0.04
wspace   = 0.03

row_h    = (top - bottom - hspace) / 2
col_frac = (right - left - (n_weeks - 1) * wspace) / n_weeks

proj  = ccrs.PlateCarree()
norm  = mcolors.TwoSlopeNorm(vmin=-CLIM, vcenter=0, vmax=CLIM)
cmap  = "RdBu_r"

row_data    = [spire_tmax, era5_tmax]
row_lats    = [spire_lat,  lat]
row_lons    = [spire_lon,  lon]
row_labels  = ["Spire Tmax forecast\n(vs ERA5 1991–2020 climo)",
               "ERA5 T2m observed\n(vs WB2 1990–2019 climo)"]
week_labels = [f"W{wk}\n(d{d0}–{d1})" for wk, (d0, d1) in WEEKS.items()]

im_ref    = None
panel_idx = 0

for row in range(2):
    data = row_data[row]
    rlat = row_lats[row]
    rlon = row_lons[row]

    for col, wk in enumerate(weeks):
        x0 = left + col * (col_frac + wspace)
        y0 = bottom + (1 - row) * (row_h + hspace)
        ax = fig.add_axes([x0, y0, col_frac, row_h], projection=proj)

        im = ax.pcolormesh(rlon, rlat, data[wk],
                           cmap=cmap, norm=norm,
                           transform=proj, rasterized=True)
        im_ref = im

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       linewidth=0.7, edgecolor="k", zorder=4)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

        gl = ax.gridlines(crs=proj, linewidth=0.25, color="gray",
                          linestyle=":", zorder=3)
        gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
        gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
        gl.top_labels = gl.right_labels = False
        gl.left_labels   = (col == 0)
        gl.bottom_labels = (row == 1)
        gl.xlabel_style  = {"size": 6.5, "color": "0.3"}
        gl.ylabel_style  = {"size": 6.5, "color": "0.3"}

        if row == 0:
            ax.set_title(week_labels[col], fontsize=8.5, fontweight="bold", pad=3)

        # India-mean value in panel corner
        lm = (rlat >= 8) & (rlat <= 35)
        om = (rlon >= 68) & (rlon <= 98)
        india_mean = float(np.nanmean(data[wk][np.ix_(lm, om)]))
        ax.text(0.98, 0.03, f"India={india_mean:+.1f}K",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=6)

        letter = string.ascii_lowercase[panel_idx]
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5)
        panel_idx += 1

    # Row label left of first column
    leftmost = fig.axes[row * n_weeks]
    leftmost.text(-0.22, 0.5, row_labels[row],
                  transform=leftmost.transAxes,
                  rotation=90, va="center", ha="center",
                  fontsize=8.5, fontweight="bold")

# Shared colorbar
cbar_ax = fig.add_axes([0.92, bottom, 0.016, top - bottom])
cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical", extend="both")
cb.set_label("Temperature anomaly (K)", fontsize=8.5, labelpad=6)
cb.set_ticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
cb.ax.tick_params(labelsize=7.5)

fig.suptitle(
    f"Spire JFM 2026 S2S  |  2m Temperature Anomaly (90-init mean, {init_range})",
    fontsize=10, fontweight="bold", y=0.97)

fig.savefig(OUTPUT_FILE, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUTPUT_FILE}")
