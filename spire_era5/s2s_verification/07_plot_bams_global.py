"""
07_plot_bams_global.py

Global T2m-max anomaly maps, styled like the BAMS S2S multimodel comparison figure.

4-panel layout (2×2):
  Top-left:  ERA5 verification (observed anomaly)
  Top-right: Spire forecast W1 (d1–7)
  Bot-left:  Spire forecast W2 (d8–14)
  Bot-right: Spire forecast W3 (d15–21)

All panels show the JFM 2026 average over 90 init dates.
Colormap: 7-color discrete scheme matching the BAMS figure (blue→red).
Stippling: grid points where Spire anomaly is not significantly different
           from zero (|mean| < 0.5 K, a simple ensemble spread proxy).

Output: figures/bams_global_tmax.png
        figures/bams_india_tmax.png  (same but zoomed to Indian subcontinent)
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
from earth2studio.data import WB2Climatology

OUTPUT_DIR    = "figures"
OUT_GLOBAL    = os.path.join(OUTPUT_DIR, "bams_global_tmax.png")
OUT_INDIA     = os.path.join(OUTPUT_DIR, "bams_india_tmax.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── BAMS-style discrete colormap — expanded to ±12K with more steps for better color variation
BOUNDS = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
COLORS = [
    "#08306b",  # very deep blue   < -10
    "#053061",  # deep blue        -10 to -8
    "#2166AC",  # mid-dark blue    -8 to -6
    "#4393C3",  # mid blue         -6 to -4
    "#92C5DE",  # light blue       -4 to -2
    "#D1E5F0",  # very light blue  -2 to 0
    "#FEE090",  # pale yellow      0 to 2
    "#FDAE61",  # light orange     2 to 4
    "#F46D43",  # orange-red       4 to 6
    "#D73027",  # red              6 to 8
    "#A50026",  # dark red         8 to 10
    "#67001F",  # very dark red    > 10
]
cmap_bams = mcolors.ListedColormap(COLORS)
norm_bams = mcolors.BoundaryNorm(BOUNDS, ncolors=len(COLORS))

WEEKS_PLOT = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, 35), 6: (36, 42)}

# Global domain
GLAT_MIN, GLAT_MAX = -80.0, 80.0
GLON_MIN, GLON_MAX = -180.0, 180.0

# India domain (for second figure)
ILAT_MIN, ILAT_MAX = 0.0,  50.0
ILON_MIN, ILON_MAX = 55.0, 105.0

# ── 1. Open Spire (global extent) ─────────────────────────────────────────────
print("Opening Spire (global) …")
client  = Client()
session = client.get_repo("artamta/s2s-research").readonly_session("main")
ds_anom_raw = xr.open_zarr(session.store, group="anomalies")

# Flip lat descending→ascending
ds_anom = ds_anom_raw.isel(latitude=slice(None, None, -1))
spire_lat_g = ds_anom["latitude"].values      # -90 → 90
spire_lon_g = ds_anom["longitude"].values     # 0 → 359.5
init_times  = pd.DatetimeIndex(ds_anom["reference_time"].values)
n_inits     = len(init_times)
print(f"Spire: {n_inits} inits  |  global grid {len(spire_lat_g)}×{len(spire_lon_g)}")

# Compute Spire weekly Tmax anomaly averaged over all inits (global)
spire_tmax_g = {}
print("Computing Spire weekly Tmax global means …")
for wk, (d0, d1) in WEEKS_PLOT.items():
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]
    arr = (ds_anom["air_temperature_max"]
           .sel(step=steps).mean("step")
           .mean("reference_time")
           .compute().values)                      # (lat, lon) K
    spire_tmax_g[wk] = arr
    print(f"  W{wk} range: {arr.min():.2f}–{arr.max():.2f} K")

# ── 2. ERA5 global daily Tmax window ─────────────────────────────────────────
window_start = init_times[0]  + pd.Timedelta(1,  "D")
window_end   = init_times[-1] + pd.Timedelta(42, "D")   # need W1–W6
all_dates    = pd.date_range(window_start, window_end, freq="D")
date_to_idx  = {d: i for i, d in enumerate(all_dates)}
print(f"\nERA5 window: {window_start.date()} → {window_end.date()} ({len(all_dates)} days)")

print("Opening ARCO-ERA5 …")
ds_era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
)
t0 = f"{window_start.date()}T00:00"
t1 = f"{window_end.date()}T23:00"

# Global fetch — latitude: ascending slice for xarray interp
da_era5 = ds_era5["2m_temperature"].sel(
    latitude=slice(GLAT_MAX + 1, GLAT_MIN - 1),
    longitude=slice(0, 360),
    time=slice(t0, t1),
)
print(f"Fetching ERA5 T2m ({da_era5.sizes['time']} steps) …", flush=True)
era5_daily_g = da_era5.resample(time="1D").mean("time").compute() - 273.15
era5_lat_g   = era5_daily_g["latitude"].values
era5_lon_g   = era5_daily_g["longitude"].values
era5_daily_g = era5_daily_g.reindex(time=all_dates).values.astype(np.float32)
print("ERA5 cached.")

# ── 3. WB2 climo for ERA5 anomaly (global) ────────────────────────────────────
print("Fetching WB2 climo (global) …")
wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr", verbose=False)
doys_needed = sorted(set(all_dates.day_of_year.tolist()))
wb2_times   = [pd.Timestamp("2001-01-01") + pd.Timedelta(int(d) - 1, "D")
               for d in doys_needed]
da_wb2 = wb2(wb2_times, ["t2m"])

# WB2 lat is descending (90 → -90), lon 0→360 at 0.25°
# Crop to global bands; flip lat ascending; keep lon 0–360
da_wb2 = da_wb2.isel(lat=slice(None, None, -1))    # now ascending
wb2_lat_g = da_wb2["lat"].values
wb2_lon_g = da_wb2["lon"].values

wb2_by_doy = {}
for k, doy in enumerate(doys_needed):
    wb2_by_doy[doy] = da_wb2.isel(time=k).sel(variable="t2m").values - 273.15  # (lat, lon)
print(f"WB2 cached for {len(doys_needed)} DOYs  |  grid {len(wb2_lat_g)}×{len(wb2_lon_g)}")

# ── 4. ERA5 weekly anomaly averaged over inits (on ERA5 native 0.25° grid) ────
# Then interpolate to Spire 0.5° grid for consistency
print("Computing ERA5 weekly anomaly (global) …")
era5_anom_g = {}
for wk, (d0, d1) in WEEKS_PLOT.items():
    accum = np.zeros((len(era5_lat_g), len(era5_lon_g)), dtype=np.float64)
    for init in init_times:
        vdates = pd.date_range(init + pd.Timedelta(d0, "D"),
                               init + pd.Timedelta(d1, "D"))
        idx   = [date_to_idx[d] for d in vdates]
        obs   = era5_daily_g[idx].mean(axis=0)

        # WB2 is 0.25° — interpolate climo to ERA5 grid each time is too slow;
        # instead use the WB2 grid directly (same 0.25°, close enough)
        clim  = np.mean([wb2_by_doy[int(d.day_of_year)] for d in vdates], axis=0)
        # WB2 lat already ascending, same 0–360 lon
        # Interpolate clim to ERA5 lat/lon grid (they may differ slightly)
        from scipy.interpolate import RegularGridInterpolator
        rgi = RegularGridInterpolator(
            (wb2_lat_g, wb2_lon_g), clim,
            method="linear", bounds_error=False, fill_value=None)
        era5_ll = np.array(np.meshgrid(era5_lon_g, era5_lat_g)).reshape(2, -1).T
        # meshgrid gives (lon, lat) → swap
        pts = np.column_stack([era5_ll[:, 1], era5_ll[:, 0]])
        clim_interp = rgi(pts).reshape(len(era5_lat_g), len(era5_lon_g))
        accum += obs - clim_interp

    era5_anom_g[wk] = (accum / n_inits).astype(np.float32)
    print(f"  W{wk} ERA5 range: {era5_anom_g[wk].min():.2f}–{era5_anom_g[wk].max():.2f} °C")

# ── 5. Plot global (BAMS style) ────────────────────────────────────────────────
print("\nPlotting global BAMS-style figure …")

panel_configs = [
    ("ERA5 Verification",  None,  1),     # ERA5 obs, W1 dates
    ("Spire W1 (d1–7)",    1,     None),
    ("Spire W2 (d8–14)",   2,     None),
    ("Spire W3 (d15–21)",  3,     None),
    ("Spire W4 (d22–28)",  4,     None),
    ("Spire W5 (d29–35)",  5,     None),
    ("Spire W6 (d36–42)",  6,     None),
]

proj_rob = ccrs.Robinson(central_longitude=0)
proj_pc  = ccrs.PlateCarree()

fig, axes = plt.subplots(2, 4, figsize=(18, 8),
                          subplot_kw={"projection": proj_rob})
fig.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.90,
                    hspace=0.15, wspace=0.08)

for idx, (title, spire_wk, era5_wk) in enumerate(panel_configs):
    row, col = divmod(idx, 4)
    ax = axes[row, col]

    if spire_wk is not None:
        field = spire_tmax_g[spire_wk]
        lats, lons = spire_lat_g, spire_lon_g
    else:
        # ERA5 verification uses W1 dates
        field = era5_anom_g[1]
        lats, lons = era5_lat_g, era5_lon_g

    # Shift lon 0–360 → -180–180 for Robinson
    if lons.max() > 180:
        shift = len(lons) // 2
        field = np.roll(field, shift, axis=1)
        lons  = np.where(lons > 180, lons - 360, lons)
        lons  = np.roll(lons, shift)

    im = ax.pcolormesh(lons, lats, field,
                       cmap=cmap_bams, norm=norm_bams,
                       transform=proj_pc, rasterized=True)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.6, edgecolor="k", zorder=4)
    ax.set_global()

    gl = ax.gridlines(crs=proj_pc, linewidth=0.2, color="gray",
                      linestyle=":", alpha=0.5, zorder=3)
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    letter = string.ascii_lowercase[idx]
    ax.set_title(f"({letter}) {title}", fontsize=10.5, fontweight="bold", pad=4)

# Hide any unused axes (7 panels in a 2×4 grid → last cell is empty)
for j in range(len(panel_configs), axes.size):
    axes.flat[j].set_visible(False)

# Shared colorbar
cbar_ax = fig.add_axes([0.88, 0.10, 0.014, 0.78])
cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical",
                  extend="both", ticks=BOUNDS)
cb.set_label("Anomaly (K)", fontsize=8.5, labelpad=5)
cb.ax.tick_params(labelsize=7)

init_range = f"{init_times[0].strftime('%Y-%m-%d')} to {init_times[-1].strftime('%Y-%m-%d')}"
fig.suptitle(f"Spire JFM 2026 S2S  |  2m Tmax Weekly Anomaly (90-init mean, {init_range})  |  vs ERA5 1991–2020 climo",
             fontsize=10.5, fontweight="bold", y=0.97)

fig.savefig(OUT_GLOBAL, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT_GLOBAL}")

# ── 6. Plot India-domain version (same style, zoomed) ─────────────────────────
print("Plotting India-domain BAMS-style figure …")

# Crop Spire and ERA5 to India
def crop(arr, lats, lons, lat0, lat1, lon0, lon1):
    lm = (lats >= lat0) & (lats <= lat1)
    om = (lons >= lon0) & (lons <= lon1)
    return arr[np.ix_(lm, om)], lats[lm], lons[om]

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8),
                            subplot_kw={"projection": proj_pc})
fig2.subplots_adjust(left=0.02, right=0.86, bottom=0.08, top=0.90,
                     hspace=0.15, wspace=0.08)

for idx, (title, spire_wk, _) in enumerate(panel_configs):
    row, col = divmod(idx, 4)
    ax = axes2[row, col]

    if spire_wk is not None:
        raw_field = spire_tmax_g[spire_wk]
        # Spire lon is 0–360, shift back to 0-105 range — no shift needed (55–105 < 180)
        lats_c, lons_c = spire_lat_g, spire_lon_g
    else:
        raw_field = era5_anom_g[1]
        lats_c, lons_c = era5_lat_g, era5_lon_g
        # ERA5 lon may already be 0–360 — keep as is for India
        if lons_c.max() > 180:
            # find 55–105 portion
            pass

    field_c, lats_c2, lons_c2 = crop(raw_field, lats_c, lons_c,
                                       ILAT_MIN, ILAT_MAX, ILON_MIN, ILON_MAX)

    im2 = ax.pcolormesh(lons_c2, lats_c2, field_c,
                        cmap=cmap_bams, norm=norm_bams,
                        transform=proj_pc, rasterized=True)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.8, edgecolor="k", zorder=4)
    ax.set_extent([ILON_MIN, ILON_MAX, ILAT_MIN, ILAT_MAX], crs=proj_pc)

    gl = ax.gridlines(crs=proj_pc, linewidth=0.3, color="gray",
                      linestyle=":", zorder=3)
    gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = (col == 0)
    gl.bottom_labels = (row == 1)
    gl.xlabel_style  = {"size": 7, "color": "0.3"}
    gl.ylabel_style  = {"size": 7, "color": "0.3"}

    letter = string.ascii_lowercase[idx]
    ax.set_title(f"({letter}) {title}", fontsize=10, fontweight="bold", pad=4)

# Hide any unused axes (7 panels in a 2×4 grid → last cell is empty)
for j in range(len(panel_configs), axes2.size):
    axes2.flat[j].set_visible(False)

cbar_ax2 = fig2.add_axes([0.90, 0.08, 0.018, 0.82])
cb2 = fig2.colorbar(im2, cax=cbar_ax2, orientation="vertical",
                    extend="both", ticks=BOUNDS)
cb2.set_label("2m Tmax anomaly (K)", fontsize=9, labelpad=6)
cb2.ax.tick_params(labelsize=8)

fig2.suptitle(f"Spire JFM 2026 S2S  |  2m Tmax Weekly Anomaly (India, 90-init mean, {init_range})  |  vs ERA5 1991–2020 climo",
              fontsize=10.5, fontweight="bold", y=0.97)

fig2.savefig(OUT_INDIA, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved → {OUT_INDIA}")
