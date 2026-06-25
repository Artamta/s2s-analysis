#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_imd_regions.py — IMD Homogeneous Rainfall Regions Analysis
===================================================================
Publication-quality figures exploring the four IMD homogeneous rainfall
regions using prebuilt region masks, ERA5 reanalysis, and ECMWF S2S
hindcast data.

Figures (saved to ./imd_regions_exploration/):
  01. IMD Region Map — colour-filled regions + state boundaries
  02. Regional JJAS Climatology — filled-contour maps per region
  03. Seasonal Cycle Comparison — ERA5 vs ECMWF W1 (per region)
  04. Interannual Variability — ERA5 vs ECMWF W1 by year
  05. Regional Bias by Lead Week — ECMWF bias vs W1-W6
  06. All-Region Summary Heatmap — bias (mm/day)
  07. Regional Monthly Rainfall Distribution — box plots
  08. Regional Extreme Events — exceedance bar chart
  09. Region-Mean TP Correlation Matrix — 4×4 heatmap
  10. Decadal Shift by Region — 2000-2009 vs 2010-2019

Usage:
    python explore_imd_regions.py             # all figures
    python explore_imd_regions.py --fig 1 5   # specific figures
"""
import os
import sys
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.stats import pearsonr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# =========================================================================== #
#  Config
# =========================================================================== #
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

ECMWF_ROOT = "/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data"

MASK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "masks", "imd_region_masks_1.5deg.nc")

LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

# Region metadata — keys match mask variable names
REGION_META = {
    "northwest_india":      {"label": "Northwest India",  "color": "#e41a1c", "short": "NW"},
    "central_india":        {"label": "Central India",    "color": "#377eb8", "short": "CI"},
    "south_peninsula":      {"label": "South Peninsula",  "color": "#4daf4a", "short": "SP"},
    "east_northeast_india": {"label": "East & NE India",  "color": "#ff7f00", "short": "ENE"},
}
REGION_KEYS = list(REGION_META.keys())

# JJAS init dates (biweekly subset)
JJAS_INITS = ["0601", "0615", "0702", "0716", "0803", "0817"]

# Week definitions (0-based step indices, end exclusive)
WEEK_DEFS = {
    "W1": (0, 7),
    "W2": (7, 14),
    "W3": (14, 21),
    "W4": (21, 28),
    "W5": (28, 35),
    "W6": (35, 42),
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imd_regions_exploration")
os.makedirs(OUT_DIR, exist_ok=True)

HINDCAST_YEARS = np.arange(2000, 2020)


# =========================================================================== #
#  Publication style
# =========================================================================== #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 200,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# =========================================================================== #
#  Colormaps
# =========================================================================== #
def precip_cmap():
    colors = [
        "#ffffff", "#f0f9e8", "#ccebc5", "#a8ddb5", "#7bccc4",
        "#4eb3d3", "#2b8cbe", "#0868ac",
        "#41ab5d", "#78c679", "#addd8e",
        "#f7fcb1", "#fee391", "#fec44f", "#fe9929",
        "#ec7014", "#cc4c02", "#993404",
        "#8c2d04", "#662506",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_sci", colors, N=256)


def anomaly_cmap():
    colors = [
        "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5",
        "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_anom", colors, N=256)


# =========================================================================== #
#  Data loading — ERA5
# =========================================================================== #
def load_era5_india(y0=2000, y1=2019):
    """Load ERA5 daily TP over India from WB2 zarr, in mm/day."""
    print("  [ERA5] Opening zarr...", flush=True)
    ds = xr.open_zarr(WB2_ZARR)
    tp24 = ds["total_precipitation_24hr"]
    # ERA5 lat is ascending → slice(low, high)
    tp24 = tp24.sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))
    tp24 = tp24.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    tp24 = tp24.sel(time=tp24["time.hour"] == 6)
    print(f"  [ERA5] Loading into memory ({tp24.sizes})...", flush=True)
    tp_daily = (tp24 * 1000.0).load()
    tp_daily = tp_daily.transpose("time", "latitude", "longitude")
    print(f"  [ERA5] Loaded! shape={tp_daily.shape}, mem≈{tp_daily.nbytes/1e6:.0f} MB", flush=True)
    return tp_daily


# =========================================================================== #
#  Data loading — Masks
# =========================================================================== #
def load_masks():
    """Load IMD region masks and align to ERA5 grid (lat ascending)."""
    print("  [MASKS] Loading from:", MASK_PATH, flush=True)
    ds = xr.open_dataset(MASK_PATH)
    # Mask file has coords 'lat' (descending 38→6.5) and 'lon' (65→99.5)
    # Rename to match ERA5 coordinate names
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    # Flip latitude to ascending order to match ERA5
    ds = ds.sortby("latitude")
    masks = {}
    for rkey in REGION_KEYS:
        masks[rkey] = ds[rkey].astype(float)
    print(f"  [MASKS] Loaded {len(masks)} regions, grid: lat={ds.latitude.values[[0,-1]]}, "
          f"lon={ds.longitude.values[[0,-1]]}", flush=True)
    return masks


# =========================================================================== #
#  Data loading — ECMWF
# =========================================================================== #
def load_ecmwf_init(mmdd):
    """Load ECMWF CF + PF for a single init date, return combined ensemble.

    Returns accumulated TP DataArray with dims (member, time, step, latitude, longitude).
    """
    cf_path = os.path.join(ECMWF_ROOT, f"tp_cf_{mmdd}.grib")
    pf_path = os.path.join(ECMWF_ROOT, f"tp_pf_{mmdd}.grib")

    ds_cf = xr.open_dataset(cf_path, engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
    tp_cf = ds_cf["tp"].load()

    cf_expanded = tp_cf.expand_dims("member", axis=0)
    cf_expanded["member"] = [0]

    if os.path.exists(pf_path):
        ds_pf = xr.open_dataset(pf_path, engine="cfgrib",
                                backend_kwargs={"indexpath": ""})
        tp_pf = ds_pf["tp"].load()
        pf_renamed = tp_pf.rename({"number": "member"})
        pf_renamed["member"] = np.arange(1, tp_pf.sizes.get("number", 10) + 1)
        combined = xr.concat([cf_expanded, pf_renamed], dim="member")
    else:
        combined = cf_expanded

    return combined


def accum_to_weekly_rate(tp_accum, week_key):
    """Convert accumulated TP to weekly-mean daily rate (mm/day)."""
    s0, s1 = WEEK_DEFS[week_key]
    if s0 == 0:
        weekly_accum = tp_accum.isel(step=s1 - 1)
    else:
        weekly_accum = tp_accum.isel(step=s1 - 1) - tp_accum.isel(step=s0 - 1)
    # TP is in m (or kg/m²), convert to mm/day
    return (weekly_accum * 1000.0) / 7.0


# =========================================================================== #
#  Helpers
# =========================================================================== #
def cos_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": lat})


def area_mean(da, lat_dim="latitude"):
    w = cos_weights(da[lat_dim])
    return da.weighted(w).mean(dim=[lat_dim, "longitude"])


def masked_area_mean(da, mask):
    """Compute cos-weighted area mean of da within mask (mask=1)."""
    # Align mask to da grid via nearest-neighbor interpolation
    mask_aligned = mask.interp(latitude=da.latitude, longitude=da.longitude,
                               method="nearest").fillna(0)
    masked = da.where(mask_aligned > 0.5)
    w = cos_weights(da.latitude)
    return masked.weighted(w).mean(dim=["latitude", "longitude"], skipna=True)


def smooth_field(data, sigma=0.8):
    return gaussian_filter(np.nan_to_num(data, nan=0.0), sigma=sigma)


def savefig(fig, name, dpi=250):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ saved {path} ({size_kb:.0f} KB)")


def add_map_features(ax, borders=True, rivers=False, gridlines=True):
    ax.coastlines(resolution="50m", linewidth=0.6, color="#333333")
    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
    if rivers:
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#4a90d9", alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor="#f9f6f0", alpha=0.15, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f0f8", alpha=0.3, zorder=0)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
    return ax


def india_extent(ax):
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=ccrs.PlateCarree())


def panel_label(ax, label, x=0.02, y=0.95):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =========================================================================== #
#  FIGURE 1: IMD Region Map
# =========================================================================== #
def fig01(masks, **kw):
    """4-region colour-filled map with state boundaries."""
    print("Fig 01: IMD Region Map...", flush=True)
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={"projection": proj})
    india_extent(ax)
    add_map_features(ax, rivers=True)

    # Build a composite region field: 0=none, 1=NW, 2=CI, 3=SP, 4=ENE
    ref_mask = masks[REGION_KEYS[0]]
    composite = xr.zeros_like(ref_mask, dtype=float)
    region_colors = []
    for i, rkey in enumerate(REGION_KEYS, 1):
        m = masks[rkey]
        composite = composite + m * i
        region_colors.append(REGION_META[rkey]["color"])

    # Custom discrete colormap
    cmap_list = ["#f0f0f0"] + region_colors
    cmap_discrete = mcolors.ListedColormap(cmap_list)
    bounds = np.arange(-0.5, len(REGION_KEYS) + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

    cf = ax.pcolormesh(composite.longitude, composite.latitude, composite.values,
                       cmap=cmap_discrete, norm=norm, transform=proj, zorder=2)

    # Add state boundaries via NaturalEarth
    states = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "50m",
        edgecolor="#555555", facecolor="none", linewidth=0.5)
    ax.add_feature(states, zorder=3)

    # Region name labels
    label_pos = {
        "northwest_india":      (74.0, 30.0),
        "central_india":        (80.0, 24.0),
        "south_peninsula":      (78.0, 13.0),
        "east_northeast_india": (91.0, 25.0),
    }
    for rkey in REGION_KEYS:
        lon_c, lat_c = label_pos[rkey]
        meta = REGION_META[rkey]
        ax.text(lon_c, lat_c, meta["label"], transform=proj,
                fontsize=10, fontweight="bold", ha="center", va="center",
                color="white", zorder=5,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=meta["color"]),
                              pe.withStroke(linewidth=5, foreground="black", alpha=0.3)])

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=REGION_META[rk]["color"], edgecolor="k", lw=0.5,
                     label=REGION_META[rk]["label"]) for rk in REGION_KEYS]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95,
              title="IMD Regions", title_fontsize=10)

    ax.set_title("IMD Homogeneous Rainfall Regions of India", fontsize=14, pad=12)
    savefig(fig, "fig01_imd_region_map.png")


# =========================================================================== #
#  FIGURE 2: Regional JJAS Climatology
# =========================================================================== #
def fig02(era5, masks, **kw):
    """4-panel JJAS-mean filled contour maps, masked to each region."""
    print("Fig 02: Regional JJAS Climatology...", flush=True)
    jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))
    jjas_clim = jjas.mean("time")

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={"projection": proj})
    labels = ["(a)", "(b)", "(c)", "(d)"]
    levels = np.arange(0, 18.1, 1.0)

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        india_extent(ax)
        add_map_features(ax, rivers=False)
        meta = REGION_META[rkey]

        # Align mask to ERA5 grid
        mask_aligned = masks[rkey].interp(latitude=jjas_clim.latitude,
                                          longitude=jjas_clim.longitude,
                                          method="nearest").fillna(0)
        masked_data = jjas_clim.where(mask_aligned > 0.5)
        sm = smooth_field(masked_data.values, sigma=0.5)

        cf = ax.contourf(jjas_clim.longitude, jjas_clim.latitude, sm,
                         levels=levels, cmap=precip_cmap(), extend="max",
                         transform=proj)
        ax.contour(jjas_clim.longitude, jjas_clim.latitude, sm,
                   levels=levels[::3], colors="k", linewidths=0.3, alpha=0.4,
                   transform=proj)

        # Draw region boundary contour
        ax.contour(mask_aligned.longitude, mask_aligned.latitude,
                   mask_aligned.values, levels=[0.5], colors=meta["color"],
                   linewidths=1.5, transform=proj, zorder=5)

        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"], pad=8)
        panel_label(ax, lbl)

    # Shared colorbar
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("JJAS Mean Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle("JJAS Precipitation Climatology by IMD Region (2000–2019)",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    savefig(fig, "fig02_regional_jjas_clim.png")


# =========================================================================== #
#  FIGURE 3: Seasonal Cycle Comparison — ERA5 vs ECMWF W1
# =========================================================================== #
def fig03(era5, masks, ecmwf_cache, **kw):
    """4 panels: ERA5 daily seasonal cycle + ECMWF W1 dots per init."""
    print("Fig 03: Seasonal Cycle Comparison...", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        meta = REGION_META[rkey]

        # ERA5: daily climatological seasonal cycle (2000-2019)
        era5_rmean = masked_area_mean(era5, masks[rkey])
        # Group by day-of-year for JJAS window (DOY 152-273 ≈ Jun 1 – Sep 30)
        doy_grp = era5_rmean.groupby("time.dayofyear")
        era5_doy_mean = doy_grp.mean()

        # Smooth with 15-day window
        k = 15
        era5_smooth = era5_doy_mean.copy()
        era5_smooth.values[:] = uniform_filter1d(era5_doy_mean.values, k, mode="wrap")

        # Plot ERA5 seasonal cycle (JJAS portion: DOY 152-273)
        jjas_doys = era5_smooth.sel(dayofyear=slice(140, 285))
        ax.plot(jjas_doys.dayofyear.values, jjas_doys.values,
                color=meta["color"], lw=2.5, label="ERA5 climatology", zorder=5)

        # ECMWF W1: for each init, compute region mean
        ecmwf_vals = []
        ecmwf_doys = []
        for mmdd in JJAS_INITS:
            if mmdd not in ecmwf_cache:
                continue
            tp_combined = ecmwf_cache[mmdd]
            # W1 rate per member, year
            w1_rate = accum_to_weekly_rate(tp_combined, "W1")
            # Crop to India domain (ECMWF lat is descending 49.5→0)
            w1_india = w1_rate.sel(latitude=slice(LAT_N, LAT_S),
                                  longitude=slice(LON_W, LON_E))
            # Region mean per member, year → then mean over members and years
            rmean = masked_area_mean(w1_india, masks[rkey])
            ecmwf_mean = float(rmean.mean().values)
            ecmwf_std = float(rmean.mean("member").std("time").values)

            # DOY for midpoint of W1 (init + 3.5 days)
            month = int(mmdd[:2])
            day = int(mmdd[2:])
            init_date = datetime.date(2000, month, day)
            mid_doy = (init_date + datetime.timedelta(days=3)).timetuple().tm_yday

            ecmwf_vals.append((mid_doy, ecmwf_mean, ecmwf_std))

        if ecmwf_vals:
            edoys = [v[0] for v in ecmwf_vals]
            emeans = [v[1] for v in ecmwf_vals]
            estds = [v[2] for v in ecmwf_vals]
            ax.errorbar(edoys, emeans, yerr=estds, fmt="o", ms=7,
                        color="#d62728", capsize=4, capthick=1.5, lw=1.5,
                        label="ECMWF W1", zorder=6, markeredgecolor="white",
                        markeredgewidth=0.8)

        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"])
        ax.set_xlim(140, 285)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # DOY → month labels
        month_doys = [152, 182, 213, 244, 274]
        month_lbls = ["Jun", "Jul", "Aug", "Sep", "Oct"]
        ax.set_xticks(month_doys)
        ax.set_xticklabels(month_lbls)
        if lbl in ["(a)", "(b)"]:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        panel_label(ax, lbl, x=0.02, y=0.93)

    for ax in axes[-1]:
        ax.set_xlabel("Month")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precipitation (mm day⁻¹)")

    fig.suptitle("JJAS Seasonal Cycle: ERA5 vs ECMWF Week-1 (2000–2019)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig03_seasonal_cycle_comparison.png")


# =========================================================================== #
#  FIGURE 4: Regional Interannual Variability
# =========================================================================== #
def fig04(era5, masks, ecmwf_cache, **kw):
    """4 panels: JJAS-mean TP by year, ERA5 bars vs ECMWF W1 dots."""
    print("Fig 04: Regional Interannual Variability...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    years = HINDCAST_YEARS

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        meta = REGION_META[rkey]

        # ERA5 yearly JJAS mean
        era5_rmean = masked_area_mean(era5_jjas, masks[rkey])
        era5_yearly = era5_rmean.groupby("time.year").mean("time")
        era5_vals = era5_yearly.sel(year=years).values

        # Bars for ERA5
        ax.bar(years, era5_vals, color=meta["color"], alpha=0.6, width=0.75,
               edgecolor="white", lw=0.3, label="ERA5 JJAS", zorder=3)

        # ECMWF W1: JJAS-mean per year (average over inits)
        ecmwf_yearly_means = []
        ecmwf_yearly_stds = []
        for yr in years:
            yr_vals = []
            for mmdd in JJAS_INITS:
                if mmdd not in ecmwf_cache:
                    continue
                tp_combined = ecmwf_cache[mmdd]
                w1_rate = accum_to_weekly_rate(tp_combined, "W1")
                w1_india = w1_rate.sel(latitude=slice(LAT_N, LAT_S),
                                      longitude=slice(LON_W, LON_E))
                # Select this year
                yr_idx = np.where(tp_combined.time.dt.year == yr)[0]
                if len(yr_idx) == 0:
                    continue
                w1_yr = w1_india.isel(time=yr_idx[0])  # (member, lat, lon)
                rmean = masked_area_mean(w1_yr, masks[rkey])
                yr_vals.append(rmean.values)  # (member,)
            if yr_vals:
                all_members = np.concatenate([np.atleast_1d(v) for v in yr_vals])
                ecmwf_yearly_means.append(np.mean(all_members))
                ecmwf_yearly_stds.append(np.std(all_members))
            else:
                ecmwf_yearly_means.append(np.nan)
                ecmwf_yearly_stds.append(np.nan)

        ecmwf_means = np.array(ecmwf_yearly_means)
        ecmwf_stds = np.array(ecmwf_yearly_stds)

        ax.errorbar(years, ecmwf_means, yerr=ecmwf_stds, fmt="D", ms=5,
                    color="#d62728", capsize=3, capthick=1, lw=1,
                    label="ECMWF W1", zorder=6, markeredgecolor="white",
                    markeredgewidth=0.5)

        # Correlation
        valid = ~np.isnan(ecmwf_means) & ~np.isnan(era5_vals)
        if valid.sum() > 5:
            r, p = pearsonr(era5_vals[valid], ecmwf_means[valid])
            ax.text(0.97, 0.95, f"r = {r:.2f}\np = {p:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=meta["color"], alpha=0.9))

        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if lbl in ["(a)", "(b)"]:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        panel_label(ax, lbl, x=0.02, y=0.93)

    for ax in axes[-1]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("JJAS Mean Precipitation (mm day⁻¹)")

    fig.suptitle("Interannual JJAS Precipitation: ERA5 vs ECMWF W1 (2000–2019)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig04_interannual_variability.png")


# =========================================================================== #
#  FIGURE 5: Regional Bias by Lead Week
# =========================================================================== #
def fig05(era5, masks, ecmwf_cache, **kw):
    """4 panels: ECMWF bias (ECMWF−ERA5) vs lead week W1-W6."""
    print("Fig 05: Regional Bias by Lead Week...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    week_names = list(WEEK_DEFS.keys())

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        meta = REGION_META[rkey]
        biases_mean = []
        biases_std = []

        for wk in week_names:
            s0, s1 = WEEK_DEFS[wk]
            if s1 > 46:
                biases_mean.append(np.nan)
                biases_std.append(np.nan)
                continue

            all_biases = []
            for mmdd in JJAS_INITS:
                if mmdd not in ecmwf_cache:
                    continue
                tp_combined = ecmwf_cache[mmdd]
                wk_rate = accum_to_weekly_rate(tp_combined, wk)
                wk_india = wk_rate.sel(latitude=slice(LAT_N, LAT_S),
                                       longitude=slice(LON_W, LON_E))
                for yr in HINDCAST_YEARS:
                    yr_idx = np.where(tp_combined.time.dt.year == yr)[0]
                    if len(yr_idx) == 0:
                        continue
                    wk_yr = wk_india.isel(time=yr_idx[0])
                    ecmwf_rmean = float(masked_area_mean(wk_yr, masks[rkey]).mean("member").values)

                    # ERA5 truth for corresponding period
                    month = int(mmdd[:2])
                    day = int(mmdd[2:])
                    init_date = datetime.date(yr, month, day)
                    start = init_date + datetime.timedelta(days=s0)
                    end = init_date + datetime.timedelta(days=s1 - 1)
                    t0 = np.datetime64(f"{yr}-{start.month:02d}-{start.day:02d}")
                    t1 = np.datetime64(f"{yr}-{end.month:02d}-{end.day:02d}")
                    try:
                        era5_sub = era5.sel(time=slice(t0, t1))
                        if era5_sub.sizes["time"] < 3:
                            continue
                        era5_wk = float(masked_area_mean(era5_sub, masks[rkey]).mean("time").values)
                    except Exception:
                        continue

                    all_biases.append(ecmwf_rmean - era5_wk)

            if all_biases:
                biases_mean.append(np.mean(all_biases))
                biases_std.append(np.std(all_biases))
            else:
                biases_mean.append(np.nan)
                biases_std.append(np.nan)

        x = np.arange(len(week_names))
        ax.bar(x, biases_mean, yerr=biases_std, width=0.6,
               color=meta["color"], alpha=0.7, edgecolor="white", capsize=5,
               error_kw={"lw": 1.5}, zorder=3)
        ax.axhline(0, color="k", lw=1, ls="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(week_names)
        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"])
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl, x=0.02, y=0.93)

    for ax in axes[-1]:
        ax.set_xlabel("Lead Week")
    for ax in axes[:, 0]:
        ax.set_ylabel("Bias: ECMWF − ERA5 (mm day⁻¹)")

    fig.suptitle("ECMWF Precipitation Bias vs Lead Week by Region (JJAS, 2000–2019)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig05_bias_by_lead_week.png")


# =========================================================================== #
#  FIGURE 6: All-Region Summary Heatmap
# =========================================================================== #
def fig06(era5, masks, ecmwf_cache, **kw):
    """Heatmap: rows = 4 regions + All India, cols = W1-W6, cell = bias."""
    print("Fig 06: Summary Bias Heatmap...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))
    week_names = list(WEEK_DEFS.keys())

    # Regions + All India
    row_keys = REGION_KEYS + ["all_india"]
    row_labels = [REGION_META[rk]["label"] for rk in REGION_KEYS] + ["All India"]

    bias_matrix = np.full((len(row_keys), len(week_names)), np.nan)

    for ri, rkey in enumerate(row_keys):
        for wi, wk in enumerate(week_names):
            s0, s1 = WEEK_DEFS[wk]
            if s1 > 46:
                continue
            all_biases = []
            for mmdd in JJAS_INITS:
                if mmdd not in ecmwf_cache:
                    continue
                tp_combined = ecmwf_cache[mmdd]
                wk_rate = accum_to_weekly_rate(tp_combined, wk)
                wk_india = wk_rate.sel(latitude=slice(LAT_N, LAT_S),
                                       longitude=slice(LON_W, LON_E))
                for yr in HINDCAST_YEARS:
                    yr_idx = np.where(tp_combined.time.dt.year == yr)[0]
                    if len(yr_idx) == 0:
                        continue
                    wk_yr = wk_india.isel(time=yr_idx[0])

                    if rkey == "all_india":
                        ecmwf_rmean = float(area_mean(wk_yr.mean("member")).values)
                    else:
                        ecmwf_rmean = float(masked_area_mean(
                            wk_yr, masks[rkey]).mean("member").values)

                    month = int(mmdd[:2])
                    day = int(mmdd[2:])
                    init_date = datetime.date(yr, month, day)
                    start = init_date + datetime.timedelta(days=s0)
                    end = init_date + datetime.timedelta(days=s1 - 1)
                    t0 = np.datetime64(f"{yr}-{start.month:02d}-{start.day:02d}")
                    t1 = np.datetime64(f"{yr}-{end.month:02d}-{end.day:02d}")
                    try:
                        era5_sub = era5.sel(time=slice(t0, t1))
                        if era5_sub.sizes["time"] < 3:
                            continue
                        if rkey == "all_india":
                            era5_sub_india = era5_sub.sel(
                                latitude=slice(LAT_S, LAT_N),
                                longitude=slice(LON_W, LON_E))
                            era5_wk = float(area_mean(era5_sub_india).mean("time").values)
                        else:
                            era5_wk = float(masked_area_mean(
                                era5_sub, masks[rkey]).mean("time").values)
                    except Exception:
                        continue
                    all_biases.append(ecmwf_rmean - era5_wk)

            if all_biases:
                bias_matrix[ri, wi] = np.mean(all_biases)

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.nanmax(np.abs(bias_matrix))
    vmax = max(vmax, 1.0)
    im = ax.imshow(bias_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    # Annotate
    for i in range(bias_matrix.shape[0]):
        for j in range(bias_matrix.shape[1]):
            val = bias_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(week_names)))
    ax.set_xticklabels(week_names, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Lead Week", fontsize=12)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Bias: ECMWF − ERA5 (mm day⁻¹)", fontsize=11)

    ax.set_title("ECMWF JJAS Precipitation Bias by Region and Lead Week (2000–2019)",
                 fontsize=13, fontweight="bold", pad=12)
    savefig(fig, "fig06_bias_heatmap.png")


# =========================================================================== #
#  FIGURE 7: Regional Monthly Rainfall Distribution
# =========================================================================== #
def fig07(era5, masks, **kw):
    """4 panels: monthly box plots of regional-mean TP."""
    print("Fig 07: Regional Monthly Rainfall Distribution...", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        meta = REGION_META[rkey]
        rmean = masked_area_mean(era5, masks[rkey])

        monthly_data = []
        for m in range(1, 13):
            monthly_vals = rmean.sel(time=rmean["time.month"] == m).values
            monthly_data.append(monthly_vals[~np.isnan(monthly_vals)])

        bp = ax.boxplot(monthly_data, positions=range(1, 13), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=meta["color"], lw=1.2),
                        capprops=dict(color=meta["color"], lw=1.2))

        for patch in bp["boxes"]:
            patch.set_facecolor(meta["color"])
            patch.set_alpha(0.7)
            patch.set_edgecolor("white")

        # JJAS highlight
        ax.axvspan(5.5, 9.5, alpha=0.08, color="#2ca02c", zorder=0)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl, x=0.02, y=0.93)

    for ax in axes[-1]:
        ax.set_xlabel("Month")
    for ax in axes[:, 0]:
        ax.set_ylabel("Daily Precipitation (mm day⁻¹)")

    fig.suptitle("Monthly Precipitation Distribution by IMD Region (2000–2019)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig07_monthly_distribution.png")


# =========================================================================== #
#  FIGURE 8: Regional Extreme Events
# =========================================================================== #
def fig08(era5, masks, **kw):
    """4 panels: % of JJAS days exceeding thresholds."""
    print("Fig 08: Regional Extreme Events...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))
    thresholds = [1, 5, 10, 20]
    thresh_labels = [f">{t} mm/day" for t in thresholds]
    thresh_colors = ["#a6d854", "#66c2a5", "#3288bd", "#5e4fa2"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, rkey, lbl in zip(axes.flat, REGION_KEYS, labels):
        meta = REGION_META[rkey]
        rmean = masked_area_mean(era5_jjas, masks[rkey])

        percentages = []
        for thresh in thresholds:
            pct = float((rmean > thresh).mean().values * 100)
            percentages.append(pct)

        x = np.arange(len(thresholds))
        bars = ax.bar(x, percentages, color=thresh_colors, width=0.6,
                      edgecolor="white", lw=0.5, zorder=3)

        # Annotate bars
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(thresh_labels, fontsize=9)
        ax.set_title(meta["label"], fontsize=12, fontweight="bold",
                     color=meta["color"])
        ax.set_ylabel("% of JJAS days")
        ax.set_ylim(0, max(percentages) * 1.25 if percentages else 100)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl, x=0.02, y=0.93)

    fig.suptitle("JJAS Extreme Precipitation Frequency by Region (ERA5, 2000–2019)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig08_extreme_events.png")


# =========================================================================== #
#  FIGURE 9: Region-Mean TP Correlation Matrix
# =========================================================================== #
def fig09(era5, masks, **kw):
    """4×4 correlation matrix of JJAS yearly mean TP between regions."""
    print("Fig 09: Region-Mean Correlation Matrix...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))
    n_regions = len(REGION_KEYS)
    corr_matrix = np.ones((n_regions, n_regions))
    pval_matrix = np.zeros((n_regions, n_regions))

    # Compute yearly means for each region
    yearly_means = {}
    for rkey in REGION_KEYS:
        rmean = masked_area_mean(era5_jjas, masks[rkey])
        yearly = rmean.groupby("time.year").mean("time")
        yearly_means[rkey] = yearly.values

    # Pairwise correlations
    for i, rk_i in enumerate(REGION_KEYS):
        for j, rk_j in enumerate(REGION_KEYS):
            if i != j:
                valid = ~np.isnan(yearly_means[rk_i]) & ~np.isnan(yearly_means[rk_j])
                if valid.sum() > 5:
                    r, p = pearsonr(yearly_means[rk_i][valid],
                                    yearly_means[rk_j][valid])
                    corr_matrix[i, j] = r
                    pval_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    region_labels_short = [REGION_META[rk]["label"] for rk in REGION_KEYS]

    # Annotate
    for i in range(n_regions):
        for j in range(n_regions):
            val = corr_matrix[i, j]
            p = pval_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            if i == j:
                sig = ""
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(region_labels_short, fontsize=10, rotation=25, ha="right")
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(region_labels_short, fontsize=10)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Pearson r", fontsize=11)

    ax.set_title("Inter-Region Correlation of JJAS Mean Precipitation\n(ERA5, 2000–2019)",
                 fontsize=13, fontweight="bold", pad=12)

    # Significance note
    ax.text(0.5, -0.12, "* p<0.1  ** p<0.05  *** p<0.01",
            transform=ax.transAxes, fontsize=9, ha="center", fontstyle="italic",
            color="#666666")

    savefig(fig, "fig09_correlation_matrix.png")


# =========================================================================== #
#  FIGURE 10: Decadal Shift by Region
# =========================================================================== #
def fig10(era5, masks, **kw):
    """Paired bar chart: 2000-2009 vs 2010-2019 JJAS-mean per region."""
    print("Fig 10: Decadal Shift...", flush=True)

    era5_jjas = era5.sel(time=era5["time.month"].isin([6, 7, 8, 9]))

    dec1_vals = []
    dec2_vals = []
    dec1_stds = []
    dec2_stds = []

    for rkey in REGION_KEYS:
        rmean = masked_area_mean(era5_jjas, masks[rkey])
        yearly = rmean.groupby("time.year").mean("time")

        d1 = yearly.sel(year=slice(2000, 2009)).values
        d2 = yearly.sel(year=slice(2010, 2019)).values
        dec1_vals.append(np.nanmean(d1))
        dec2_vals.append(np.nanmean(d2))
        dec1_stds.append(np.nanstd(d1))
        dec2_stds.append(np.nanstd(d2))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(REGION_KEYS))
    width = 0.35

    bars1 = ax.bar(x - width/2, dec1_vals, width, yerr=dec1_stds,
                   color="#4393c3", alpha=0.8, edgecolor="white", lw=0.5,
                   capsize=5, label="2000–2009", zorder=3)
    bars2 = ax.bar(x + width/2, dec2_vals, width, yerr=dec2_stds,
                   color="#d6604d", alpha=0.8, edgecolor="white", lw=0.5,
                   capsize=5, label="2010–2019", zorder=3)

    # Annotate with difference
    for i, (v1, v2) in enumerate(zip(dec1_vals, dec2_vals)):
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        color = "#2ca02c" if diff > 0 else "#d62728"
        ypos = max(v1, v2) + max(dec1_stds[i], dec2_stds[i]) + 0.3
        ax.text(i, ypos, f"Δ = {diff:+.2f}\n({pct:+.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([REGION_META[rk]["label"] for rk in REGION_KEYS],
                       fontsize=11)
    ax.set_ylabel("JJAS Mean Precipitation (mm day⁻¹)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Decadal Shift in JJAS Precipitation by Region (ERA5)",
                 fontsize=14, fontweight="bold", pad=12)
    savefig(fig, "fig10_decadal_shift.png")


# =========================================================================== #
#  ECMWF loader (with caching)
# =========================================================================== #
def load_ecmwf_inits(init_list):
    """Load and cache ECMWF combined (CF+PF) for each init date."""
    cache = {}
    for mmdd in init_list:
        cf_path = os.path.join(ECMWF_ROOT, f"tp_cf_{mmdd}.grib")
        if not os.path.exists(cf_path):
            print(f"  [ECMWF] Skipping {mmdd} — file not found", flush=True)
            continue
        print(f"  [ECMWF] Loading {mmdd}...", end=" ", flush=True)
        try:
            combined = load_ecmwf_init(mmdd)
            cache[mmdd] = combined
            print(f"shape={combined.shape}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
    return cache


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="IMD Region Analysis Figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()
    figs_requested = set(args.fig) if args.fig else set(range(1, 11))

    print(f"{'═'*60}")
    print(f"  IMD Homogeneous Rainfall Regions — Publication Figures")
    print(f"{'═'*60}")
    print(f"  WB2 zarr: {WB2_ZARR}")
    print(f"  ECMWF root: {ECMWF_ROOT}")
    print(f"  Mask file: {MASK_PATH}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Figures: {sorted(figs_requested)}")
    print()

    # Load masks
    masks = load_masks()
    print()

    # Load ERA5 (needed by all figures except fig01)
    era5 = None
    if figs_requested - {1}:
        era5 = load_era5_india(y0=2000, y1=2019)
        print()

    # Load ECMWF (needed by figs 3, 4, 5, 6)
    ecmwf_cache = {}
    ecmwf_figs = {3, 4, 5, 6}
    if figs_requested & ecmwf_figs:
        print("  Loading ECMWF hindcast data...", flush=True)
        ecmwf_cache = load_ecmwf_inits(JJAS_INITS)
        print(f"  [ECMWF] Loaded {len(ecmwf_cache)} init dates.\n", flush=True)

    # Figure dispatch
    fig_funcs = {
        1: fig01,
        2: fig02,
        3: fig03,
        4: fig04,
        5: fig05,
        6: fig06,
        7: fig07,
        8: fig08,
        9: fig09,
        10: fig10,
    }

    for n in sorted(figs_requested):
        if n in fig_funcs:
            try:
                fig_funcs[n](era5=era5, masks=masks, ecmwf_cache=ecmwf_cache)
            except Exception as e:
                print(f"  ✗ Fig {n:02d} FAILED: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"  ⚠ Unknown figure number: {n}")

    print(f"\n{'═'*60}")
    print(f"  Done! {len(figs_requested)} figures → {OUT_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
