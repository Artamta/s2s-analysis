#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_tp_publication.py — Publication-Quality ERA5 Precipitation Analysis
============================================================================
Generates journal-ready figures (npj Climate / GRL / Nature Comms level)
for understanding Indian precipitation patterns from WeatherBench2 ERA5.

Features:
  • Cartopy projections with coastlines, state borders, rivers
  • Scientific colormaps, contour overlays, hatching for significance
  • Gaussian-smoothed fields, proper anomaly statistics
  • Comprehensive analysis: climatology, intraseasonal variability,
    active-break spells, monsoon progression, extremes, teleconnections

Figures (saved to ./tp_exploration_pub/):
  01. JJAS & JFM climatology — filled contours + topographic shading
  02. Monthly march of the monsoon — 12-panel contour maps
  03. All-India seasonal cycle — ribbon plot with percentiles
  04. Interannual JJAS variability — bar + running mean + trend
  05. Hovmöller — monsoon onset/withdrawal (lat–time)
  06. Wet/Dry year composites with significance hatching
  07. Extreme precipitation frequency + intensity
  08. Regional seasonal cycles — 4 IMD regions, stacked
  09. Intraseasonal variability — std of 30–60 day bandpass
  10. Precipitation intensity PDF — distribution by region
  11. JJAS trend map — spatial pattern of long-term TP trends
  12. Monsoon onset pentad map — mean onset timing across India

Usage:
    python explore_tp_publication.py             # all figures
    python explore_tp_publication.py --fig 1 5   # specific figures
"""
import os
import sys
import argparse
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
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# =========================================================================== #
#  Config
# =========================================================================== #
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

IMD_REGIONS = {
    "Northwest India":  {"lat": (25.0, 36.0), "lon": (68.0, 80.0), "color": "#e41a1c"},
    "Central India":    {"lat": (20.0, 28.0), "lon": (74.0, 86.0), "color": "#377eb8"},
    "South Peninsula":  {"lat": ( 8.0, 20.0), "lon": (74.0, 82.0), "color": "#4daf4a"},
    "East & NE India":  {"lat": (20.0, 30.0), "lon": (86.0, 98.0), "color": "#ff7f00"},
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp_exploration_pub")
os.makedirs(OUT_DIR, exist_ok=True)

CLIM_Y0, CLIM_Y1 = 1991, 2020
FULL_Y0, FULL_Y1 = 1959, 2022  # full analysis range


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
    """Rich scientific precipitation colormap (white→blue→green→yellow→red→purple)."""
    colors = [
        "#ffffff", "#f0f9e8", "#ccebc5", "#a8ddb5", "#7bccc4",
        "#4eb3d3", "#2b8cbe", "#0868ac",
        "#41ab5d", "#78c679", "#addd8e",
        "#f7fcb1", "#fee391", "#fec44f", "#fe9929",
        "#ec7014", "#cc4c02", "#993404",
        "#8c2d04", "#662506",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_sci", colors, N=256)


def precip_cmap_compact():
    """Compact version for monthly panels."""
    colors = [
        "#f7fbff", "#d0e1f2", "#94c4df", "#4a98c9",
        "#1764ab", "#2d904e", "#78c679",
        "#fedd6e", "#feb24c", "#fd8d3c",
        "#e31a1c", "#b10026", "#67001f",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_c", colors, N=256)


def anomaly_cmap():
    """Brown–white–teal diverging colormap (BrBG-like)."""
    colors = [
        "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5",
        "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_anom", colors, N=256)


def trend_cmap():
    """Red–white–blue for trend maps."""
    colors = [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
        "#f7f7f7",
        "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
    ]
    return mcolors.LinearSegmentedColormap.from_list("trend", colors, N=256)


# =========================================================================== #
#  Data loading
# =========================================================================== #
def load_wb2_tp_india(y0=None, y1=None):
    """Load WB2 ERA5 daily TP over India, eagerly into memory."""
    print("  [1/3] Opening zarr...")
    ds = xr.open_zarr(WB2_ZARR)
    tp24 = ds["total_precipitation_24hr"]
    tp24 = tp24.sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))

    t0 = f"{y0}-01-01" if y0 else None
    t1 = f"{y1}-12-31" if y1 else None
    if t0 or t1:
        tp24 = tp24.sel(time=slice(t0, t1))

    tp24 = tp24.sel(time=tp24["time.hour"] == 6)
    print(f"  [2/3] Loading into memory ({tp24.sizes})...")
    tp_daily = (tp24 * 1000.0).load()
    tp_daily = tp_daily.transpose("time", "latitude", "longitude")
    print(f"  [3/3] Loaded! shape={tp_daily.shape}, mem≈{tp_daily.nbytes/1e6:.0f} MB")
    return tp_daily


# =========================================================================== #
#  Helpers
# =========================================================================== #
def cos_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": lat})


def area_mean(da, lat_dim="latitude"):
    w = cos_weights(da[lat_dim])
    return da.weighted(w).mean(dim=[lat_dim, "longitude"])


def region_mean(da, lat_range, lon_range):
    sub = da.sel(latitude=slice(lat_range[0], lat_range[1]),
                 longitude=slice(lon_range[0], lon_range[1]))
    return area_mean(sub)


def smooth_field(data, sigma=0.8):
    """Gaussian smooth a 2D field for contour plotting."""
    return gaussian_filter(np.nan_to_num(data, nan=0.0), sigma=sigma)


def savefig(fig, name, dpi=250):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {path}")


def add_map_features(ax, borders=True, rivers=False, gridlines=True):
    """Add coastlines, borders, and optional gridlines to a cartopy axis."""
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
    """Add (a), (b), etc. panel labels."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


# =========================================================================== #
#  FIGURE 1: JJAS & JFM climatology with filled contours
# =========================================================================== #
def fig01(tp):
    print("Fig 01: JJAS & JFM seasonal climatology...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    jjas = clim.sel(time=clim["time.month"].isin([6, 7, 8, 9])).mean("time")
    jfm  = clim.sel(time=clim["time.month"].isin([1, 2, 3])).mean("time")

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), subplot_kw={"projection": proj})

    configs = [
        (jjas, "JJAS (Jun–Sep)", np.arange(0, 20.1, 1.0), "(a)"),
        (jfm,  "JFM (Jan–Mar)", np.arange(0, 6.1, 0.5),  "(b)"),
    ]

    for ax, (data, title, levels, lbl) in zip(axes, configs):
        india_extent(ax)
        add_map_features(ax, rivers=True)
        lon, lat = data.longitude, data.latitude
        sm = smooth_field(data.values, sigma=0.6)
        cf = ax.contourf(lon, lat, sm, levels=levels, cmap=precip_cmap(),
                         extend="max", transform=proj)
        cs = ax.contour(lon, lat, sm, levels=levels[::3], colors="k",
                        linewidths=0.3, alpha=0.4, transform=proj)
        ax.set_title(title, fontsize=13, pad=10)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                          pad=0.06, aspect=30)
        cb.set_label("Precipitation (mm day⁻¹)", fontsize=10)
        cb.ax.tick_params(labelsize=8)
        panel_label(ax, lbl)

        # IMD region boxes
        for rname, rbox in IMD_REGIONS.items():
            lat0, lat1 = rbox["lat"]
            lon0, lon1 = rbox["lon"]
            ax.plot([lon0, lon1, lon1, lon0, lon0],
                    [lat0, lat0, lat1, lat1, lat0],
                    color=rbox["color"], lw=1.2, ls="--", alpha=0.8,
                    transform=proj)

    fig.suptitle(f"ERA5 Precipitation Climatology ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=15, fontweight="bold", y=1.02)
    savefig(fig, "fig01_seasonal_clim_pub.png")


# =========================================================================== #
#  FIGURE 2: Monthly march of the monsoon (12-panel)
# =========================================================================== #
def fig02(tp):
    print("Fig 02: Monthly climatology panels...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    monthly = clim.groupby("time.month").mean("time")
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(3, 4, figsize=(16, 14), subplot_kw={"projection": proj})
    cmap = precip_cmap_compact()

    for i, ax in enumerate(axes.flat):
        m = i + 1
        india_extent(ax)
        add_map_features(ax, gridlines=False, rivers=False)
        data = monthly.sel(month=m)
        vmax = 18 if m in [6, 7, 8, 9] else 8
        levels = np.linspace(0, vmax, 13)
        sm = smooth_field(data.values, sigma=0.5)
        cf = ax.contourf(data.longitude, data.latitude, sm, levels=levels,
                         cmap=cmap, extend="max", transform=proj)
        ax.set_title(MONTH_NAMES[i], fontsize=12, fontweight="bold", pad=5)
        panel_label(ax, f"({chr(97+i)})", x=0.03, y=0.97)

    # shared colorbar
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle(f"Monthly Mean Precipitation over India ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    savefig(fig, "fig02_monthly_clim_pub.png")


# =========================================================================== #
#  FIGURE 3: All-India seasonal cycle — ribbon with percentiles
# =========================================================================== #
def fig03(tp):
    print("Fig 03: All-India seasonal cycle...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    daily_ai = area_mean(clim)
    doy_grp = daily_ai.groupby("time.dayofyear")

    mean = doy_grp.mean()
    p05 = doy_grp.quantile(0.05)
    p25 = doy_grp.quantile(0.25)
    p75 = doy_grp.quantile(0.75)
    p95 = doy_grp.quantile(0.95)

    k = 15
    for da in [mean, p05, p25, p75, p95]:
        da.values[:] = uniform_filter1d(da.values, k, mode="wrap")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    doy = mean.dayofyear.values

    ax.fill_between(doy, p05.values, p95.values, alpha=0.12, color="#2171b5",
                    label="5th–95th pctl")
    ax.fill_between(doy, p25.values, p75.values, alpha=0.25, color="#2171b5",
                    label="25th–75th pctl (IQR)")
    ax.plot(doy, mean.values, color="#08519c", lw=2.5, label="Mean", zorder=5)

    # JJAS window
    ax.axvspan(152, 273, alpha=0.06, color="#2ca02c", zorder=0)
    ax.text(212, ax.get_ylim()[1]*0.92, "JJAS", fontsize=11, fontweight="bold",
            color="#2ca02c", ha="center", alpha=0.7)
    # JFM window
    ax.axvspan(1, 90, alpha=0.06, color="#ff7f0e", zorder=0)
    ax.text(45, ax.get_ylim()[1]*0.92, "JFM", fontsize=11, fontweight="bold",
            color="#ff7f0e", ha="center", alpha=0.7)

    ax.set_xlabel("Month")
    ax.set_ylabel("Precipitation (mm day⁻¹)")
    ax.set_title(f"All-India Daily Mean Precipitation — Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_xlim(1, 366)
    ax.set_ylim(bottom=0)
    ax.set_xticks(MONTH_DOY)
    ax.set_xticklabels(MONTH_NAMES)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig03_seasonal_cycle_pub.png")


# =========================================================================== #
#  FIGURE 4: Interannual JJAS variability — bar + smoothed + trend
# =========================================================================== #
def fig04(tp):
    print("Fig 04: Interannual JJAS variability...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    yearly = area_mean(jjas).resample(time="YE").mean()
    years = yearly["time.year"].values.astype(float)
    vals = yearly.values
    lt_mean = float(np.nanmean(vals))
    anom = vals - lt_mean
    std_val = float(np.nanstd(anom))

    fig, ax = plt.subplots(figsize=(15, 5))

    # color by intensity
    norm = mcolors.Normalize(vmin=-2*std_val, vmax=2*std_val)
    cmap_bar = plt.cm.RdBu
    colors = [cmap_bar(norm(a)) for a in anom]

    bars = ax.bar(years, anom, color=colors, width=0.75, edgecolor="white", lw=0.3,
                  zorder=3)

    # ±1σ bands
    ax.axhspan(-std_val, std_val, alpha=0.06, color="gray", zorder=1)
    ax.axhline(std_val, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.axhline(-std_val, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.text(years[-1]+1, std_val, "+1σ", fontsize=8, color="gray", va="bottom")
    ax.text(years[-1]+1, -std_val, "−1σ", fontsize=8, color="gray", va="top")

    # 11-year running mean
    k = 11
    rm = uniform_filter1d(anom, k, mode="nearest")
    ax.plot(years, rm, color="k", lw=2, ls="-", alpha=0.8, zorder=5,
            label=f"{k}-yr running mean")

    # linear trend
    mask = ~np.isnan(vals)
    slope, intercept, rval, pval, _ = linregress(years[mask], vals[mask])
    trend_line = slope * years + intercept - lt_mean
    trend_str = f"Trend: {slope*10:+.3f} mm day⁻¹ decade⁻¹ (p={pval:.3f})"
    ax.plot(years, trend_line, color="#d62728", ls="--", lw=1.5, zorder=4,
            label=trend_str)

    ax.axhline(0, color="k", lw=0.8, zorder=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("JJAS Precipitation Anomaly (mm day⁻¹)")
    ax.set_title("All-India JJAS Mean Precipitation Anomaly (ERA5)", fontsize=13)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.set_xlim(years[0]-1, years[-1]+3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig04_interannual_pub.png")


# =========================================================================== #
#  FIGURE 5: Hovmöller — monsoon onset/withdrawal
# =========================================================================== #
def fig05(tp):
    print("Fig 05: Hovmöller diagram...")
    sub = tp.sel(time=slice("2002", "2019"))
    sub = sub.sel(time=sub["time.month"].isin([4, 5, 6, 7, 8, 9, 10]))
    lon_avg = sub.sel(longitude=slice(70, 90)).mean("longitude")
    pentad = lon_avg.resample(time="5D").mean()
    pentad_doy = pentad.groupby("time.dayofyear").mean("time")

    fig, ax = plt.subplots(figsize=(12, 7))
    doys = pentad_doy.dayofyear.values
    lats = pentad_doy.latitude.values
    sm = gaussian_filter(pentad_doy.values.T, sigma=0.8)

    levels = np.arange(0, 16.1, 1.0)
    cf = ax.contourf(doys, lats, sm, levels=levels, cmap=precip_cmap(), extend="max")
    cs = ax.contour(doys, lats, sm, levels=[2, 5, 10], colors="k",
                    linewidths=[0.4, 0.6, 0.9], alpha=0.6)
    ax.clabel(cs, fmt="%.0f", fontsize=7)

    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_title("Monsoon Progression: Latitude–Time Hovmöller (70–90°E, 2002–2019)",
                 fontsize=13, pad=10)

    month_doy = [91, 121, 152, 182, 213, 244, 274, 305]
    month_lbl = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    ax.set_xticks(month_doy)
    ax.set_xticklabels(month_lbl)
    ax.set_ylim(LAT_S, LAT_N)

    # annotate key features
    ax.annotate("Monsoon\nonset", xy=(155, 10), fontsize=9, fontweight="bold",
                color="white", ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.annotate("Peak\nmonsoon", xy=(210, 22), fontsize=9, fontweight="bold",
                color="white", ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig05_hovmoller_pub.png")


# =========================================================================== #
#  FIGURE 6: Wet/Dry year composites with significance
# =========================================================================== #
def fig06(tp):
    print("Fig 06: Wet/Dry composites with significance...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    yearly_map = jjas.groupby("time.year").mean("time")
    yearly_ai = area_mean(jjas).resample(time="YE").mean()

    years = yearly_ai["time.year"].values
    vals = yearly_ai.values
    lt_mean = float(np.nanmean(vals))
    anom_ts = vals - lt_mean
    std_val = float(np.nanstd(anom_ts))

    wet_yrs = years[anom_ts > 0.5 * std_val]
    dry_yrs = years[anom_ts < -0.5 * std_val]
    clim_map = yearly_map.mean("year")

    wet_comp = yearly_map.sel(year=wet_yrs).mean("year") - clim_map
    dry_comp = yearly_map.sel(year=dry_yrs).mean("year") - clim_map

    # significance via bootstrap: compare composite mean to all-year std
    all_std = yearly_map.std("year")

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})
    cmap = anomaly_cmap()
    levels = np.arange(-4, 4.1, 0.5)

    configs = [
        (wet_comp, f"Wet Years (n={len(wet_yrs)})", wet_yrs, "(a)"),
        (dry_comp, f"Dry Years (n={len(dry_yrs)})", dry_yrs, "(b)"),
    ]

    for ax, (data, title, yrs, lbl) in zip(axes, configs):
        india_extent(ax)
        add_map_features(ax, rivers=True)
        sm = smooth_field(data.values, sigma=0.5)
        cf = ax.contourf(data.longitude, data.latitude, sm, levels=levels,
                         cmap=cmap, extend="both", transform=proj)
        # hatching where |anomaly| > 1σ of interannual variability
        sig = np.abs(data.values) > all_std.values
        if sig.any():
            ax.contourf(data.longitude, data.latitude, sig.astype(float),
                        levels=[0.5, 1.5], hatches=["///"], colors="none",
                        transform=proj, alpha=0)
        ax.set_title(title, fontsize=12, pad=8)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                          pad=0.06, aspect=30)
        cb.set_label("Anomaly (mm day⁻¹)", fontsize=10)
        panel_label(ax, lbl)

        yr_str = ", ".join(str(int(y)) for y in sorted(yrs))
        ax.text(0.02, 0.02, yr_str, transform=ax.transAxes, fontsize=7,
                color="#555555", va="bottom", fontstyle="italic",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    fig.suptitle("JJAS Precipitation Anomaly Composites (ERA5, ±0.5σ threshold)",
                 fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, "fig06_wet_dry_pub.png")


# =========================================================================== #
#  FIGURE 7: Extreme precipitation frequency + intensity
# =========================================================================== #
def fig07(tp):
    print("Fig 07: Extreme precipitation maps...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    jjas = jjas.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    heavy_frac = (jjas > 10.0).groupby("time.year").mean("time").mean("year") * 100
    p99 = jjas.groupby("time.year").quantile(0.99, dim="time").mean("year")

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})

    # Panel a: heavy rain frequency
    ax = axes[0]
    india_extent(ax)
    add_map_features(ax, rivers=True)
    sm = smooth_field(heavy_frac.values, sigma=0.5)
    levels = np.arange(0, 45, 3)
    cf = ax.contourf(heavy_frac.longitude, heavy_frac.latitude, sm, levels=levels,
                     cmap="YlOrRd", extend="max", transform=proj)
    cs = ax.contour(heavy_frac.longitude, heavy_frac.latitude, sm,
                    levels=[10, 20, 30], colors="k", linewidths=0.5, alpha=0.5,
                    transform=proj)
    ax.clabel(cs, fmt="%.0f%%", fontsize=7)
    ax.set_title("Heavy Rain Frequency (>10 mm day⁻¹)", fontsize=12, pad=8)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                      pad=0.06, aspect=30)
    cb.set_label("% of JJAS days", fontsize=10)
    panel_label(ax, "(a)")

    # Panel b: 99th percentile intensity
    ax = axes[1]
    india_extent(ax)
    add_map_features(ax, rivers=True)
    sm2 = smooth_field(p99.values, sigma=0.5)
    levels2 = np.arange(0, 55, 5)
    cf2 = ax.contourf(p99.longitude, p99.latitude, sm2, levels=levels2,
                      cmap="magma_r", extend="max", transform=proj)
    cs2 = ax.contour(p99.longitude, p99.latitude, sm2,
                     levels=[15, 25, 35, 45], colors="white", linewidths=0.5,
                     alpha=0.7, transform=proj)
    ax.clabel(cs2, fmt="%.0f", fontsize=7, colors="white")
    ax.set_title("99th Percentile Daily Rainfall", fontsize=12, pad=8)
    cb2 = fig.colorbar(cf2, ax=ax, orientation="horizontal", shrink=0.85,
                       pad=0.06, aspect=30)
    cb2.set_label("Precipitation (mm day⁻¹)", fontsize=10)
    panel_label(ax, "(b)")

    fig.suptitle(f"Extreme JJAS Precipitation ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, "fig07_extreme_pub.png")


# =========================================================================== #
#  FIGURE 8: Regional seasonal cycles — stacked
# =========================================================================== #
def fig08(tp):
    print("Fig 08: Regional seasonal cycles...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for ax, (rname, rbox) in zip(axes.flat, IMD_REGIONS.items()):
        rmean = region_mean(clim, rbox["lat"], rbox["lon"])
        doy_grp = rmean.groupby("time.dayofyear")
        mean = doy_grp.mean()
        p10 = doy_grp.quantile(0.10)
        p25 = doy_grp.quantile(0.25)
        p75 = doy_grp.quantile(0.75)
        p90 = doy_grp.quantile(0.90)

        k = 15
        for da in [mean, p10, p25, p75, p90]:
            da.values[:] = uniform_filter1d(da.values, k, mode="wrap")

        doy = mean.dayofyear.values
        c = rbox["color"]
        ax.fill_between(doy, p10.values, p90.values, alpha=0.1, color=c)
        ax.fill_between(doy, p25.values, p75.values, alpha=0.25, color=c)
        ax.plot(doy, mean.values, color=c, lw=2.5, zorder=5)

        # monsoon fraction annotation
        jjas_mean = float(mean.sel(dayofyear=slice(152, 273)).mean())
        ann_mean = float(mean.mean())
        pct = jjas_mean / ann_mean * 100 if ann_mean > 0 else 0
        ax.text(0.97, 0.95, f"JJAS: {jjas_mean:.1f} mm/day\n({pct:.0f}% of annual)",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c, alpha=0.9))

        ax.set_title(rname, fontsize=12, fontweight="bold", color=c)
        ax.set_xlim(1, 366)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(MONTH_DOY)
        ax.set_xticklabels(MONTH_NAMES)

    for ax in axes[-1]:
        ax.set_xlabel("Month")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precipitation (mm day⁻¹)")

    fig.suptitle(f"Regional Precipitation Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig08_regional_cycles_pub.png")


# =========================================================================== #
#  FIGURE 9: Intraseasonal variability — std of 10–90 day bandpass
# =========================================================================== #
def fig09(tp):
    print("Fig 09: Intraseasonal variability...")
    # Compute per-gridpoint std of JJAS daily anomalies (proxy for ISV)
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    jjas = clim.sel(time=clim["time.month"].isin([6, 7, 8, 9]))

    # daily climatology (mean of each DOY)
    doy_clim = jjas.groupby("time.dayofyear").mean("time")
    # anomalies
    anom = jjas.groupby("time.dayofyear") - doy_clim
    # std of anomalies (intraseasonal variability)
    isv = anom.std("time")

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": proj})
    india_extent(ax)
    add_map_features(ax, rivers=True)

    sm = smooth_field(isv.values, sigma=0.5)
    levels = np.arange(0, 14, 1.0)
    cf = ax.contourf(isv.longitude, isv.latitude, sm, levels=levels,
                     cmap="inferno_r", extend="max", transform=proj)
    cs = ax.contour(isv.longitude, isv.latitude, sm, levels=[4, 8, 12],
                    colors="white", linewidths=0.6, alpha=0.7, transform=proj)
    ax.clabel(cs, fmt="%.0f", fontsize=8, colors="white")

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                      pad=0.06, aspect=30)
    cb.set_label("Intraseasonal Std Dev (mm day⁻¹)", fontsize=10)
    ax.set_title(f"JJAS Intraseasonal Precipitation Variability ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=13, pad=10)
    savefig(fig, "fig09_intraseasonal_pub.png")


# =========================================================================== #
#  FIGURE 10: Precipitation intensity PDF by region
# =========================================================================== #
def fig10(tp):
    print("Fig 10: Precipitation intensity PDF...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    jjas = jjas.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 60, 1.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # All India
    ai = area_mean(jjas).values
    hist_ai, _ = np.histogram(ai[ai > 0.5], bins=bins, density=True)
    ax.semilogy(bin_centers, hist_ai, color="k", lw=2.5, label="All India", zorder=5)

    for rname, rbox in IMD_REGIONS.items():
        rv = region_mean(jjas, rbox["lat"], rbox["lon"]).values
        hist_r, _ = np.histogram(rv[rv > 0.5], bins=bins, density=True)
        ax.semilogy(bin_centers, hist_r, lw=1.8, color=rbox["color"],
                    label=rname, alpha=0.85)

    ax.set_xlabel("Daily Precipitation (mm day⁻¹)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"JJAS Daily Precipitation Distribution ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=13)
    ax.set_xlim(0, 50)
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(axis="both", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate tail
    ax.annotate("Heavy tail →\nextreme events", xy=(35, 5e-4), fontsize=9,
                fontstyle="italic", color="#666666")
    savefig(fig, "fig10_intensity_pdf_pub.png")


# =========================================================================== #
#  FIGURE 11: JJAS precipitation trend map
# =========================================================================== #
def fig11(tp):
    print("Fig 11: JJAS trend map...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    yearly = jjas.groupby("time.year").mean("time")
    years_arr = yearly.year.values.astype(float)

    # per-gridpoint linear trend
    nlat, nlon = yearly.shape[1], yearly.shape[2]
    slope_map = np.full((nlat, nlon), np.nan)
    pval_map = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            ts = yearly.values[:, i, j]
            mask = ~np.isnan(ts)
            if mask.sum() > 10:
                res = linregress(years_arr[mask], ts[mask])
                slope_map[i, j] = res.slope * 10  # per decade
                pval_map[i, j] = res.pvalue

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": proj})
    india_extent(ax)
    add_map_features(ax, rivers=True)

    sm = gaussian_filter(np.nan_to_num(slope_map, nan=0), sigma=0.5)
    levels = np.arange(-1.0, 1.05, 0.1)
    cf = ax.contourf(yearly.longitude, yearly.latitude, sm, levels=levels,
                     cmap=trend_cmap(), extend="both", transform=proj)

    # stipple significant trends (p < 0.05)
    sig = pval_map < 0.05
    if sig.any():
        lon2d, lat2d = np.meshgrid(yearly.longitude.values, yearly.latitude.values)
        ax.scatter(lon2d[sig], lat2d[sig], s=3, c="k", alpha=0.4,
                   marker=".", transform=proj, zorder=6)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                      pad=0.06, aspect=30)
    cb.set_label("Trend (mm day⁻¹ decade⁻¹)", fontsize=10)
    ax.set_title(f"JJAS Precipitation Trend ({FULL_Y0}–{FULL_Y1})",
                 fontsize=13, pad=10)
    savefig(fig, "fig11_jjas_trend_pub.png")


# =========================================================================== #
#  FIGURE 12: Coefficient of variation map
# =========================================================================== #
def fig12(tp):
    print("Fig 12: JJAS coefficient of variation...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    jjas = jjas.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    yearly = jjas.groupby("time.year").mean("time")
    cv = (yearly.std("year") / yearly.mean("year")) * 100  # percent

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": proj})
    india_extent(ax)
    add_map_features(ax, rivers=True)

    sm = smooth_field(cv.values, sigma=0.5)
    levels = np.arange(0, 55, 5)
    cf = ax.contourf(cv.longitude, cv.latitude, sm, levels=levels,
                     cmap="YlOrBr", extend="max", transform=proj)
    cs = ax.contour(cv.longitude, cv.latitude, sm, levels=[15, 25, 35],
                    colors="k", linewidths=0.5, alpha=0.5, transform=proj)
    ax.clabel(cs, fmt="%.0f%%", fontsize=8)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                      pad=0.06, aspect=30)
    cb.set_label("Coefficient of Variation (%)", fontsize=10)
    ax.set_title(f"Interannual Variability of JJAS Rainfall ({CLIM_Y0}–{CLIM_Y1})",
                 fontsize=13, pad=10)
    savefig(fig, "fig12_cv_map_pub.png")


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="ERA5 TP Publication Figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()
    figs = set(args.fig) if args.fig else set(range(1, 13))

    print(f"═══ ERA5 TP Publication Figures ═══")
    print(f"  WB2 zarr: {WB2_ZARR}")
    print(f"  India box: {LAT_S}–{LAT_N}°N, {LON_W}–{LON_E}°E")
    print(f"  Output dir: {OUT_DIR}\n")

    tp = load_wb2_tp_india()
    print()

    fig_funcs = {
        1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06,
        7: fig07, 8: fig08, 9: fig09, 10: fig10, 11: fig11, 12: fig12,
    }

    for n in sorted(figs):
        if n in fig_funcs:
            try:
                fig_funcs[n](tp)
            except Exception as e:
                print(f"  ✗ Fig {n} FAILED: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"  ⚠ Unknown figure number: {n}")

    print(f"\n═══ Done! {len(figs)} figures in: {OUT_DIR} ═══")


if __name__ == "__main__":
    main()
