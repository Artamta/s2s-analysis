#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_monsoon_dynamics.py — Advanced Monsoon Dynamics Analysis (ERA5)
========================================================================
Generates 12 publication-quality figures exploring Indian monsoon dynamics
from WeatherBench2 ERA5 reanalysis.

Figures:
  01. Active & Break spell identification (example year 2019)
  02. Active-Break spell climatology (histograms)
  03. Active vs Break composite precipitation anomaly maps
  04. Intraseasonal variance fraction (20-60 day band)
  05. Northward propagation lag-latitude diagram (MISO)
  06. MISO power spectrum (Welch method)
  07. Variance decomposition by timescale (stacked bars)
  08. Wet spell duration distribution (4 IMD regions)
  09. Dry spell duration distribution (4 IMD regions)
  10. Monsoon onset date interannual variability
  11. Onset date vs total JJAS rainfall scatter
  12. Autocorrelation function (4 IMD regions)

Usage:
    python explore_monsoon_dynamics.py
"""
import os
import sys
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
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import linregress
from scipy.optimize import curve_fit

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# =========================================================================== #
#  Config
# =========================================================================== #
WB2_ZARR_15 = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
               "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

ANALYSIS_Y0, ANALYSIS_Y1 = 1979, 2022

IMD_REGIONS = {
    "Northwest India":  {"lat": (25.0, 36.0), "lon": (68.0, 80.0), "color": "#e41a1c"},
    "Central India":    {"lat": (20.0, 28.0), "lon": (74.0, 86.0), "color": "#377eb8"},
    "South Peninsula":  {"lat": ( 8.0, 20.0), "lon": (74.0, 82.0), "color": "#4daf4a"},
    "East & NE India":  {"lat": (20.0, 30.0), "lon": (86.0, 98.0), "color": "#ff7f00"},
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monsoon_dynamics")
os.makedirs(OUT_DIR, exist_ok=True)


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
    """Rich scientific precipitation colormap."""
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
    """Brown–white–teal diverging colormap (BrBG-like)."""
    colors = [
        "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5",
        "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_anom", colors, N=256)


def miso_cmap():
    """Blue-white-red diverging colormap for correlation/MISO plots."""
    colors = [
        "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#f7f7f7",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
    ]
    return mcolors.LinearSegmentedColormap.from_list("miso", colors, N=256)


# =========================================================================== #
#  Data loading
# =========================================================================== #
def load_wb2_tp_india(y0=None, y1=None):
    """Load WB2 ERA5 1.5° daily TP over India, eagerly into memory."""
    print("  [1/3] Opening zarr...")
    ds = xr.open_zarr(WB2_ZARR_15)
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
    sz = os.path.getsize(path) / 1024
    print(f"  ✓ saved {path} ({sz:.0f} KB)")


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


def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    """Apply Butterworth bandpass filter using second-order sections."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)


def get_jjas_daily(tp, year=None):
    """Extract JJAS daily data, optionally for a specific year."""
    if year is not None:
        sub = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
    else:
        sub = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    return sub


def compute_daily_climatology(tp):
    """Compute day-of-year climatology from JJAS data."""
    jjas = get_jjas_daily(tp)
    return jjas.groupby("time.dayofyear").mean("time")


def compute_daily_anomalies(tp):
    """Compute daily anomalies by removing DOY climatology."""
    jjas = get_jjas_daily(tp)
    doy_clim = jjas.groupby("time.dayofyear").mean("time")
    return jjas.groupby("time.dayofyear") - doy_clim


def find_consecutive_spells(mask, min_length=3):
    """
    Find consecutive True runs in a boolean array.
    Returns list of (start_idx, end_idx, length) tuples for runs >= min_length.
    """
    spells = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            length = i - start
            if length >= min_length:
                spells.append((start, i - 1, length))
        else:
            i += 1
    return spells


# =========================================================================== #
#  FIGURE 1: Active & Break Spell Identification (Example Year 2019)
# =========================================================================== #
def fig01(tp):
    print("Fig 01: Active & Break spell identification (2019)...")
    year = 2019

    # JJAS data for all years to compute climatology
    jjas_all = get_jjas_daily(tp)
    ai_all = area_mean(jjas_all)

    # DOY climatology and std
    doy_clim = ai_all.groupby("time.dayofyear").mean("time")
    doy_std = ai_all.groupby("time.dayofyear").std("time")

    # Year 2019 data
    jjas_yr = get_jjas_daily(tp, year=year)
    ai_yr = area_mean(jjas_yr)

    # Standardized anomaly
    days_yr = ai_yr["time.dayofyear"]
    anom = ai_yr.values - doy_clim.sel(dayofyear=days_yr).values
    std_vals = doy_std.sel(dayofyear=days_yr).values
    std_anom = anom / std_vals

    # Active: > +1σ for ≥3 consecutive days
    active_mask = std_anom > 1.0
    break_mask = std_anom < -1.0
    active_spells = find_consecutive_spells(active_mask, min_length=3)
    break_spells = find_consecutive_spells(break_mask, min_length=3)

    active_days = sum(s[2] for s in active_spells)
    break_days = sum(s[2] for s in break_spells)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 5.5))
    times = np.arange(len(std_anom))
    dates_str = [str(t.values)[:10] for t in ai_yr.time]

    # Time series
    ax.plot(times, std_anom, color="#333333", lw=1.2, zorder=5, label="Standardized anomaly")
    ax.fill_between(times, 0, std_anom, where=std_anom > 0, alpha=0.15,
                    color="#2171b5", interpolate=True)
    ax.fill_between(times, 0, std_anom, where=std_anom < 0, alpha=0.15,
                    color="#d62728", interpolate=True)

    # Shade active spells
    for start, end, length in active_spells:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.25, color="#2171b5", zorder=1)

    # Shade break spells
    for start, end, length in break_spells:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.25, color="#d62728", zorder=1)

    # Reference lines
    ax.axhline(1.0, color="#2171b5", ls="--", lw=0.8, alpha=0.7, label="+1σ threshold")
    ax.axhline(-1.0, color="#d62728", ls="--", lw=0.8, alpha=0.7, label="−1σ threshold")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)

    # Annotations
    ax.text(0.02, 0.97, f"Active spells: {len(active_spells)} ({active_days} days)",
            transform=ax.transAxes, fontsize=10, va="top", color="#2171b5",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.text(0.02, 0.89, f"Break spells: {len(break_spells)} ({break_days} days)",
            transform=ax.transAxes, fontsize=10, va="top", color="#d62728",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # X-axis: show dates at ~weekly intervals
    tick_idx = np.arange(0, len(times), 7)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates_str[i][5:] for i in tick_idx], rotation=45, ha="right")
    ax.set_xlim(-1, len(times))
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized Precipitation Anomaly (σ)")
    ax.set_title(f"Active & Break Spells — JJAS {year} (All-India Daily Rainfall Anomaly)",
                 fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig01_active_break_2019.png")


# =========================================================================== #
#  FIGURE 2: Active-Break Spell Climatology (Histograms)
# =========================================================================== #
def fig02(tp):
    print("Fig 02: Active-Break spell climatology...")
    jjas_all = get_jjas_daily(tp)
    ai_all = area_mean(jjas_all)

    doy_clim = ai_all.groupby("time.dayofyear").mean("time")
    doy_std = ai_all.groupby("time.dayofyear").std("time")

    all_active_durations = []
    all_break_durations = []
    active_count_per_year = []
    break_count_per_year = []

    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            jjas_yr = get_jjas_daily(tp, year=year)
            ai_yr = area_mean(jjas_yr)
            days_yr = ai_yr["time.dayofyear"]
            anom = ai_yr.values - doy_clim.sel(dayofyear=days_yr).values
            std_vals = doy_std.sel(dayofyear=days_yr).values
            std_anom = anom / std_vals

            active_spells = find_consecutive_spells(std_anom > 1.0, min_length=3)
            break_spells = find_consecutive_spells(std_anom < -1.0, min_length=3)

            all_active_durations.extend([s[2] for s in active_spells])
            all_break_durations.extend([s[2] for s in break_spells])
            active_count_per_year.append(len(active_spells))
            break_count_per_year.append(len(break_spells))
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): Active spells
    ax = axes[0]
    max_dur = max(max(all_active_durations, default=3), max(all_break_durations, default=3))
    bins = np.arange(3, max_dur + 2) - 0.5
    ax.hist(all_active_durations, bins=bins, color="#2171b5", edgecolor="white",
            alpha=0.85, lw=0.5, zorder=3)
    mean_ad = np.mean(all_active_durations) if all_active_durations else 0
    max_ad = max(all_active_durations) if all_active_durations else 0
    ax.axvline(mean_ad, color="#08519c", ls="--", lw=2, alpha=0.8,
               label=f"Mean = {mean_ad:.1f} days")
    ax.set_xlabel("Spell Duration (days)")
    ax.set_ylabel("Frequency (count)")
    ax.set_title("(a)  Active Spell Durations", fontsize=12, fontweight="bold")
    mean_freq_a = np.mean(active_count_per_year) if active_count_per_year else 0
    stats_text_a = (f"Mean freq: {mean_freq_a:.1f} spells/yr\n"
                    f"Mean dur: {mean_ad:.1f} days\n"
                    f"Max dur: {max_ad} days\n"
                    f"Total: {len(all_active_durations)} spells")
    ax.text(0.97, 0.95, stats_text_a, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#d0e1f2",
                      edgecolor="#2171b5", alpha=0.9))
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "(a)")

    # Panel (b): Break spells
    ax = axes[1]
    ax.hist(all_break_durations, bins=bins, color="#d62728", edgecolor="white",
            alpha=0.85, lw=0.5, zorder=3)
    mean_bd = np.mean(all_break_durations) if all_break_durations else 0
    max_bd = max(all_break_durations) if all_break_durations else 0
    ax.axvline(mean_bd, color="#a50f15", ls="--", lw=2, alpha=0.8,
               label=f"Mean = {mean_bd:.1f} days")
    ax.set_xlabel("Spell Duration (days)")
    ax.set_ylabel("Frequency (count)")
    ax.set_title("(b)  Break Spell Durations", fontsize=12, fontweight="bold")
    mean_freq_b = np.mean(break_count_per_year) if break_count_per_year else 0
    stats_text_b = (f"Mean freq: {mean_freq_b:.1f} spells/yr\n"
                    f"Mean dur: {mean_bd:.1f} days\n"
                    f"Max dur: {max_bd} days\n"
                    f"Total: {len(all_break_durations)} spells")
    ax.text(0.97, 0.95, stats_text_b, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fddbc7",
                      edgecolor="#d62728", alpha=0.9))
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "(b)")

    fig.suptitle(f"Active & Break Spell Climatology (JJAS {ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig02_active_break_climatology.png")


# =========================================================================== #
#  FIGURE 3: Active vs Break Composite Maps
# =========================================================================== #
def fig03(tp):
    print("Fig 03: Active vs Break composite maps...")
    jjas_all = get_jjas_daily(tp)
    ai_all = area_mean(jjas_all)

    doy_clim_ai = ai_all.groupby("time.dayofyear").mean("time")
    doy_std_ai = ai_all.groupby("time.dayofyear").std("time")

    # Spatial DOY climatology
    doy_clim_map = jjas_all.groupby("time.dayofyear").mean("time")

    # Collect active/break day indices
    active_times = []
    break_times = []

    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            jjas_yr = get_jjas_daily(tp, year=year)
            ai_yr = area_mean(jjas_yr)
            days_yr = ai_yr["time.dayofyear"]
            anom = ai_yr.values - doy_clim_ai.sel(dayofyear=days_yr).values
            std_vals = doy_std_ai.sel(dayofyear=days_yr).values
            std_anom = anom / std_vals

            active_spells = find_consecutive_spells(std_anom > 1.0, min_length=3)
            break_spells = find_consecutive_spells(std_anom < -1.0, min_length=3)

            for start, end, _ in active_spells:
                active_times.extend(jjas_yr.time.values[start:end + 1])
            for start, end, _ in break_spells:
                break_times.extend(jjas_yr.time.values[start:end + 1])
        except Exception:
            continue

    print(f"    Active days: {len(active_times)}, Break days: {len(break_times)}")

    # Compute anomaly composites
    active_data = jjas_all.sel(time=active_times)
    break_data = jjas_all.sel(time=break_times)

    # Anomalies: subtract DOY climatology
    active_anom = active_data.groupby("time.dayofyear") - doy_clim_map
    break_anom = break_data.groupby("time.dayofyear") - doy_clim_map

    active_comp = active_anom.mean("time")
    break_comp = break_anom.mean("time")

    # Significance: t-test like check (|mean| > std/sqrt(N) * 2)
    active_std = active_anom.std("time")
    break_std = break_anom.std("time")
    active_sig = np.abs(active_comp.values) > 2 * active_std.values / np.sqrt(len(active_times))
    break_sig = np.abs(break_comp.values) > 2 * break_std.values / np.sqrt(len(break_times))

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})
    levels = np.arange(-6, 6.1, 0.5)
    cmap = anomaly_cmap()

    configs = [
        (active_comp, active_sig, f"Active Spell Composite (n={len(active_times)} days)", "(a)"),
        (break_comp, break_sig, f"Break Spell Composite (n={len(break_times)} days)", "(b)"),
    ]

    for ax, (data, sig, title, lbl) in zip(axes, configs):
        india_extent(ax)
        add_map_features(ax, rivers=True)
        sm = smooth_field(data.values, sigma=0.5)
        cf = ax.contourf(data.longitude, data.latitude, sm, levels=levels,
                         cmap=cmap, extend="both", transform=proj)
        # Significance stippling
        if sig.any():
            lon2d, lat2d = np.meshgrid(data.longitude.values, data.latitude.values)
            ax.scatter(lon2d[sig], lat2d[sig], s=2, c="k", alpha=0.35,
                       marker=".", transform=proj, zorder=6)
        ax.set_title(title, fontsize=11, pad=8)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                          pad=0.06, aspect=30)
        cb.set_label("TP Anomaly (mm day⁻¹)", fontsize=10)
        panel_label(ax, lbl)

    fig.suptitle(f"JJAS Precipitation Anomaly: Active vs Break Composites ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=13, fontweight="bold", y=1.02)
    savefig(fig, "fig03_active_break_composites.png")


# =========================================================================== #
#  FIGURE 4: Intraseasonal Variance Fraction (20-60 day)
# =========================================================================== #
def fig04(tp):
    print("Fig 04: Intraseasonal variance fraction (20-60 day)...")
    nlat = tp.sizes["latitude"]
    nlon = tp.sizes["longitude"]

    var_frac = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            # Concatenate JJAS data across years for this gridpoint
            vals_list = []
            for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
                try:
                    jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                    ts = jjas_yr.values[:, i, j]
                    if len(ts) >= 90:
                        vals_list.append(ts)
                except Exception:
                    continue

            if not vals_list:
                continue

            # For each year, compute filtered variance / total variance
            fracs = []
            for ts in vals_list:
                if len(ts) < 60:  # need enough data for 60-day filter
                    continue
                # Remove mean
                ts_dm = ts - np.nanmean(ts)
                total_var = np.nanvar(ts_dm)
                if total_var < 1e-10:
                    continue
                try:
                    filtered = bandpass_filter(ts_dm, 1.0/60.0, 1.0/20.0, fs=1.0, order=4)
                    filt_var = np.nanvar(filtered)
                    fracs.append(filt_var / total_var)
                except Exception:
                    continue

            if fracs:
                var_frac[i, j] = np.mean(fracs)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": proj})
    india_extent(ax)
    add_map_features(ax, rivers=True)

    sm = smooth_field(var_frac, sigma=0.5)
    levels = np.arange(0, 0.55, 0.05)
    cf = ax.contourf(tp.longitude, tp.latitude, sm, levels=levels,
                     cmap="YlOrRd", extend="max", transform=proj)
    cs = ax.contour(tp.longitude, tp.latitude, sm, levels=[0.1, 0.2, 0.3, 0.4],
                    colors="k", linewidths=0.5, alpha=0.5, transform=proj)
    ax.clabel(cs, fmt="%.1f", fontsize=8)

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                      pad=0.06, aspect=30)
    cb.set_label("Fraction of Total JJAS Variance in 20-60 Day Band", fontsize=10)
    ax.set_title(f"Intraseasonal (MISO) Variance Fraction ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=13, pad=10)
    savefig(fig, "fig04_intraseasonal_variance_fraction.png")


# =========================================================================== #
#  FIGURE 5: Northward Propagation (Lag-Latitude Diagram)
# =========================================================================== #
def fig05(tp):
    print("Fig 05: Northward propagation lag-latitude diagram...")
    # Reference region: Central India (15-25°N, 75-85°E)
    ref_lat = (15, 25)
    ref_lon = (75, 85)
    # Average over 70-90°E for latitude profiles
    lon_avg_range = (70, 90)

    max_lag = 20
    lats = tp.sel(latitude=slice(LAT_S, 35.0)).latitude.values

    # Build lag-correlation matrix
    lag_corr = np.full((2 * max_lag + 1, len(lats)), np.nan)

    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
            if jjas_yr.sizes["time"] < 90:
                continue

            # Reference time series (Central India mean)
            ref = region_mean(jjas_yr, ref_lat, ref_lon).values
            ref_dm = ref - np.mean(ref)
            if len(ref_dm) < 60:
                continue
            ref_filt = bandpass_filter(ref_dm, 1.0/60.0, 1.0/20.0, fs=1.0, order=4)

            # Latitude profiles averaged over 70-90°E
            lon_avg = jjas_yr.sel(longitude=slice(lon_avg_range[0], lon_avg_range[1])).mean("longitude")

            for ilat, lat in enumerate(lats):
                ts = lon_avg.sel(latitude=lat, method="nearest").values
                ts_dm = ts - np.mean(ts)
                if len(ts_dm) != len(ref_filt):
                    continue
                try:
                    ts_filt = bandpass_filter(ts_dm, 1.0/60.0, 1.0/20.0, fs=1.0, order=4)
                except Exception:
                    continue

                for ilag, lag in enumerate(range(-max_lag, max_lag + 1)):
                    if lag >= 0:
                        r = ref_filt[:len(ref_filt) - lag]
                        t = ts_filt[lag:]
                    else:
                        r = ref_filt[-lag:]
                        t = ts_filt[:len(ts_filt) + lag]
                    if len(r) > 10:
                        corr = np.corrcoef(r, t)[0, 1]
                        if np.isnan(lag_corr[ilag, ilat]):
                            lag_corr[ilag, ilat] = corr
                        else:
                            lag_corr[ilag, ilat] += corr
        except Exception as e:
            continue

    # Average over years
    n_years = ANALYSIS_Y1 - ANALYSIS_Y0 + 1
    lag_corr /= n_years

    fig, ax = plt.subplots(figsize=(10, 7))
    lags = np.arange(-max_lag, max_lag + 1)
    sm = gaussian_filter(np.nan_to_num(lag_corr, nan=0), sigma=0.8)
    levels = np.arange(-0.6, 0.65, 0.05)

    cf = ax.contourf(lags, lats, sm.T, levels=levels, cmap=miso_cmap(), extend="both")
    cs = ax.contour(lags, lats, sm.T, levels=[-0.3, -0.2, 0.2, 0.3],
                    colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, fmt="%.1f", fontsize=7)

    # Reference latitude band
    ax.axhspan(ref_lat[0], ref_lat[1], alpha=0.1, color="red", zorder=0)
    ax.axvline(0, color="k", ls="-", lw=0.8, alpha=0.5)

    # Annotate ~1°/day propagation line
    prop_lats = np.array([10, 30])
    prop_lags = (prop_lats - 20) * 1.0  # ~1°/day from 20°N reference
    ax.plot(prop_lags, prop_lats, "k--", lw=1.5, alpha=0.5, label="~1°/day propagation")

    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Lag Correlation", fontsize=10)

    ax.set_xlabel("Lag (days, positive = reference leads)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(f"MISO Northward Propagation — Lag-Latitude Diagram\n"
                 f"(Reference: {ref_lat[0]}–{ref_lat[1]}°N, {ref_lon[0]}–{ref_lon[1]}°E; "
                 f"Averaged: {lon_avg_range[0]}–{lon_avg_range[1]}°E; "
                 f"{ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=11, pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig05_northward_propagation.png")


# =========================================================================== #
#  FIGURE 6: MISO Power Spectrum (Welch)
# =========================================================================== #
def fig06(tp):
    print("Fig 06: MISO power spectrum...")
    # All-India JJAS daily time series for each year
    psds = []
    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            jjas_yr = get_jjas_daily(tp, year=year)
            ai = area_mean(jjas_yr).values
            if len(ai) < 90:
                continue
            ai_dm = ai - np.mean(ai)
            freqs, psd = welch(ai_dm, fs=1.0, nperseg=min(len(ai_dm), 90),
                               noverlap=min(len(ai_dm)//2, 45))
            psds.append(psd)
        except Exception:
            continue

    # Average PSD across years
    min_len = min(len(p) for p in psds)
    psds_arr = np.array([p[:min_len] for p in psds])
    mean_psd = np.mean(psds_arr, axis=0)
    freqs = freqs[:min_len]
    periods = 1.0 / freqs[1:]  # skip frequency 0

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.semilogy(periods, mean_psd[1:], color="#2171b5", lw=2, zorder=5)

    # Shade the 20-60 day MISO band
    ax.axvspan(20, 60, alpha=0.15, color="#ff7f0e", zorder=0, label="20–60 day MISO band")
    # Also mark synoptic band
    ax.axvspan(2, 10, alpha=0.1, color="#2ca02c", zorder=0, label="2–10 day synoptic band")

    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("Power Spectral Density (mm² day⁻² / Hz)", fontsize=11)
    ax.set_title(f"Power Spectrum of All-India Daily JJAS Precipitation ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=13, pad=10)
    ax.set_xlim(2, 90)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([2, 5, 10, 20, 30, 45, 60, 90])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(axis="both", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate peaks
    miso_mask = (periods >= 20) & (periods <= 60)
    if miso_mask.any():
        peak_idx = np.argmax(mean_psd[1:][miso_mask])
        peak_period = periods[miso_mask][peak_idx]
        peak_power = mean_psd[1:][miso_mask][peak_idx]
        ax.annotate(f"MISO peak\n~{peak_period:.0f} days",
                    xy=(peak_period, peak_power),
                    xytext=(peak_period * 1.3, peak_power * 3),
                    fontsize=9, fontweight="bold", color="#ff7f0e",
                    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    savefig(fig, "fig06_miso_power_spectrum.png")


# =========================================================================== #
#  FIGURE 7: Variance Decomposition by Timescale (Stacked Bars)
# =========================================================================== #
def fig07(tp):
    print("Fig 07: Variance decomposition by timescale...")
    regions = {"All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}}
    regions.update({k: {"lat": v["lat"], "lon": v["lon"]} for k, v in IMD_REGIONS.items()})

    results = {}

    for rname, rbox in regions.items():
        synoptic_vars = []
        intraseasonal_vars = []
        total_vars = []

        for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
            try:
                jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                rm = region_mean(jjas_yr, rbox["lat"], rbox["lon"]).values

                if len(rm) < 90:
                    continue

                rm_dm = rm - np.mean(rm)
                total_var = np.var(rm_dm)

                if total_var < 1e-10:
                    continue

                # Synoptic: 2-10 day
                try:
                    syn = bandpass_filter(rm_dm, 1.0/10.0, 1.0/2.0, fs=1.0, order=4)
                    synoptic_vars.append(np.var(syn))
                except Exception:
                    synoptic_vars.append(0)

                # Intraseasonal: 10-90 day
                try:
                    iso = bandpass_filter(rm_dm, 1.0/90.0, 1.0/10.0, fs=1.0, order=4)
                    intraseasonal_vars.append(np.var(iso))
                except Exception:
                    intraseasonal_vars.append(0)

                total_vars.append(total_var)
            except Exception:
                continue

        if total_vars:
            mean_total = np.mean(total_vars)
            mean_syn = np.mean(synoptic_vars)
            mean_iso = np.mean(intraseasonal_vars)

            # Interannual: variance of seasonal means
            seasonal_means = []
            for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
                try:
                    jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                    sm = float(region_mean(jjas_yr, rbox["lat"], rbox["lon"]).mean().values)
                    seasonal_means.append(sm)
                except Exception:
                    continue
            interannual_var = np.var(seasonal_means) if seasonal_means else 0

            results[rname] = {
                "synoptic": mean_syn / mean_total,
                "intraseasonal": mean_iso / mean_total,
                "interannual": interannual_var / mean_total,
            }

    fig, ax = plt.subplots(figsize=(12, 6))

    region_names = list(results.keys())
    x = np.arange(len(region_names))
    width = 0.55

    syn_vals = [results[r]["synoptic"] for r in region_names]
    iso_vals = [results[r]["intraseasonal"] for r in region_names]
    ia_vals = [results[r]["interannual"] for r in region_names]

    # Clip and normalize so they don't exceed 1
    for i in range(len(region_names)):
        total = syn_vals[i] + iso_vals[i] + ia_vals[i]
        residual = max(0, 1.0 - total)
        # Don't normalize, just show raw fractions

    bars1 = ax.bar(x, syn_vals, width, label="Synoptic (2–10 days)",
                   color="#4daf4a", edgecolor="white", lw=0.5)
    bars2 = ax.bar(x, iso_vals, width, bottom=syn_vals,
                   label="Intraseasonal (10–90 days)",
                   color="#377eb8", edgecolor="white", lw=0.5)
    bars3 = ax.bar(x, ia_vals, width,
                   bottom=[s + i for s, i in zip(syn_vals, iso_vals)],
                   label="Interannual (seasonal mean)",
                   color="#e41a1c", edgecolor="white", lw=0.5)

    # Add percentage labels
    for i in range(len(region_names)):
        # Synoptic
        if syn_vals[i] > 0.05:
            ax.text(x[i], syn_vals[i] / 2, f"{syn_vals[i]*100:.0f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        # Intraseasonal
        if iso_vals[i] > 0.05:
            ax.text(x[i], syn_vals[i] + iso_vals[i] / 2,
                    f"{iso_vals[i]*100:.0f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        # Interannual
        if ia_vals[i] > 0.03:
            ax.text(x[i], syn_vals[i] + iso_vals[i] + ia_vals[i] / 2,
                    f"{ia_vals[i]*100:.0f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of Total JJAS Daily Variance")
    ax.set_title(f"JJAS Precipitation Variance Decomposition ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig07_variance_decomposition.png")


# =========================================================================== #
#  FIGURE 8: Wet Spell Duration Distribution
# =========================================================================== #
def fig08(tp):
    print("Fig 08: Wet spell duration distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (rname, rbox), lbl in zip(axes.flat, IMD_REGIONS.items(), panel_labels):
        all_durations = []
        for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
            try:
                jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                rm = region_mean(jjas_yr, rbox["lat"], rbox["lon"]).values

                # Wet day: > 1 mm/day
                wet_mask = rm > 1.0
                spells = find_consecutive_spells(wet_mask, min_length=1)
                all_durations.extend([s[2] for s in spells])
            except Exception:
                continue

        if not all_durations:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        max_dur = min(max(all_durations), 40)
        bins = np.arange(1, max_dur + 2) - 0.5
        counts, edges = np.histogram(all_durations, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        ax.bar(bin_centers, counts, width=0.8, color=rbox["color"], alpha=0.75,
               edgecolor="white", lw=0.5, zorder=3)

        # Fit exponential: f(x) = a * exp(-b * x)
        try:
            def exp_func(x, a, b):
                return a * np.exp(-b * x)
            valid = counts > 0
            popt, _ = curve_fit(exp_func, bin_centers[valid], counts[valid],
                                p0=[counts[0], 0.3], maxfev=5000)
            x_fit = np.linspace(1, max_dur, 100)
            ax.plot(x_fit, exp_func(x_fit, *popt), color="k", ls="--", lw=1.5,
                    alpha=0.7, label=f"Exp fit (λ={popt[1]:.2f})")
        except Exception:
            pass

        mean_dur = np.mean(all_durations)
        ax.axvline(mean_dur, color="k", ls=":", lw=1.5, alpha=0.7)
        ax.text(mean_dur + 0.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                f"Mean={mean_dur:.1f}d", fontsize=8, color="k")

        ax.set_title(rname, fontsize=12, fontweight="bold", color=rbox["color"])
        ax.set_xlabel("Spell Duration (days)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0.5, max_dur + 0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl)

    fig.suptitle(f"Wet Spell Duration Distribution (>{1} mm/day, JJAS {ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig08_wet_spell_distribution.png")


# =========================================================================== #
#  FIGURE 9: Dry Spell Duration Distribution
# =========================================================================== #
def fig09(tp):
    print("Fig 09: Dry spell duration distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (rname, rbox), lbl in zip(axes.flat, IMD_REGIONS.items(), panel_labels):
        all_durations = []
        for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
            try:
                jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                rm = region_mean(jjas_yr, rbox["lat"], rbox["lon"]).values

                # Dry day: < 1 mm/day
                dry_mask = rm < 1.0
                spells = find_consecutive_spells(dry_mask, min_length=1)
                all_durations.extend([s[2] for s in spells])
            except Exception:
                continue

        if not all_durations:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        max_dur = min(max(all_durations), 40)
        bins = np.arange(1, max_dur + 2) - 0.5
        counts, edges = np.histogram(all_durations, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        ax.bar(bin_centers, counts, width=0.8, color=rbox["color"], alpha=0.75,
               edgecolor="white", lw=0.5, zorder=3)

        # Fit exponential
        try:
            def exp_func(x, a, b):
                return a * np.exp(-b * x)
            valid = counts > 0
            popt, _ = curve_fit(exp_func, bin_centers[valid], counts[valid],
                                p0=[counts[0], 0.3], maxfev=5000)
            x_fit = np.linspace(1, max_dur, 100)
            ax.plot(x_fit, exp_func(x_fit, *popt), color="k", ls="--", lw=1.5,
                    alpha=0.7, label=f"Exp fit (λ={popt[1]:.2f})")
        except Exception:
            pass

        mean_dur = np.mean(all_durations)
        ax.axvline(mean_dur, color="k", ls=":", lw=1.5, alpha=0.7)
        ax.text(mean_dur + 0.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                f"Mean={mean_dur:.1f}d", fontsize=8, color="k")

        ax.set_title(rname, fontsize=12, fontweight="bold", color=rbox["color"])
        ax.set_xlabel("Spell Duration (days)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0.5, max_dur + 0.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl)

    fig.suptitle(f"Dry Spell Duration Distribution (<{1} mm/day, JJAS {ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig09_dry_spell_distribution.png")


# =========================================================================== #
#  FIGURE 10: Monsoon Onset Date Interannual Variability
# =========================================================================== #
def fig10(tp):
    print("Fig 10: Monsoon onset date variability...")
    onset_dates = {}

    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            # Use May-September data to capture onset
            yr_data = tp.sel(time=slice(f"{year}-05-01", f"{year}-09-30"))
            ai = area_mean(yr_data).values
            times = yr_data.time.values

            if len(ai) < 100:
                continue

            # 5-day running mean
            ai_smooth = uniform_filter1d(ai, 5, mode="nearest")
            # Annual mean TP threshold
            annual_data = tp.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            annual_mean = float(area_mean(annual_data).mean().values)

            # Find onset: first day when 5-day running mean exceeds annual mean,
            # sustained for at least 5 consecutive days
            above = ai_smooth > annual_mean
            onset_found = False
            for i in range(len(above) - 5):
                if all(above[i:i + 5]):
                    onset_time = times[i]
                    onset_dt = np.datetime64(onset_time, "D")
                    jan1 = np.datetime64(f"{year}-01-01", "D")
                    doy = int((onset_dt - jan1) / np.timedelta64(1, "D")) + 1
                    onset_dates[year] = doy
                    onset_found = True
                    break

            if not onset_found:
                # Fallback: use June 1 if no onset detected
                onset_dates[year] = 152  # June 1
        except Exception:
            continue

    years = sorted(onset_dates.keys())
    doys = [onset_dates[y] for y in years]
    mean_doy = np.mean(doys)
    std_doy = np.std(doys)

    fig, ax = plt.subplots(figsize=(15, 5.5))

    # Color by early/late
    colors = []
    for d in doys:
        if d < mean_doy - std_doy:
            colors.append("#2171b5")  # Early (blue)
        elif d > mean_doy + std_doy:
            colors.append("#d62728")  # Late (red)
        else:
            colors.append("#666666")  # Normal (gray)

    ax.bar(years, doys, color=colors, edgecolor="white", lw=0.5, width=0.75, zorder=3)
    ax.axhline(mean_doy, color="k", ls="--", lw=1.5, alpha=0.7,
               label=f"Mean onset: DOY {mean_doy:.0f}")
    ax.axhspan(mean_doy - std_doy, mean_doy + std_doy, alpha=0.08, color="gray",
               zorder=0, label=f"±1σ ({std_doy:.1f} days)")

    # Convert DOY axis to month-day labels
    # Create secondary y-axis labels showing month-day
    ref_doys = [121, 135, 152, 166, 182]  # May 1, May 15, Jun 1, Jun 15, Jul 1
    ref_labels = ["May 1", "May 15", "Jun 1", "Jun 15", "Jul 1"]
    ax.set_yticks(ref_doys)
    ax.set_yticklabels(ref_labels)

    # Annotate early/late years
    early_years = [y for y, d in zip(years, doys) if d < mean_doy - std_doy]
    late_years = [y for y, d in zip(years, doys) if d > mean_doy + std_doy]
    if early_years:
        ax.text(0.02, 0.02, f"Early: {', '.join(str(y) for y in early_years)}",
                transform=ax.transAxes, fontsize=8, color="#2171b5", va="bottom")
    if late_years:
        ax.text(0.98, 0.02, f"Late: {', '.join(str(y) for y in late_years)}",
                transform=ax.transAxes, fontsize=8, color="#d62728", va="bottom", ha="right")

    ax.set_xlabel("Year")
    ax.set_ylabel("Onset Date")
    ax.set_title(f"Indian Monsoon Onset Date ({ANALYSIS_Y0}–{ANALYSIS_Y1})\n"
                 f"(5-day running mean > annual mean TP, sustained ≥5 days)",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig10_onset_variability.png")

    return onset_dates


# =========================================================================== #
#  FIGURE 11: Onset Date vs Total JJAS Rainfall
# =========================================================================== #
def fig11(tp, onset_dates=None):
    print("Fig 11: Onset vs total JJAS rainfall...")
    if onset_dates is None:
        # Recompute onset dates
        onset_dates = {}
        for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
            try:
                yr_data = tp.sel(time=slice(f"{year}-05-01", f"{year}-09-30"))
                ai = area_mean(yr_data).values
                times = yr_data.time.values
                if len(ai) < 100:
                    continue
                ai_smooth = uniform_filter1d(ai, 5, mode="nearest")
                annual_data = tp.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                annual_mean = float(area_mean(annual_data).mean().values)
                above = ai_smooth > annual_mean
                for i in range(len(above) - 5):
                    if all(above[i:i + 5]):
                        onset_time = times[i]
                        onset_dt = np.datetime64(onset_time, "D")
                        jan1 = np.datetime64(f"{year}-01-01", "D")
                        doy = int((onset_dt - jan1) / np.timedelta64(1, "D")) + 1
                        onset_dates[year] = doy
                        break
            except Exception:
                continue

    # Total JJAS rainfall per year
    total_rain = {}
    for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
        try:
            jjas_yr = get_jjas_daily(tp, year=year)
            ai = area_mean(jjas_yr).values
            total_rain[year] = np.sum(ai)  # total mm
        except Exception:
            continue

    # Common years
    common_years = sorted(set(onset_dates.keys()) & set(total_rain.keys()))
    onsets = [onset_dates[y] for y in common_years]
    totals = [total_rain[y] for y in common_years]

    # Correlation
    r, p = np.corrcoef(onsets, totals)[0, 1], linregress(onsets, totals).pvalue
    slope, intercept = linregress(onsets, totals).slope, linregress(onsets, totals).intercept

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(onsets, totals, c=common_years, cmap="viridis",
                         s=60, edgecolor="white", lw=0.8, zorder=5)

    # Regression line
    x_fit = np.linspace(min(onsets), max(onsets), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="#d62728", ls="--", lw=1.5,
            alpha=0.7, label=f"r = {r:.3f}, p = {p:.3f}")

    # Colorbar for year
    cb = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Year", fontsize=10)

    # Convert x-axis to date labels
    ref_doys = [121, 135, 152, 166, 182]
    ref_labels = ["May 1", "May 15", "Jun 1", "Jun 15", "Jul 1"]
    ax.set_xticks(ref_doys)
    ax.set_xticklabels(ref_labels)

    ax.set_xlabel("Monsoon Onset Date", fontsize=11)
    ax.set_ylabel("Total JJAS Rainfall (mm)", fontsize=11)
    ax.set_title(f"Onset Date vs Total JJAS Rainfall ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate interpretation
    if r < -0.1:
        ax.text(0.02, 0.02, "Earlier onset → more rain",
                transform=ax.transAxes, fontsize=9, fontstyle="italic",
                color="#666666", va="bottom")
    elif r > 0.1:
        ax.text(0.02, 0.02, "Later onset → more rain (unexpected)",
                transform=ax.transAxes, fontsize=9, fontstyle="italic",
                color="#666666", va="bottom")
    else:
        ax.text(0.02, 0.02, "Weak relationship",
                transform=ax.transAxes, fontsize=9, fontstyle="italic",
                color="#666666", va="bottom")

    savefig(fig, "fig11_onset_vs_rainfall.png")


# =========================================================================== #
#  FIGURE 12: Autocorrelation Function (4 IMD Regions)
# =========================================================================== #
def fig12(tp):
    print("Fig 12: Autocorrelation function (4 regions)...")
    max_lag = 30

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (rname, rbox), lbl in zip(axes.flat, IMD_REGIONS.items(), panel_labels):
        # Collect all JJAS daily values across years for this region
        all_acf = []

        for year in range(ANALYSIS_Y0, ANALYSIS_Y1 + 1):
            try:
                jjas_yr = tp.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
                rm = region_mean(jjas_yr, rbox["lat"], rbox["lon"]).values
                if len(rm) < max_lag + 10:
                    continue

                rm_dm = rm - np.mean(rm)
                var = np.var(rm_dm)
                if var < 1e-10:
                    continue

                acf = np.zeros(max_lag + 1)
                for lag in range(max_lag + 1):
                    if lag == 0:
                        acf[lag] = 1.0
                    else:
                        acf[lag] = np.mean(rm_dm[lag:] * rm_dm[:-lag]) / var
                all_acf.append(acf)
            except Exception:
                continue

        if not all_acf:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        mean_acf = np.mean(all_acf, axis=0)
        std_acf = np.std(all_acf, axis=0)
        lags = np.arange(max_lag + 1)

        # Plot
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                         alpha=0.2, color=rbox["color"])
        ax.plot(lags, mean_acf, color=rbox["color"], lw=2.5, zorder=5, marker="o",
                markersize=3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)

        # 95% confidence bound (white noise)
        n_eff = 122  # approximate JJAS days
        ci = 1.96 / np.sqrt(n_eff)
        ax.axhline(ci, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-ci, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.text(max_lag - 1, ci + 0.02, "95% CI", fontsize=7, color="gray",
                ha="right", va="bottom")

        # e-folding time
        e_fold = None
        for i in range(1, len(mean_acf)):
            if mean_acf[i] < 1.0 / np.e:
                e_fold = i
                break

        if e_fold is not None:
            ax.axvline(e_fold, color=rbox["color"], ls="--", lw=1, alpha=0.5)
            ax.text(e_fold + 0.5, 0.85, f"e-fold: {e_fold}d",
                    fontsize=8, color=rbox["color"],
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        ax.set_title(rname, fontsize=12, fontweight="bold", color=rbox["color"])
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation")
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.3, 1.05)
        ax.grid(alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, lbl)

    fig.suptitle(f"Daily JJAS Precipitation Autocorrelation ({ANALYSIS_Y0}–{ANALYSIS_Y1})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig12_autocorrelation.png")


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    print(f"{'═' * 60}")
    print(f"  Advanced Monsoon Dynamics Analysis (ERA5)")
    print(f"{'═' * 60}")
    print(f"  WB2 zarr: {WB2_ZARR_15}")
    print(f"  India box: {LAT_S}–{LAT_N}°N, {LON_W}–{LON_E}°E")
    print(f"  Period: {ANALYSIS_Y0}–{ANALYSIS_Y1}")
    print(f"  Output dir: {OUT_DIR}\n")

    tp = load_wb2_tp_india(y0=ANALYSIS_Y0, y1=ANALYSIS_Y1)
    print()

    fig_funcs = {
        1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06,
        7: fig07, 8: fig08, 9: fig09,
    }

    onset_dates = None
    for n in sorted(fig_funcs.keys()):
        try:
            fig_funcs[n](tp)
        except Exception as e:
            print(f"  ✗ Fig {n:02d} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Fig 10 returns onset_dates for Fig 11
    try:
        onset_dates = fig10(tp)
    except Exception as e:
        print(f"  ✗ Fig 10 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        fig11(tp, onset_dates=onset_dates)
    except Exception as e:
        print(f"  ✗ Fig 11 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        fig12(tp)
    except Exception as e:
        print(f"  ✗ Fig 12 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Report summary
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY — Output: {OUT_DIR}")
    print(f"{'═' * 60}")
    if os.path.isdir(OUT_DIR):
        files = sorted(os.listdir(OUT_DIR))
        for f in files:
            fpath = os.path.join(OUT_DIR, f)
            if os.path.isfile(fpath):
                sz = os.path.getsize(fpath) / 1024
                print(f"    {f:45s} {sz:8.0f} KB")
        print(f"  Total files: {len(files)}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
