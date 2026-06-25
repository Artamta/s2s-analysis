#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_ecmwf_tp.py — Publication-Quality ECMWF S2S Hindcast TP Exploration
=============================================================================
Generates 10 journal-ready figures (npj Climate / GRL level) exploring
ECMWF S2S hindcast total precipitation patterns over India.

Features:
  • Cartopy projections with 50m coastlines, state borders
  • Scientific colormaps, filled contours, panel labels (a)(b)
  • Proper accumulated-TP → daily-rate conversion
  • Ensemble mean/spread from 11 members (1 CF + 10 PF)
  • ERA5 truth comparison from WeatherBench2

Figures (saved to ./ecmwf_tp_exploration/):
  01. ECMWF TP Climatology by Lead Week (W1-W6)
  02. ECMWF vs ERA5 Bias by Lead Week (W1-W6)
  03. Lead-Time Skill Degradation (W1, W3, W6 vs ERA5)
  04. Ensemble Spread Growth (W1-W6)
  05. All-India TP vs Lead Time (line plot)
  06. ECMWF Seasonal Cycle (Jan, Apr, Jul, Oct W1 climatology)
  07. Interannual Correlation (W1, W3 vs ERA5)
  08. ECMWF Model Climate vs ERA5 Climate (3 panels)
  09. Ensemble Member Spread (11 members, one init)
  10. Regional TP by Lead Time (4 IMD regions)

Usage:
    python explore_ecmwf_tp.py             # all figures
    python explore_ecmwf_tp.py --fig 1 5   # specific figures
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
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =========================================================================== #
#  Config
# =========================================================================== #
ECMWF_ROOT = "/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data"

WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

# India domain
LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

# IMD sub-regions
IMD_REGIONS = {
    "Northwest India":  {"lat": (25.0, 36.0), "lon": (68.0, 80.0), "color": "#e41a1c"},
    "Central India":    {"lat": (20.0, 28.0), "lon": (74.0, 86.0), "color": "#377eb8"},
    "South Peninsula":  {"lat": ( 8.0, 20.0), "lon": (74.0, 82.0), "color": "#4daf4a"},
    "East & NE India":  {"lat": (20.0, 30.0), "lon": (86.0, 98.0), "color": "#ff7f00"},
}

# JJAS init dates (biweekly subset for speed; full list commented)
JJAS_INITS_SUBSET = ["0601", "0615", "0702", "0716", "0803", "0817"]
JJAS_INITS_ALL = [
    "0601", "0604", "0608", "0611", "0615", "0618", "0622", "0625", "0629",
    "0702", "0706", "0709", "0713", "0716", "0720", "0723", "0727", "0730",
    "0803", "0806", "0810", "0813", "0817", "0820", "0824", "0827",
]

# Which inits to use (configurable)
JJAS_INITS = JJAS_INITS_SUBSET

# Seasonal cycle representative inits
SEASONAL_INITS = {
    "Jan": "0102", "Apr": "0402", "Jul": "0702", "Oct": "1001",
}

# Week definitions (0-based step indices)
WEEK_DEFS = {
    "W1": (0, 7),   # steps 0-6 → days 1-7
    "W2": (7, 14),   # steps 7-13 → days 8-14
    "W3": (14, 21),  # steps 14-20 → days 15-21
    "W4": (21, 28),  # steps 21-27 → days 22-28
    "W5": (28, 35),  # steps 28-34 → days 29-35
    "W6": (35, 42),  # steps 35-41 → days 36-42
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecmwf_tp_exploration")
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


def bias_cmap():
    """Brown–white–teal diverging colormap (BrBG-style)."""
    colors = [
        "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5",
        "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
    ]
    return mcolors.LinearSegmentedColormap.from_list("bias", colors, N=256)


def spread_cmap():
    """YlOrRd-like for ensemble spread."""
    colors = [
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
        "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
    ]
    return mcolors.LinearSegmentedColormap.from_list("spread", colors, N=256)


def corr_cmap():
    """Red–white–blue for correlation."""
    colors = [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
        "#f7f7f7",
        "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
    ]
    return mcolors.LinearSegmentedColormap.from_list("corr", colors, N=256)


# =========================================================================== #
#  Data loading: ECMWF GRIB
# =========================================================================== #
def load_ecmwf_init(mmdd, load_pf=True):
    """Load ECMWF CF (and optionally PF) for a single init date.

    Returns accumulated TP as xarray DataArrays.
    CF shape: (time=20, step=46, lat=34, lon=34)
    PF shape: (number=10, time=20, step=46, lat=34, lon=34)
    """
    cf_path = os.path.join(ECMWF_ROOT, f"tp_cf_{mmdd}.grib")
    pf_path = os.path.join(ECMWF_ROOT, f"tp_pf_{mmdd}.grib")

    print(f"    Loading CF {mmdd}...", end=" ", flush=True)
    ds_cf = xr.open_dataset(cf_path, engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
    tp_cf = ds_cf["tp"].load()
    print(f"shape={tp_cf.shape}", flush=True)

    tp_pf = None
    if load_pf and os.path.exists(pf_path):
        print(f"    Loading PF {mmdd}...", end=" ", flush=True)
        ds_pf = xr.open_dataset(pf_path, engine="cfgrib",
                                backend_kwargs={"indexpath": ""})
        tp_pf = ds_pf["tp"].load()
        print(f"shape={tp_pf.shape}", flush=True)

    return tp_cf, tp_pf


def accum_to_weekly_rate(tp_accum, week_key):
    """Convert accumulated TP to weekly-mean daily rate (mm/day).

    tp_accum: DataArray with 'step' as one dimension.
    week_key: e.g. 'W1', 'W2', ...

    For W1: rate = tp[:, 6, :, :] / 7  (since accumulation starts at 0)
    For Wn: rate = (tp[:, end-1, :, :] - tp[:, start-1, :, :]) / 7
    """
    s0, s1 = WEEK_DEFS[week_key]
    if s0 == 0:
        # W1: accumulation from 0 to step 6
        weekly_accum = tp_accum.isel(step=s1 - 1)
    else:
        weekly_accum = tp_accum.isel(step=s1 - 1) - tp_accum.isel(step=s0 - 1)
    return weekly_accum / 7.0


def accum_to_daily_rates(tp_accum):
    """Convert accumulated TP to daily rates for all 46 steps.

    Returns array with same shape, daily rate for each step.
    Day 1 (step 0): tp[:, 0, :, :]
    Day i (step i): tp[:, i, :, :] - tp[:, i-1, :, :]
    """
    # diff along step dimension
    daily = tp_accum.diff("step")
    # prepend the first step (which IS day-1 accumulation)
    first = tp_accum.isel(step=0)
    daily = xr.concat([first, daily], dim="step")
    return daily


def combine_cf_pf(tp_cf, tp_pf):
    """Combine CF and PF into a single ensemble array with 'member' dim.

    CF (no number dim) → member 0
    PF (number=1..10) → members 1..10
    Returns: DataArray with dims (member, time, step, lat, lon)
    """
    cf_expanded = tp_cf.expand_dims("member", axis=0)
    cf_expanded["member"] = [0]

    if tp_pf is not None:
        pf_renamed = tp_pf.rename({"number": "member"})
        pf_renamed["member"] = np.arange(1, tp_pf.sizes.get("number", 10) + 1)
        combined = xr.concat([cf_expanded, pf_renamed], dim="member")
    else:
        combined = cf_expanded

    return combined


# =========================================================================== #
#  Data loading: ERA5 truth
# =========================================================================== #
def load_era5_jjas_clim():
    """Load ERA5 JJAS daily TP climatology over India from WB2.

    Returns daily-mean field averaged over 2000-2019 JJAS.
    """
    print("  Loading ERA5 from WB2...", flush=True)
    ds = xr.open_zarr(WB2_ZARR)
    tp24 = ds["total_precipitation_24hr"]
    # Select domain — ERA5 lat is ascending
    tp24 = tp24.sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))
    # Select 06Z for daily total
    tp24 = tp24.sel(time=tp24["time.hour"] == 6)
    # Select 2000-2019
    tp24 = tp24.sel(time=slice("2000-01-01", "2019-12-31"))
    # JJAS
    tp24 = tp24.sel(time=tp24["time.month"].isin([6, 7, 8, 9]))
    # Convert m to mm, ensure (time, latitude, longitude) order
    tp_mm = (tp24 * 1000.0).load()
    tp_mm = tp_mm.transpose("time", "latitude", "longitude")
    print(f"  ERA5 loaded: {tp_mm.shape}, {tp_mm.nbytes/1e6:.0f} MB", flush=True)
    return tp_mm


def get_era5_weekly_truth(era5_daily, mmdd, week_key, year):
    """Get ERA5 7-day mean for a specific init date, week, and year.

    mmdd: init date string e.g. '0702'
    week_key: e.g. 'W1'
    year: hindcast year

    Returns 2D field (lat, lon) of 7-day mean TP in mm/day.
    """
    month = int(mmdd[:2])
    day = int(mmdd[2:])
    s0, s1 = WEEK_DEFS[week_key]
    # Lead days: s0+1 to s1 (1-indexed days from init)
    init_date = datetime.date(year, month, day)
    start_date = init_date + datetime.timedelta(days=s0)
    end_date = init_date + datetime.timedelta(days=s1 - 1)

    t0 = np.datetime64(f"{year}-{start_date.month:02d}-{start_date.day:02d}")
    t1 = np.datetime64(f"{year}-{end_date.month:02d}-{end_date.day:02d}")

    try:
        subset = era5_daily.sel(time=slice(t0, t1))
        if subset.sizes["time"] == 0:
            return None
        return subset.mean("time")
    except Exception:
        return None


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
    return gaussian_filter(np.nan_to_num(data, nan=0.0), sigma=sigma)


def savefig(fig, name, dpi=250):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sz = os.path.getsize(path) / 1e6
    print(f"  ✓ saved {path} ({sz:.1f} MB)")


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


def india_extent(ax):
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=ccrs.PlateCarree())


def panel_label(ax, label, x=0.02, y=0.95):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


# =========================================================================== #
#  Master data loader
# =========================================================================== #
class ECMWFDataCache:
    """Cache ECMWF and ERA5 data to avoid redundant loading."""

    def __init__(self):
        self._cf_cache = {}
        self._pf_cache = {}
        self._era5_jjas = None
        self._weekly_cache = {}  # (mmdd, week_key) -> ensemble-mean weekly rate
        self._weekly_spread_cache = {}

    def get_cf_pf(self, mmdd, load_pf=True):
        """Get CF/PF for a single init, caching."""
        if mmdd not in self._cf_cache:
            cf, pf = load_ecmwf_init(mmdd, load_pf=load_pf)
            self._cf_cache[mmdd] = cf
            if pf is not None:
                self._pf_cache[mmdd] = pf
        return self._cf_cache[mmdd], self._pf_cache.get(mmdd)

    def get_era5(self):
        """Get ERA5 JJAS daily data."""
        if self._era5_jjas is None:
            self._era5_jjas = load_era5_jjas_clim()
        return self._era5_jjas

    def get_weekly_ensmean(self, mmdd, week_key):
        """Get ensemble-mean weekly rate for one init, cached.

        Returns: (time=20, lat=34, lon=34) field in mm/day.
        """
        key = (mmdd, week_key)
        if key not in self._weekly_cache:
            cf, pf = self.get_cf_pf(mmdd, load_pf=True)
            combined = combine_cf_pf(cf, pf)
            weekly = accum_to_weekly_rate(combined, week_key)
            self._weekly_cache[key] = weekly.mean("member")
            self._weekly_spread_cache[key] = weekly.std("member")
        return self._weekly_cache[key]

    def get_weekly_spread(self, mmdd, week_key):
        """Get ensemble spread (std) for one init, cached."""
        key = (mmdd, week_key)
        if key not in self._weekly_spread_cache:
            self.get_weekly_ensmean(mmdd, week_key)
        return self._weekly_spread_cache[key]


# Global cache
CACHE = ECMWFDataCache()


def get_jjas_weekly_climatology(week_key):
    """Compute JJAS climatology for a given week across sampled inits.

    Returns: 2D (lat, lon) mean field in mm/day, averaged over
    all sampled inits × 20 years.
    """
    all_means = []
    for mmdd in JJAS_INITS:
        wm = CACHE.get_weekly_ensmean(mmdd, week_key)
        # Average over time (20 years)
        all_means.append(wm.mean("time"))
    # Average over inits
    return sum(all_means) / len(all_means)


def get_jjas_weekly_spread_clim(week_key):
    """Compute JJAS ensemble spread climatology for a given week."""
    all_spreads = []
    for mmdd in JJAS_INITS:
        ws = CACHE.get_weekly_spread(mmdd, week_key)
        all_spreads.append(ws.mean("time"))
    return sum(all_spreads) / len(all_spreads)


# =========================================================================== #
#  FIGURE 1: ECMWF TP Climatology by Lead Week (W1-W6)
# =========================================================================== #
def fig01():
    print("Fig 01: ECMWF TP Climatology by Lead Week...")
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), subplot_kw={"projection": proj})
    cmap = precip_cmap()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, (wk, ax) in enumerate(zip(["W1", "W2", "W3", "W4", "W5", "W6"], axes.flat)):
        india_extent(ax)
        add_map_features(ax, rivers=True)
        clim = get_jjas_weekly_climatology(wk)
        lon = clim.longitude.values if hasattr(clim, "longitude") else clim.coords["longitude"].values
        lat = clim.latitude.values if hasattr(clim, "latitude") else clim.coords["latitude"].values
        data = clim.values if hasattr(clim, "values") else np.array(clim)
        sm = smooth_field(data, sigma=0.6)
        levels = np.arange(0, 18.1, 1.0)
        cf = ax.contourf(lon, lat, sm, levels=levels, cmap=cmap,
                         extend="max", transform=proj)
        ax.contour(lon, lat, sm, levels=levels[::3], colors="k",
                   linewidths=0.3, alpha=0.4, transform=proj)
        s0, s1 = WEEK_DEFS[wk]
        ax.set_title(f"{wk} (days {s0+1}–{s1})", fontsize=12, pad=8)
        panel_label(ax, labels[i])

    # Shared colorbar
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle(f"ECMWF S2S JJAS TP Climatology by Lead Week\n"
                 f"(2000–2019, {len(JJAS_INITS)} inits, 11-member mean)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    savefig(fig, "fig01_ecmwf_tp_clim_by_week.png")


# =========================================================================== #
#  FIGURE 2: ECMWF vs ERA5 Bias by Lead Week
# =========================================================================== #
def fig02():
    print("Fig 02: ECMWF vs ERA5 Bias by Lead Week...")
    era5_daily = CACHE.get_era5()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), subplot_kw={"projection": proj})
    cmap = bias_cmap()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, (wk, ax) in enumerate(zip(["W1", "W2", "W3", "W4", "W5", "W6"], axes.flat)):
        india_extent(ax)
        add_map_features(ax, rivers=True)

        # ECMWF climatology
        ecmwf_clim = get_jjas_weekly_climatology(wk)

        # ERA5 truth: for each init and year, get the corresponding 7-day mean
        era5_fields = []
        for mmdd in JJAS_INITS:
            for year in HINDCAST_YEARS:
                era5_wk = get_era5_weekly_truth(era5_daily, mmdd, wk, year)
                if era5_wk is not None:
                    era5_fields.append(era5_wk.values)

        if len(era5_fields) == 0:
            ax.set_title(f"{wk} — no ERA5 data", fontsize=12)
            panel_label(ax, labels[i])
            continue

        # Regrid ERA5 to ECMWF grid (nearest neighbor)
        era5_mean_raw = np.nanmean(era5_fields, axis=0)
        ecmwf_lat = ecmwf_clim.latitude.values
        ecmwf_lon = ecmwf_clim.longitude.values

        # Use ERA5 lat/lon directly from the loaded dataset
        era5_lat = era5_daily.latitude.values
        era5_lon = era5_daily.longitude.values

        # Create xr DataArray for ERA5 mean and interpolate
        # Ensure shape matches: data should be (lat, lon)
        era5_da = xr.DataArray(era5_mean_raw,
                               dims=["latitude", "longitude"],
                               coords={"latitude": era5_lat[:era5_mean_raw.shape[0]],
                                       "longitude": era5_lon[:era5_mean_raw.shape[1]]})
        era5_interp = era5_da.interp(latitude=ecmwf_lat, longitude=ecmwf_lon,
                                     method="nearest")

        bias = ecmwf_clim.values - era5_interp.values
        sm = smooth_field(bias, sigma=0.5)
        levels = np.arange(-6, 6.1, 0.5)
        cf = ax.contourf(ecmwf_lon, ecmwf_lat, sm, levels=levels, cmap=cmap,
                         extend="both", transform=proj)
        ax.contour(ecmwf_lon, ecmwf_lat, sm, levels=[0], colors="k",
                   linewidths=0.8, transform=proj)

        s0, s1 = WEEK_DEFS[wk]
        bias_mean = float(np.nanmean(bias))
        ax.set_title(f"{wk} (days {s0+1}–{s1}), mean={bias_mean:+.1f}", fontsize=11, pad=8)
        panel_label(ax, labels[i])

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Bias: ECMWF − ERA5 (mm day⁻¹)", fontsize=11)

    fig.suptitle("ECMWF S2S TP Bias vs ERA5 by Lead Week (JJAS 2000–2019)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    savefig(fig, "fig02_ecmwf_era5_bias_by_week.png")


# =========================================================================== #
#  FIGURE 3: Lead-Time Skill Degradation
# =========================================================================== #
def fig03():
    print("Fig 03: Lead-Time Skill Degradation...")
    era5_daily = CACHE.get_era5()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), subplot_kw={"projection": proj})
    cmap = precip_cmap()
    levels = np.arange(0, 18.1, 1.0)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    titles = ["ECMWF W1\n(days 1–7)", "ECMWF W3\n(days 15–21)",
              "ECMWF W6\n(days 36–42)", "ERA5 JJAS\nClimatology"]

    # ECMWF W1, W3, W6
    for i, wk in enumerate(["W1", "W3", "W6"]):
        ax = axes[i]
        india_extent(ax)
        add_map_features(ax, rivers=True, gridlines=False)
        clim = get_jjas_weekly_climatology(wk)
        sm = smooth_field(clim.values, sigma=0.6)
        cf = ax.contourf(clim.longitude, clim.latitude, sm, levels=levels,
                         cmap=cmap, extend="max", transform=proj)
        ax.set_title(titles[i], fontsize=11, pad=8)
        panel_label(ax, labels[i])

    # ERA5 climatology
    ax = axes[3]
    india_extent(ax)
    add_map_features(ax, rivers=True, gridlines=False)
    era5_jjas_clim = era5_daily.mean("time").transpose("latitude", "longitude")
    sm_era5 = smooth_field(era5_jjas_clim.values, sigma=0.6)
    cf = ax.contourf(era5_jjas_clim.longitude, era5_jjas_clim.latitude, sm_era5,
                     levels=levels, cmap=cmap, extend="max", transform=proj)
    ax.set_title(titles[3], fontsize=11, pad=8)
    panel_label(ax, labels[3])

    cax = fig.add_axes([0.25, -0.02, 0.5, 0.02])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle("Skill Degradation: ECMWF JJAS TP at Increasing Lead vs ERA5",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.08)
    savefig(fig, "fig03_skill_degradation.png")


# =========================================================================== #
#  FIGURE 4: Ensemble Spread Growth (W1-W6)
# =========================================================================== #
def fig04():
    print("Fig 04: Ensemble Spread Growth...")
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), subplot_kw={"projection": proj})
    cmap = spread_cmap()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    vmax_global = 0
    spreads = []
    for wk in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        sp = get_jjas_weekly_spread_clim(wk)
        spreads.append(sp)
        vmax_global = max(vmax_global, float(np.nanmax(sp.values)))

    levels = np.linspace(0, min(vmax_global, 10), 15)

    for i, (wk, sp, ax) in enumerate(zip(
            ["W1", "W2", "W3", "W4", "W5", "W6"], spreads, axes.flat)):
        india_extent(ax)
        add_map_features(ax, rivers=False)
        sm = smooth_field(sp.values, sigma=0.5)
        cf = ax.contourf(sp.longitude, sp.latitude, sm, levels=levels,
                         cmap=cmap, extend="max", transform=proj)
        s0, s1 = WEEK_DEFS[wk]
        mean_sp = float(np.nanmean(sp.values))
        ax.set_title(f"{wk} (days {s0+1}–{s1}), σ̄={mean_sp:.2f}", fontsize=11, pad=8)
        panel_label(ax, labels[i])

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Ensemble Spread σ (mm day⁻¹)", fontsize=11)

    fig.suptitle("ECMWF S2S Ensemble Spread (Inter-Member Std Dev) by Lead Week\n"
                 f"(JJAS 2000–2019, {len(JJAS_INITS)} inits)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    savefig(fig, "fig04_ensemble_spread_growth.png")


# =========================================================================== #
#  FIGURE 5: All-India TP vs Lead Time
# =========================================================================== #
def fig05():
    print("Fig 05: All-India TP vs Lead Time...")

    # Compute daily rates for each init, average across ensemble + inits + years
    all_daily_means = []
    all_daily_spreads = []

    for mmdd in JJAS_INITS:
        cf, pf = CACHE.get_cf_pf(mmdd, load_pf=True)
        combined = combine_cf_pf(cf, pf)
        daily = accum_to_daily_rates(combined)
        # Area mean over India
        ai_daily = area_mean(daily, lat_dim="latitude")
        # Mean across members, then years
        ens_mean = ai_daily.mean("member").mean("time")
        ens_spread = ai_daily.std("member").mean("time")
        all_daily_means.append(ens_mean.values)
        all_daily_spreads.append(ens_spread.values)

    # Average across inits
    mean_curve = np.nanmean(all_daily_means, axis=0)
    spread_curve = np.nanmean(all_daily_spreads, axis=0)
    lead_days = np.arange(1, len(mean_curve) + 1)

    # ERA5 climatological mean for JJAS
    era5_daily = CACHE.get_era5()
    era5_ai = area_mean(era5_daily).mean("time").values
    era5_line = float(era5_ai)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Ensemble mean ± spread
    ax.fill_between(lead_days, mean_curve - spread_curve, mean_curve + spread_curve,
                    alpha=0.2, color="#2171b5", label="Ensemble spread (±1σ)")
    ax.plot(lead_days, mean_curve, color="#08519c", lw=2.5, label="ECMWF ensemble mean",
            zorder=5)

    # ERA5 climatology line
    ax.axhline(era5_line, color="#d62728", ls="--", lw=2, alpha=0.8,
               label=f"ERA5 JJAS clim ({era5_line:.1f} mm/day)")

    # Week boundaries
    for wk, (s0, s1) in WEEK_DEFS.items():
        ax.axvline(s1, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.text(s0 + 3.5, mean_curve.max() * 1.08,
                wk, fontsize=8, color="gray", ha="center", va="bottom")

    ax.set_xlabel("Lead Day", fontsize=12)
    ax.set_ylabel("Precipitation (mm day⁻¹)", fontsize=12)
    ax.set_title("All-India Mean TP vs Lead Day\n"
                 f"(ECMWF S2S, JJAS 2000–2019, {len(JJAS_INITS)} inits, cos-lat weighted)",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_xlim(1, 46)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig05_all_india_tp_vs_lead.png")


# =========================================================================== #
#  FIGURE 6: ECMWF Seasonal Cycle (Jan, Apr, Jul, Oct)
# =========================================================================== #
def fig06():
    print("Fig 06: ECMWF Seasonal Cycle (4 seasons, W1)...")
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), subplot_kw={"projection": proj})
    cmap = precip_cmap()
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (month_name, mmdd) in enumerate(SEASONAL_INITS.items()):
        ax = axes[i]
        india_extent(ax)
        add_map_features(ax, rivers=True, gridlines=False)

        cf, pf = CACHE.get_cf_pf(mmdd, load_pf=True)
        combined = combine_cf_pf(cf, pf)
        weekly = accum_to_weekly_rate(combined, "W1")
        clim = weekly.mean("member").mean("time")

        sm = smooth_field(clim.values, sigma=0.6)
        vmax = 18 if month_name in ["Jul"] else 10
        levels = np.linspace(0, vmax, 13)
        cf_plot = ax.contourf(clim.longitude, clim.latitude, sm, levels=levels,
                              cmap=cmap, extend="max", transform=proj)
        ax.set_title(f"{month_name} (init {mmdd})", fontsize=12, pad=8)
        panel_label(ax, labels[i])

    cax = fig.add_axes([0.25, -0.02, 0.5, 0.02])
    cb = fig.colorbar(cf_plot, cax=cax, orientation="horizontal")
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle("ECMWF S2S Seasonal Cycle: W1 TP Climatology by Season\n(2000–2019)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.08)
    savefig(fig, "fig06_seasonal_cycle.png")


# =========================================================================== #
#  FIGURE 7: Interannual Correlation (W1, W3 vs ERA5)
# =========================================================================== #
def fig07():
    print("Fig 07: Interannual Correlation...")
    era5_daily = CACHE.get_era5()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})
    cmap = corr_cmap()
    labels = ["(a)", "(b)"]

    for idx, (wk, ax) in enumerate(zip(["W1", "W3"], axes)):
        india_extent(ax)
        add_map_features(ax, rivers=True)

        # For each year, compute ECMWF and ERA5 JJAS-mean TP
        ecmwf_yearly = []
        era5_yearly = []

        for yi, year in enumerate(HINDCAST_YEARS):
            ecmwf_fields = []
            era5_fields = []
            for mmdd in JJAS_INITS:
                wm = CACHE.get_weekly_ensmean(mmdd, wk)
                ecmwf_fields.append(wm.isel(time=yi).values)

                era5_wk = get_era5_weekly_truth(era5_daily, mmdd, wk, year)
                if era5_wk is not None:
                    era5_fields.append(era5_wk.values)

            ecmwf_yearly.append(np.nanmean(ecmwf_fields, axis=0))
            if len(era5_fields) > 0:
                era5_yearly.append(np.nanmean(era5_fields, axis=0))
            else:
                era5_yearly.append(np.full_like(ecmwf_fields[0], np.nan))

        ecmwf_arr = np.array(ecmwf_yearly)  # (20, lat, lon)
        era5_arr = np.array(era5_yearly)     # (20, lat_era5, lon_era5)

        # Interpolate ERA5 to ECMWF grid
        sample_clim = get_jjas_weekly_climatology("W1")
        ecmwf_lat = sample_clim.latitude.values
        ecmwf_lon = sample_clim.longitude.values

        # Use ERA5 lat/lon directly from the loaded dataset
        era5_lat = era5_daily.latitude.values
        era5_lon = era5_daily.longitude.values

        # Interpolate each year
        era5_regridded = np.zeros((20, len(ecmwf_lat), len(ecmwf_lon)))
        for yi in range(20):
            da_tmp = xr.DataArray(era5_arr[yi],
                                  dims=["latitude", "longitude"],
                                  coords={"latitude": era5_lat[:era5_arr.shape[1]],
                                          "longitude": era5_lon[:era5_arr.shape[2]]})
            era5_regridded[yi] = da_tmp.interp(latitude=ecmwf_lat,
                                                longitude=ecmwf_lon,
                                                method="nearest").values

        # Compute per-gridpoint correlation
        nlat, nlon = len(ecmwf_lat), len(ecmwf_lon)
        corr_map = np.full((nlat, nlon), np.nan)
        for ilat in range(nlat):
            for ilon in range(nlon):
                ts_ecmwf = ecmwf_arr[:, ilat, ilon]
                ts_era5 = era5_regridded[:, ilat, ilon]
                mask = ~(np.isnan(ts_ecmwf) | np.isnan(ts_era5))
                if mask.sum() > 5:
                    r, _ = pearsonr(ts_ecmwf[mask], ts_era5[mask])
                    corr_map[ilat, ilon] = r

        sm = smooth_field(corr_map, sigma=0.5)
        levels = np.arange(-1, 1.05, 0.1)
        cf = ax.contourf(ecmwf_lon, ecmwf_lat, sm, levels=levels,
                         cmap=cmap, extend="both", transform=proj)
        ax.contour(ecmwf_lon, ecmwf_lat, sm, levels=[0], colors="k",
                   linewidths=0.8, transform=proj)

        mean_corr = float(np.nanmean(corr_map))
        s0, s1 = WEEK_DEFS[wk]
        ax.set_title(f"{wk} (days {s0+1}–{s1}), mean r = {mean_corr:.2f}",
                     fontsize=12, pad=8)
        panel_label(ax, labels[idx])

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Pearson Correlation (r)", fontsize=11)

    fig.suptitle("Interannual Correlation: ECMWF vs ERA5 JJAS TP (2000–2019)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.12)
    savefig(fig, "fig07_interannual_correlation.png")


# =========================================================================== #
#  FIGURE 8: Model Climate vs ERA5 Climate (3 panels)
# =========================================================================== #
def fig08():
    print("Fig 08: ECMWF Model Climate vs ERA5...")
    era5_daily = CACHE.get_era5()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": proj})
    labels = ["(a)", "(b)", "(c)"]
    titles = ["ECMWF W1 JJAS\nClimatology", "ERA5 JJAS\nClimatology",
              "Difference\n(ECMWF − ERA5)"]

    # ECMWF W1 climatology
    ecmwf_clim = get_jjas_weekly_climatology("W1")
    ecmwf_lat = ecmwf_clim.latitude.values
    ecmwf_lon = ecmwf_clim.longitude.values

    # ERA5 JJAS climatology
    era5_clim = era5_daily.mean("time")
    era5_interp = era5_clim.interp(latitude=ecmwf_lat, longitude=ecmwf_lon,
                                    method="nearest")

    diff = ecmwf_clim.values - era5_interp.values

    cmap_p = precip_cmap()
    cmap_d = bias_cmap()
    levels_p = np.arange(0, 18.1, 1.0)
    levels_d = np.arange(-6, 6.1, 0.5)

    datasets = [
        (ecmwf_clim.values, ecmwf_lat, ecmwf_lon, levels_p, cmap_p, "max",
         "Precipitation (mm day⁻¹)"),
        (era5_interp.values, ecmwf_lat, ecmwf_lon, levels_p, cmap_p, "max",
         "Precipitation (mm day⁻¹)"),
        (diff, ecmwf_lat, ecmwf_lon, levels_d, cmap_d, "both",
         "Bias (mm day⁻¹)"),
    ]

    for i, (data, lat, lon, lvls, cm, ext, cb_label) in enumerate(datasets):
        ax = axes[i]
        india_extent(ax)
        add_map_features(ax, rivers=True, gridlines=False)
        sm = smooth_field(data, sigma=0.5)
        cf = ax.contourf(lon, lat, sm, levels=lvls, cmap=cm,
                         extend=ext, transform=proj)
        if i == 2:
            ax.contour(lon, lat, sm, levels=[0], colors="k",
                       linewidths=0.8, transform=proj)
        ax.set_title(titles[i], fontsize=11, pad=8)
        panel_label(ax, labels[i])
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.85,
                          pad=0.06, aspect=25)
        cb.set_label(cb_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle("ECMWF S2S Model Climate vs ERA5 (JJAS W1, 2000–2019)",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.15)
    savefig(fig, "fig08_model_vs_era5_climate.png")


# =========================================================================== #
#  FIGURE 9: Ensemble Member Spread (one init, 11 members)
# =========================================================================== #
def fig09():
    print("Fig 09: Ensemble Member Spread...")
    # Use init 0702, year index 19 (= 2019)
    mmdd = "0702"
    year_idx = 19  # 2019

    cf_data, pf_data = CACHE.get_cf_pf(mmdd, load_pf=True)
    combined = combine_cf_pf(cf_data, pf_data)
    weekly = accum_to_weekly_rate(combined, "W1")

    # Select year 2019
    member_fields = weekly.isel(time=year_idx)  # (member, lat, lon)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 4, figsize=(18, 14), subplot_kw={"projection": proj})
    cmap = precip_cmap()
    levels = np.arange(0, 25.1, 1.5)

    n_members = member_fields.sizes["member"]
    for i, ax in enumerate(axes.flat):
        india_extent(ax)
        add_map_features(ax, gridlines=False, borders=True, rivers=False)

        if i < n_members:
            data = member_fields.isel(member=i)
            sm = smooth_field(data.values, sigma=0.4)
            cf = ax.contourf(data.longitude, data.latitude, sm, levels=levels,
                             cmap=cmap, extend="max", transform=proj)
            label_txt = "CF" if i == 0 else f"PF-{i}"
            ax.set_title(f"Member {label_txt}", fontsize=10, pad=5)
            panel_label(ax, f"({chr(97+i)})", x=0.03, y=0.97)
        elif i == n_members:
            # Ensemble mean
            ens_mean = member_fields.mean("member")
            sm = smooth_field(ens_mean.values, sigma=0.4)
            cf = ax.contourf(ens_mean.longitude, ens_mean.latitude, sm,
                             levels=levels, cmap=cmap, extend="max", transform=proj)
            ax.set_title("Ensemble Mean", fontsize=10, fontweight="bold", pad=5)
            panel_label(ax, f"({chr(97+i)})", x=0.03, y=0.97)
        else:
            ax.set_visible(False)

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle(f"ECMWF Ensemble Members: W1 TP (init {mmdd}, year 2019)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.12, wspace=0.05)
    savefig(fig, "fig09_ensemble_members.png")


# =========================================================================== #
#  FIGURE 10: Regional TP by Lead Time (4 IMD regions)
# =========================================================================== #
def fig10():
    print("Fig 10: Regional TP by Lead Time...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # ERA5 regional climatology
    era5_daily = CACHE.get_era5()

    for ax, (rname, rbox) in zip(axes.flat, IMD_REGIONS.items()):
        # Compute daily rates for each init
        all_daily = []
        all_spread = []
        for mmdd in JJAS_INITS:
            cf, pf = CACHE.get_cf_pf(mmdd, load_pf=True)
            combined = combine_cf_pf(cf, pf)
            daily = accum_to_daily_rates(combined)

            # Regional mean
            lat_range = rbox["lat"]
            lon_range = rbox["lon"]

            # Select region — latitude may be descending in ECMWF
            lat_vals = daily.latitude.values
            if lat_vals[0] > lat_vals[-1]:
                lat_sel = slice(lat_range[1], lat_range[0])
            else:
                lat_sel = slice(lat_range[0], lat_range[1])

            regional = daily.sel(latitude=lat_sel,
                                 longitude=slice(lon_range[0], lon_range[1]))
            # Cosine weighted mean
            w = np.cos(np.deg2rad(regional.latitude))
            w = w / w.mean()
            weighted = regional.weighted(xr.DataArray(w, dims=["latitude"]))
            rm = weighted.mean(dim=["latitude", "longitude"])

            ens_mean = rm.mean("member").mean("time").values
            ens_spread = rm.std("member").mean("time").values
            all_daily.append(ens_mean)
            all_spread.append(ens_spread)

        mean_curve = np.nanmean(all_daily, axis=0)
        spread_curve = np.nanmean(all_spread, axis=0)
        lead_days = np.arange(1, len(mean_curve) + 1)
        # Limit to 42 days
        mask42 = lead_days <= 42
        lead_days = lead_days[mask42]
        mean_curve = mean_curve[mask42]
        spread_curve = spread_curve[mask42]

        c = rbox["color"]
        ax.fill_between(lead_days, mean_curve - spread_curve,
                        mean_curve + spread_curve,
                        alpha=0.15, color=c)
        ax.plot(lead_days, mean_curve, color=c, lw=2.5, zorder=5)

        # ERA5 regional clim
        era5_reg = era5_daily.sel(
            latitude=slice(lat_range[0], lat_range[1]),
            longitude=slice(lon_range[0], lon_range[1]))
        w_era5 = np.cos(np.deg2rad(era5_reg.latitude))
        w_era5 = w_era5 / w_era5.mean()
        era5_rm = era5_reg.weighted(xr.DataArray(w_era5, dims=["latitude"])).mean(
            dim=["latitude", "longitude"]).mean("time").values
        ax.axhline(float(era5_rm), color=c, ls="--", lw=1.5, alpha=0.6)
        ax.text(43, float(era5_rm), "ERA5", fontsize=8, color=c, va="center")

        # Week boundaries
        for wk, (s0, s1) in WEEK_DEFS.items():
            ax.axvline(s1, color="gray", ls=":", lw=0.4, alpha=0.4)

        ax.set_title(rname, fontsize=12, fontweight="bold", color=c)
        ax.set_xlim(1, 42)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Lead Day", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Precipitation (mm day⁻¹)", fontsize=11)

    fig.suptitle("Regional TP vs Lead Day (ECMWF S2S, JJAS 2000–2019)\n"
                 "Shading: ensemble spread (±1σ), Dashed: ERA5 climatology",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig10_regional_tp_vs_lead.png")


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="ECMWF S2S TP Exploration Figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()
    figs = set(args.fig) if args.fig else set(range(1, 11))

    print("═══════════════════════════════════════════════════════════════")
    print("  ECMWF S2S Hindcast TP Pattern Exploration")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  ECMWF root:  {ECMWF_ROOT}")
    print(f"  WB2 zarr:    {WB2_ZARR}")
    print(f"  JJAS inits:  {JJAS_INITS}")
    print(f"  India box:   {LAT_S}–{LAT_N}°N, {LON_W}–{LON_E}°E")
    print(f"  Output dir:  {OUT_DIR}")
    print(f"  Figures:     {sorted(figs)}")
    print("═══════════════════════════════════════════════════════════════\n")

    fig_funcs = {
        1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05,
        6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10,
    }

    for n in sorted(figs):
        if n in fig_funcs:
            try:
                fig_funcs[n]()
            except Exception as e:
                print(f"  ✗ Fig {n:02d} FAILED: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"  ⚠ Unknown figure number: {n}")

    # Summary
    print(f"\n═══ Done! Requested {len(figs)} figures ═══")
    print(f"Output directory: {OUT_DIR}")
    if os.path.isdir(OUT_DIR):
        for f in sorted(os.listdir(OUT_DIR)):
            fp = os.path.join(OUT_DIR, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp) / 1e6
                print(f"  {f}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
