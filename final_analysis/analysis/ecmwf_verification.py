#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecmwf_verification.py — ECMWF S2S Hindcast Verification Against ERA5
======================================================================
Comprehensive forecast verification analysis for Indian monsoon precipitation.
Generates 15 publication-quality figures comparing ECMWF S2S hindcasts (2000-2019)
against ERA5 reanalysis ground truth.

Figures:
  01. ACC vs Lead Week (The Money Plot)
  02. RMSE vs Lead Week
  03. Bias vs Lead Week (±1σ whiskers)
  04. Spatial ACC Maps (W1 and W3)
  05. Spatial RMSE Maps (W1 and W3)
  06. Reliability Diagram (Ensemble Calibration)
  07. Rank Histogram (Talagrand Diagram)
  08. Conditional Skill: Wet vs Dry Spells
  09. Forecast Busts: Worst Failures
  10. Scale-Dependent Skill (NOVEL)
  11. Skill Score Decomposition (Murphy 1988)
  12. Useful Skill Horizon by Region
  13. Extreme Event Detection: ROC Curves
  14. Spread-Skill Relationship
  15. Interannual Skill: Year-to-Year Variability

Usage:
    python ecmwf_verification.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# =========================================================================== #
#  Config
# =========================================================================== #
ECMWF_DIR = "/storage/raj.ayush/archive/All_Model_Data/models/ecmwf/data"
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

# India domain
LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

# JJAS init dates (biweekly subset for speed)
JJAS_INITS = ['0601', '0615', '0702', '0716', '0803', '0817']

# Hindcast years
YEARS = list(range(2000, 2020))
N_YEARS = len(YEARS)

# Weekly definitions: step indices for accumulated TP
# W1: steps 0-6 → lead days 1-7, accumulated at step 6
# W2: steps 7-13 → lead days 8-14, accumulated at step 13 minus step 6
# etc.
WEEK_DEFS = {
    'W1': (0, 6),
    'W2': (6, 13),
    'W3': (13, 20),
    'W4': (20, 27),
    'W5': (27, 34),
    'W6': (34, 41),
}

N_ENSEMBLE = 11  # 1 CF + 10 PF

IMD_REGIONS = {
    "Northwest India":  {"lat": (25.0, 36.0), "lon": (68.0, 80.0), "color": "#e41a1c"},
    "Central India":    {"lat": (20.0, 28.0), "lon": (74.0, 86.0), "color": "#377eb8"},
    "South Peninsula":  {"lat": ( 8.0, 20.0), "lon": (74.0, 82.0), "color": "#4daf4a"},
    "East & NE India":  {"lat": (20.0, 30.0), "lon": (86.0, 98.0), "color": "#ff7f00"},
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecmwf_verification")
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
def skill_cmap():
    """Blue-white-red for ACC/skill maps."""
    colors = [
        "#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#f7f7f7",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f",
    ]
    return mcolors.LinearSegmentedColormap.from_list("skill", colors, N=256)


def error_cmap():
    """Sequential for RMSE maps."""
    colors = [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b",
    ]
    return mcolors.LinearSegmentedColormap.from_list("error", colors, N=256)


# =========================================================================== #
#  Helpers
# =========================================================================== #
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


def india_extent(ax):
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=ccrs.PlateCarree())


def panel_label(ax, label, x=0.02, y=0.95):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def cos_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def area_mean_np(data, lat):
    """Area-weighted mean over lat, lon dimensions. data shape: (..., nlat, nlon)."""
    w = cos_weights(lat)
    # broadcast weights
    w_bc = w.reshape((1,) * (data.ndim - 2) + (len(lat), 1))
    weighted = data * w_bc
    return np.nanmean(weighted, axis=(-2, -1))


def region_mask(lats, lons, lat_range, lon_range):
    """Boolean mask for a lat/lon box."""
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    return lat_mask, lon_mask


def region_mean_np(data, lats, lons, lat_range, lon_range):
    """Area-weighted mean for a region. data shape: (..., nlat, nlon)."""
    lat_m, lon_m = region_mask(lats, lons, lat_range, lon_range)
    sub = data[..., lat_m, :]
    sub = sub[..., :, lon_m]
    sub_lats = lats[lat_m]
    return area_mean_np(sub, sub_lats)


def smooth_field(data, sigma=0.8):
    return gaussian_filter(np.nan_to_num(data, nan=0.0), sigma=sigma)


# =========================================================================== #
#  Data Loading
# =========================================================================== #
def load_ecmwf_grib(mmdd, member_type='cf'):
    """Load ECMWF GRIB for a given init date."""
    fname = f"tp_{member_type}_{mmdd}.grib"
    path = os.path.join(ECMWF_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
    tp = ds['tp']
    return tp.load()


def load_ecmwf_ensemble(mmdd):
    """Load CF + PF into ensemble array.

    Returns:
        ens_weekly: dict of week_name -> np.ndarray of shape (n_members, n_years, nlat, nlon)
                    containing weekly-mean TP rate in mm/day.
        lats, lons: coordinate arrays (India subset)
    """
    print(f"  Loading ECMWF init {mmdd}...")
    cf = load_ecmwf_grib(mmdd, 'cf')
    pf = load_ecmwf_grib(mmdd, 'pf')

    if cf is None:
        return None, None, None

    # Crop to India: lat descending slice(38, 5), lon slice(65, 100)
    cf_india = cf.sel(latitude=slice(38, 5), longitude=slice(65, 100))
    lats = cf_india.latitude.values
    lons = cf_india.longitude.values

    # CF shape: (time=20, step=46, lat, lon)
    cf_vals = cf_india.values  # (20, 46, nlat, nlon) — already accumulated TP in kg/m²=mm

    # PF: (number=10, time=20, step=46, lat, lon)
    if pf is not None:
        pf_india = pf.sel(latitude=slice(38, 5), longitude=slice(65, 100))
        pf_vals = pf_india.values  # (10, 20, 46, nlat, nlon)
    else:
        # fallback: use CF replicated
        pf_vals = np.repeat(cf_vals[np.newaxis, ...], 10, axis=0)

    nlat, nlon = len(lats), len(lons)
    n_years = cf_vals.shape[0]

    # Compute weekly means for each week
    ens_weekly = {}
    for wname, (s_start, s_end) in WEEK_DEFS.items():
        if s_end >= cf_vals.shape[1]:
            continue

        # CF weekly rate (mm/day)
        if s_start == 0:
            cf_week = cf_vals[:, s_end, :, :] / 7.0  # (n_years, nlat, nlon)
        else:
            cf_week = (cf_vals[:, s_end, :, :] - cf_vals[:, s_start, :, :]) / 7.0

        # PF weekly rate
        if s_start == 0:
            pf_week = pf_vals[:, :, s_end, :, :] / 7.0  # (10, n_years, nlat, nlon)
        else:
            pf_week = (pf_vals[:, :, s_end, :, :] - pf_vals[:, :, s_start, :, :]) / 7.0

        # Combine: member 0 = CF, members 1-10 = PF
        ens = np.zeros((N_ENSEMBLE, n_years, nlat, nlon))
        ens[0] = cf_week
        ens[1:] = pf_week

        ens_weekly[wname] = ens

    return ens_weekly, lats, lons


def load_era5_weekly(mmdd, lats_target, lons_target):
    """Load ERA5 weekly means matching ECMWF init dates.

    Returns:
        era5_weekly: dict of week_name -> np.ndarray of shape (n_years, nlat, nlon)
                     containing weekly-mean TP in mm/day.
    """
    print(f"  Loading ERA5 for init {mmdd}...")
    ds = xr.open_zarr(WB2_ZARR)
    tp24 = ds["total_precipitation_24hr"]
    # Select 06 UTC
    tp24 = tp24.sel(time=tp24["time.hour"] == 6)
    # Select India (ERA5 lat ascending)
    tp24 = tp24.sel(latitude=slice(LAT_S - 2, LAT_N + 2),
                    longitude=slice(LON_W - 2, LON_E + 2))

    mm_int = int(mmdd[:2])
    dd_int = int(mmdd[2:])

    era5_weekly = {}
    for wname, (s_start, s_end) in WEEK_DEFS.items():
        week_data = np.zeros((N_YEARS, len(lats_target), len(lons_target)))

        for yi, year in enumerate(YEARS):
            try:
                init_date = pd.Timestamp(year, mm_int, dd_int)
                # Valid dates for this week
                d0 = init_date + pd.Timedelta(days=s_start + 1)  # step s_start+1 = lead day
                d1 = init_date + pd.Timedelta(days=s_end)        # step s_end = lead day
                # Select ERA5 daily for this week
                era5_sub = tp24.sel(time=slice(str(d0), str(d1)))
                if len(era5_sub.time) == 0:
                    week_data[yi] = np.nan
                    continue
                era5_mean = era5_sub.mean("time") * 1000.0  # m -> mm/day

                # Interpolate to ECMWF grid
                era5_interp = era5_mean.interp(
                    latitude=xr.DataArray(lats_target, dims="latitude"),
                    longitude=xr.DataArray(lons_target, dims="longitude"),
                    method="nearest"
                )
                week_data[yi] = era5_interp.values
            except Exception as e:
                week_data[yi] = np.nan

        era5_weekly[wname] = week_data

    return era5_weekly


# =========================================================================== #
#  Core Verification Computation
# =========================================================================== #
def compute_all_data():
    """Load all ECMWF and ERA5 data, compute verification arrays.

    Returns a dict with all the data needed for all 15 figures.
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    all_ens = {}    # init -> week -> (n_members, n_years, nlat, nlon)
    all_era5 = {}   # init -> week -> (n_years, nlat, nlon)
    lats = None
    lons = None

    for mmdd in JJAS_INITS:
        ens_weekly, lat_i, lon_i = load_ecmwf_ensemble(mmdd)
        if ens_weekly is None:
            continue
        if lats is None:
            lats = lat_i
            lons = lon_i

        era5_weekly = load_era5_weekly(mmdd, lats, lons)
        all_ens[mmdd] = ens_weekly
        all_era5[mmdd] = era5_weekly

    print(f"\nLoaded {len(all_ens)} init dates.")
    if lats is not None:
        print(f"Grid: {len(lats)} lats × {len(lons)} lons")
        print(f"Lat range: {lats.min():.1f} – {lats.max():.1f}")
        print(f"Lon range: {lons.min():.1f} – {lons.max():.1f}")

    return {
        'all_ens': all_ens,
        'all_era5': all_era5,
        'lats': lats,
        'lons': lons,
    }


def compute_acc(fcst_anom, obs_anom, axis=0):
    """Anomaly correlation coefficient along an axis."""
    f = fcst_anom - np.nanmean(fcst_anom, axis=axis, keepdims=True)
    o = obs_anom - np.nanmean(obs_anom, axis=axis, keepdims=True)
    num = np.nansum(f * o, axis=axis)
    den = np.sqrt(np.nansum(f**2, axis=axis) * np.nansum(o**2, axis=axis))
    den = np.where(den == 0, np.nan, den)
    return num / den


def compute_rmse(fcst, obs, axis=0):
    """Root mean square error along an axis."""
    return np.sqrt(np.nanmean((fcst - obs)**2, axis=axis))


# =========================================================================== #
#  FIGURE 1: ACC vs Lead Week
# =========================================================================== #
def fig01(data):
    print("\nFig 01: ACC vs Lead Week (The Money Plot)...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())

    # Compute ACC for All-India and 4 regions
    regions = {"All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E), "color": "k"}}
    regions.update(IMD_REGIONS)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) All-India
    ax = axes[0]
    for rname, rbox in [("All India", regions["All India"])]:
        acc_by_week = []
        for wname in weeks:
            accs_per_init = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
                obs = all_era5[mmdd][wname]  # (20, nlat, nlon)
                # Ensemble mean
                fcst = np.nanmean(ens, axis=0)  # (20, nlat, nlon)
                # Area means
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                # Anomalies (remove mean over years)
                fcst_anom = fcst_am - np.nanmean(fcst_am)
                obs_anom = obs_am - np.nanmean(obs_am)
                acc = compute_acc(fcst_anom, obs_anom)
                if np.isfinite(acc):
                    accs_per_init.append(acc)
            acc_by_week.append(np.mean(accs_per_init) if accs_per_init else np.nan)

        ax.plot(range(1, len(weeks)+1), acc_by_week, 'ko-', lw=2.5, ms=8,
                label="All India", zorder=5)

    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.7, label="Useful skill (ACC=0.5)")
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("ACC")
    ax.set_title("(a) All-India ACC vs Lead Week", fontsize=12)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) All regions
    ax = axes[1]
    for rname, rbox in regions.items():
        acc_by_week = []
        for wname in weeks:
            accs_per_init = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                fcst_anom = fcst_am - np.nanmean(fcst_am)
                obs_anom = obs_am - np.nanmean(obs_am)
                acc = compute_acc(fcst_anom, obs_anom)
                if np.isfinite(acc):
                    accs_per_init.append(acc)
            acc_by_week.append(np.mean(accs_per_init) if accs_per_init else np.nan)

        ax.plot(range(1, len(weeks)+1), acc_by_week, 'o-', lw=2, ms=7,
                color=rbox["color"], label=rname)

    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.7, label="ACC=0.5")
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("ACC")
    ax.set_title("(b) ACC by IMD Region", fontsize=12)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("ECMWF S2S Weekly Precipitation — Anomaly Correlation Coefficient\n"
                 f"(JJAS Hindcast 2000–2019, {len(all_ens)} init dates, verified against ERA5)",
                 fontsize=13, fontweight="bold", y=1.04)
    savefig(fig, "fig01_acc_vs_lead_week.png")


# =========================================================================== #
#  FIGURE 2: RMSE vs Lead Week
# =========================================================================== #
def fig02(data):
    print("\nFig 02: RMSE vs Lead Week...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())
    regions = {"All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E), "color": "k"}}
    regions.update(IMD_REGIONS)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) RMSE
    ax = axes[0]
    for rname, rbox in regions.items():
        rmse_by_week = []
        clim_rmse_by_week = []
        for wname in weeks:
            errors = []
            clim_errors = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                clim = np.nanmean(obs_am)  # climatology = 20-yr mean
                errors.extend((fcst_am - obs_am).tolist())
                clim_errors.extend((clim - obs_am).tolist())
            rmse_by_week.append(np.sqrt(np.nanmean(np.array(errors)**2)))
            clim_rmse_by_week.append(np.sqrt(np.nanmean(np.array(clim_errors)**2)))

        ax.plot(range(1, len(weeks)+1), rmse_by_week, 'o-', lw=2, ms=7,
                color=rbox["color"], label=rname)
        if rname == "All India":
            ax.plot(range(1, len(weeks)+1), clim_rmse_by_week, 's--', lw=1.5,
                    ms=5, color='gray', alpha=0.7, label="Climatology RMSE")

    ax.set_xlabel("Lead Week")
    ax.set_ylabel("RMSE (mm/day)")
    ax.set_title("(a) RMSE vs Lead Week", fontsize=12)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Skill Score
    ax = axes[1]
    for rname, rbox in regions.items():
        ss_by_week = []
        for wname in weeks:
            errors = []
            clim_errors = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                clim = np.nanmean(obs_am)
                errors.extend((fcst_am - obs_am).tolist())
                clim_errors.extend((clim - obs_am).tolist())
            rmse_f = np.sqrt(np.nanmean(np.array(errors)**2))
            rmse_c = np.sqrt(np.nanmean(np.array(clim_errors)**2))
            ss = 1.0 - rmse_f / rmse_c if rmse_c > 0 else 0.0
            ss_by_week.append(ss)

        ax.plot(range(1, len(weeks)+1), ss_by_week, 'o-', lw=2, ms=7,
                color=rbox["color"], label=rname)

    ax.axhline(0, color='gray', ls='-', lw=0.8)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("Skill Score (1 - RMSE/RMSE_clim)")
    ax.set_title("(b) RMSE Skill Score", fontsize=12)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("ECMWF S2S Weekly Precipitation — RMSE & Skill Score\n"
                 "(JJAS 2000–2019, verified against ERA5)",
                 fontsize=13, fontweight="bold", y=1.04)
    savefig(fig, "fig02_rmse_vs_lead_week.png")


# =========================================================================== #
#  FIGURE 3: Bias vs Lead Week
# =========================================================================== #
def fig03(data):
    print("\nFig 03: Bias vs Lead Week...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())
    regions = {"All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E), "color": "k"}}
    regions.update(IMD_REGIONS)

    fig, ax = plt.subplots(figsize=(10, 6))

    for rname, rbox in regions.items():
        bias_mean = []
        bias_std = []
        for wname in weeks:
            biases = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                biases.extend((fcst_am - obs_am).tolist())
            bias_arr = np.array(biases)
            bias_mean.append(np.nanmean(bias_arr))
            bias_std.append(np.nanstd(bias_arr))

        x = np.arange(1, len(weeks)+1)
        bm = np.array(bias_mean)
        bs = np.array(bias_std)
        ax.errorbar(x, bm, yerr=bs, fmt='o-', lw=2, ms=7, capsize=4,
                    color=rbox["color"], label=rname)

    ax.axhline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("Bias (mm/day)")
    ax.set_title("ECMWF S2S Bias (ECMWF − ERA5) vs Lead Week\n"
                 "(JJAS 2000–2019, ±1σ whiskers)", fontsize=13)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig03_bias_vs_lead_week.png")


# =========================================================================== #
#  FIGURE 4: Spatial ACC Maps (W1 and W3)
# =========================================================================== #
def fig04(data):
    print("\nFig 04: Spatial ACC Maps...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})

    for idx, (wname, lbl) in enumerate([('W1', '(a) Week 1'), ('W3', '(b) Week 3')]):
        ax = axes[idx]
        india_extent(ax)
        add_map_features(ax, rivers=True)

        # Stack forecasts and obs over all init dates
        fcst_stack = []
        obs_stack = []
        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
            obs = all_era5[mmdd][wname]  # (20, nlat, nlon)
            fcst = np.nanmean(ens, axis=0)  # (20, nlat, nlon)
            fcst_stack.append(fcst)
            obs_stack.append(obs)

        if not fcst_stack:
            continue

        # Per-gridpoint ACC across years (for each init, then average)
        nlat, nlon = len(lats), len(lons)
        acc_map = np.full((nlat, nlon), np.nan)

        for i in range(nlat):
            for j in range(nlon):
                accs = []
                for fi, oi in zip(fcst_stack, obs_stack):
                    f_ts = fi[:, i, j]
                    o_ts = oi[:, i, j]
                    mask = np.isfinite(f_ts) & np.isfinite(o_ts)
                    if mask.sum() >= 5:
                        fa = f_ts[mask] - np.mean(f_ts[mask])
                        oa = o_ts[mask] - np.mean(o_ts[mask])
                        r = compute_acc(fa, oa)
                        if np.isfinite(r):
                            accs.append(r)
                if accs:
                    acc_map[i, j] = np.mean(accs)

        sm = smooth_field(acc_map, sigma=0.5)
        levels = np.arange(-0.4, 1.01, 0.1)
        cmap = skill_cmap()
        cf = ax.contourf(lons, lats, sm, levels=levels, cmap=cmap,
                         extend="both", transform=proj)
        cs = ax.contour(lons, lats, sm, levels=[0.5], colors="k",
                        linewidths=1.2, transform=proj)
        ax.clabel(cs, fmt="%.1f", fontsize=8)
        ax.set_title(lbl, fontsize=12, pad=8)

        # IMD boxes
        for rn, rb in IMD_REGIONS.items():
            lat0, lat1 = rb["lat"]
            lon0, lon1 = rb["lon"]
            ax.plot([lon0, lon1, lon1, lon0, lon0],
                    [lat0, lat0, lat1, lat1, lat0],
                    color=rb["color"], lw=1.2, ls="--", alpha=0.8, transform=proj)

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("Anomaly Correlation Coefficient", fontsize=10)

    fig.suptitle("ECMWF S2S Spatial ACC — Weekly Precipitation\n"
                 "(JJAS 2000–2019, verified against ERA5)",
                 fontsize=13, fontweight="bold", y=1.02)
    savefig(fig, "fig04_spatial_acc_maps.png")


# =========================================================================== #
#  FIGURE 5: Spatial RMSE Maps (W1 and W3)
# =========================================================================== #
def fig05(data):
    print("\nFig 05: Spatial RMSE Maps...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": proj})

    for idx, (wname, lbl) in enumerate([('W1', '(a) Week 1 RMSE'), ('W3', '(b) Week 3 RMSE')]):
        ax = axes[idx]
        india_extent(ax)
        add_map_features(ax, rivers=True)

        nlat, nlon = len(lats), len(lons)
        rmse_map = np.full((nlat, nlon), np.nan)

        for i in range(nlat):
            for j in range(nlon):
                errors = []
                for mmdd in all_ens:
                    if wname not in all_ens[mmdd]:
                        continue
                    ens = all_ens[mmdd][wname]
                    obs = all_era5[mmdd][wname]
                    fcst = np.nanmean(ens, axis=0)
                    f_ts = fcst[:, i, j]
                    o_ts = obs[:, i, j]
                    mask = np.isfinite(f_ts) & np.isfinite(o_ts)
                    errors.extend((f_ts[mask] - o_ts[mask]).tolist())
                if errors:
                    rmse_map[i, j] = np.sqrt(np.mean(np.array(errors)**2))

        sm = smooth_field(rmse_map, sigma=0.5)
        levels = np.arange(0, 12.1, 1.0)
        cf = ax.contourf(lons, lats, sm, levels=levels, cmap=error_cmap(),
                         extend="max", transform=proj)
        cs = ax.contour(lons, lats, sm, levels=[3, 6, 9], colors="k",
                        linewidths=0.5, alpha=0.5, transform=proj)
        ax.clabel(cs, fmt="%.0f", fontsize=8)
        ax.set_title(lbl, fontsize=12, pad=8)

        for rn, rb in IMD_REGIONS.items():
            lat0, lat1 = rb["lat"]
            lon0, lon1 = rb["lon"]
            ax.plot([lon0, lon1, lon1, lon0, lon0],
                    [lat0, lat0, lat1, lat1, lat0],
                    color=rb["color"], lw=1.2, ls="--", alpha=0.8, transform=proj)

    cax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cb.set_label("RMSE (mm/day)", fontsize=10)

    fig.suptitle("ECMWF S2S Spatial RMSE — Weekly Precipitation\n"
                 "(JJAS 2000–2019, verified against ERA5)",
                 fontsize=13, fontweight="bold", y=1.02)
    savefig(fig, "fig05_spatial_rmse_maps.png")


# =========================================================================== #
#  FIGURE 6: Reliability Diagram
# =========================================================================== #
def fig06(data):
    print("\nFig 06: Reliability Diagram...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for idx, (wname, lbl) in enumerate([('W1', '(a) Week 1'), ('W3', '(b) Week 3')]):
        ax = axes[idx]

        # Collect all gridpoint forecasts and obs
        all_fcst_prob = []
        all_obs_binary = []

        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
            obs = all_era5[mmdd][wname]  # (20, nlat, nlon)

            # Compute median of ERA5 over years for this init (climatological median)
            obs_median = np.nanmedian(obs, axis=0)  # (nlat, nlon)

            # For each year: is obs above median? (binary event)
            obs_above = (obs > obs_median[np.newaxis, :, :]).astype(float)

            # Forecast probability: fraction of members above median
            ens_above = (ens > obs_median[np.newaxis, np.newaxis, :, :]).astype(float)
            fcst_prob = np.nanmean(ens_above, axis=0)  # (20, nlat, nlon)

            # Use All-India area mean
            fcst_am = area_mean_np(fcst_prob, lats)
            obs_am = area_mean_np(obs_above, lats)

            all_fcst_prob.extend(fcst_am.flatten().tolist())
            all_obs_binary.extend(obs_am.flatten().tolist())

        fp = np.array(all_fcst_prob)
        ob = np.array(all_obs_binary)
        mask = np.isfinite(fp) & np.isfinite(ob)
        fp = fp[mask]
        ob = ob[mask]

        # Bin into probability categories
        bins = np.arange(0, 1.1, 0.1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        obs_freq = []
        counts = []
        for i in range(len(bins)-1):
            sel = (fp >= bins[i]) & (fp < bins[i+1])
            if i == len(bins)-2:  # include right edge
                sel = (fp >= bins[i]) & (fp <= bins[i+1])
            if sel.sum() > 0:
                obs_freq.append(np.mean(ob[sel]))
                counts.append(sel.sum())
            else:
                obs_freq.append(np.nan)
                counts.append(0)

        obs_freq = np.array(obs_freq)
        counts = np.array(counts)

        # Plot
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="Perfect reliability")
        ax.plot(bin_centers, obs_freq, 'o-', color="#2171b5", lw=2, ms=8,
                label="ECMWF", zorder=5)

        # Sharpness histogram inset
        ax_in = ax.inset_axes([0.55, 0.05, 0.4, 0.3])
        ax_in.bar(bin_centers, counts, width=0.08, color="#2171b5", alpha=0.6,
                  edgecolor="white")
        ax_in.set_xlabel("Forecast prob.", fontsize=7)
        ax_in.set_ylabel("Count", fontsize=7)
        ax_in.tick_params(labelsize=6)
        ax_in.set_title("Sharpness", fontsize=7)

        ax.set_xlabel("Forecast Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{lbl}: Reliability Diagram", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)

    fig.suptitle("ECMWF Ensemble Reliability: Weekly TP Above Median\n"
                 "(JJAS 2000–2019, All-India)",
                 fontsize=13, fontweight="bold", y=1.03)
    savefig(fig, "fig06_reliability_diagram.png")


# =========================================================================== #
#  FIGURE 7: Rank Histogram
# =========================================================================== #
def fig07(data):
    print("\nFig 07: Rank Histogram...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, (wname, lbl) in enumerate([('W1', '(a) Week 1'), ('W3', '(b) Week 3')]):
        ax = axes[idx]
        ranks = []

        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
            obs = all_era5[mmdd][wname]  # (20, nlat, nlon)

            # All-India area means
            ens_am = np.array([area_mean_np(ens[m], lats) for m in range(N_ENSEMBLE)])  # (11, 20)
            obs_am = area_mean_np(obs, lats)  # (20,)

            for yr in range(N_YEARS):
                ens_vals = ens_am[:, yr]
                obs_val = obs_am[yr]
                if np.isfinite(obs_val) and np.all(np.isfinite(ens_vals)):
                    combined = np.append(ens_vals, obs_val)
                    rank = np.searchsorted(np.sort(combined), obs_val)
                    ranks.append(rank + 1)  # 1-indexed

        ranks = np.array(ranks)
        n_ranks = N_ENSEMBLE + 1  # 12

        ax.hist(ranks, bins=np.arange(0.5, n_ranks + 1.5, 1), density=True,
                color="#4292c6", edgecolor="white", lw=0.8, alpha=0.85)
        ax.axhline(1.0 / n_ranks, color='red', ls='--', lw=1.5, alpha=0.7,
                   label=f"Uniform ({1/n_ranks:.3f})")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Relative Frequency")
        ax.set_title(f"{lbl}: Rank Histogram (Talagrand)", fontsize=12)
        ax.set_xticks(range(1, n_ranks + 1))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5, axis='y')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate interpretation
        shape = "U-shape" if ranks.std() > 2 else "Flat"
        ax.text(0.02, 0.95, f"n={len(ranks)}", transform=ax.transAxes,
                fontsize=9, va="top")

    fig.suptitle("ECMWF Ensemble Rank Histogram — All-India Weekly TP\n"
                 "(JJAS 2000–2019, 11-member ensemble)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig07_rank_histogram.png")


# =========================================================================== #
#  FIGURE 8: Conditional Skill — Wet vs Dry
# =========================================================================== #
def fig08(data):
    print("\nFig 08: Conditional Skill (Wet vs Dry)...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())
    rbox = {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}

    # First, collect all All-India obs values to compute percentiles
    all_obs_vals = []
    for mmdd in all_era5:
        for wname in all_era5[mmdd]:
            obs = all_era5[mmdd][wname]
            obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
            all_obs_vals.extend(obs_am.tolist())
    all_obs_arr = np.array(all_obs_vals)
    p75 = np.nanpercentile(all_obs_arr, 75)
    p25 = np.nanpercentile(all_obs_arr, 25)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, threshold_fn, color, marker in [
        ("Wet (ERA5 > 75th pctl)", lambda x: x > p75, "#2171b5", "o"),
        ("Dry (ERA5 < 25th pctl)", lambda x: x < p25, "#d6604d", "s"),
        ("All conditions", lambda x: np.ones_like(x, dtype=bool), "k", "^"),
    ]:
        acc_by_week = []
        for wname in weeks:
            accs = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])

                # Filter by condition
                cond = threshold_fn(obs_am)
                if cond.sum() >= 5:
                    fa = fcst_am[cond] - np.nanmean(fcst_am[cond])
                    oa = obs_am[cond] - np.nanmean(obs_am[cond])
                    r = compute_acc(fa, oa)
                    if np.isfinite(r):
                        accs.append(r)
            acc_by_week.append(np.mean(accs) if accs else np.nan)

        ax.plot(range(1, len(weeks)+1), acc_by_week, f'{marker}-', lw=2.5, ms=9,
                color=color, label=condition, zorder=5)

    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5, label="ACC=0.5")
    ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("ACC")
    ax.set_title("Conditional Forecast Skill: Wet vs Dry Conditions\n"
                 "(All-India, JJAS 2000–2019)", fontsize=13)
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.set_ylim(-0.3, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig08_conditional_skill.png")


# =========================================================================== #
#  FIGURE 9: Forecast Busts
# =========================================================================== #
def fig09(data):
    print("\nFig 09: Forecast Busts...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    wname = 'W1'
    rbox = {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}

    # Collect all W1 forecasts with metadata
    records = []
    spatial_errors = []

    for mmdd in all_ens:
        if wname not in all_ens[mmdd]:
            continue
        ens = all_ens[mmdd][wname]
        obs = all_era5[mmdd][wname]
        fcst = np.nanmean(ens, axis=0)  # (20, nlat, nlon)
        fcst_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
        obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])

        for yi in range(N_YEARS):
            if np.isfinite(fcst_am[yi]) and np.isfinite(obs_am[yi]):
                err = fcst_am[yi] - obs_am[yi]
                records.append({
                    'year': YEARS[yi],
                    'init': mmdd,
                    'fcst': fcst_am[yi],
                    'obs': obs_am[yi],
                    'error': err,
                    'abs_error': abs(err),
                    'spatial_error': fcst[yi] - obs[yi],
                })

    records.sort(key=lambda x: x['abs_error'], reverse=True)
    top5 = records[:5]

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

    # (a) Bar chart of top 5 busts
    ax = fig.add_subplot(gs[0, :])
    labels = [f"{r['year']}\n({r['init']})" for r in top5]
    x = np.arange(len(top5))
    w = 0.35
    ax.bar(x - w/2, [r['fcst'] for r in top5], w, label='ECMWF',
           color='#2171b5', edgecolor='white')
    ax.bar(x + w/2, [r['obs'] for r in top5], w, label='ERA5',
           color='#d6604d', edgecolor='white')
    for i, r in enumerate(top5):
        ax.annotate(f"err: {r['error']:+.1f}", (i, max(r['fcst'], r['obs'])),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha='center', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Weekly TP (mm/day)")
    ax.set_title("(a) Top 5 Worst W1 Forecast Busts — All-India", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Spatial pattern of worst bust
    proj = ccrs.PlateCarree()
    worst = top5[0]
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    india_extent(ax2)
    add_map_features(ax2, rivers=True)
    err_map = worst['spatial_error']
    sm = smooth_field(err_map, sigma=0.5)
    levels = np.arange(-10, 10.1, 1)
    cmap_div = plt.cm.RdBu_r
    cf = ax2.contourf(lons, lats, sm, levels=levels, cmap=cmap_div,
                      extend="both", transform=proj)
    cb = fig.colorbar(cf, ax=ax2, orientation="horizontal", shrink=0.85, pad=0.08)
    cb.set_label("Error (mm/day)", fontsize=9)
    ax2.set_title(f"(b) Worst Bust: {worst['year']} ({worst['init']})", fontsize=11)

    # (c) Spatial pattern of 2nd worst
    if len(top5) > 1:
        worst2 = top5[1]
        ax3 = fig.add_subplot(gs[1, 1], projection=proj)
        india_extent(ax3)
        add_map_features(ax3, rivers=True)
        err_map2 = worst2['spatial_error']
        sm2 = smooth_field(err_map2, sigma=0.5)
        cf2 = ax3.contourf(lons, lats, sm2, levels=levels, cmap=cmap_div,
                           extend="both", transform=proj)
        cb2 = fig.colorbar(cf2, ax=ax3, orientation="horizontal", shrink=0.85, pad=0.08)
        cb2.set_label("Error (mm/day)", fontsize=9)
        ax3.set_title(f"(c) 2nd Worst: {worst2['year']} ({worst2['init']})", fontsize=11)

    fig.suptitle("ECMWF S2S Forecast Busts — Week 1 All-India TP",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig09_forecast_busts.png")


# =========================================================================== #
#  FIGURE 10: Scale-Dependent Skill
# =========================================================================== #
def fig10(data):
    print("\nFig 10: Scale-Dependent Skill...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    nlat, nlon = len(lats), len(lons)

    # Spatial averaging scales
    scales = {
        '1.5° (1×1)': 1,
        '4.5° (3×3)': 3,
        '7.5° (5×5)': 5,
        'Regional': 'regional',
        'All India': 'all_india',
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors_w = {'W1': '#2171b5', 'W2': '#41ab5d', 'W3': '#fd8d3c'}

    for wname, wcol in colors_w.items():
        accs_by_scale = []
        for sname, sval in scales.items():
            accs = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)  # (20, nlat, nlon)

                if sval == 'all_india':
                    fa = area_mean_np(fcst, lats)
                    oa = area_mean_np(obs, lats)
                    fa_a = fa - np.nanmean(fa)
                    oa_a = oa - np.nanmean(oa)
                    r = compute_acc(fa_a, oa_a)
                    if np.isfinite(r):
                        accs.append(r)
                elif sval == 'regional':
                    for rn, rb in IMD_REGIONS.items():
                        fa = region_mean_np(fcst, lats, lons, rb["lat"], rb["lon"])
                        oa = region_mean_np(obs, lats, lons, rb["lat"], rb["lon"])
                        fa_a = fa - np.nanmean(fa)
                        oa_a = oa - np.nanmean(oa)
                        r = compute_acc(fa_a, oa_a)
                        if np.isfinite(r):
                            accs.append(r)
                else:
                    # Box averaging
                    k = sval
                    for i in range(0, nlat - k + 1, max(1, k//2)):
                        for j in range(0, nlon - k + 1, max(1, k//2)):
                            sub_f = fcst[:, i:i+k, j:j+k]
                            sub_o = obs[:, i:i+k, j:j+k]
                            sub_lats = lats[i:i+k]
                            fa = area_mean_np(sub_f, sub_lats)
                            oa = area_mean_np(sub_o, sub_lats)
                            fa_a = fa - np.nanmean(fa)
                            oa_a = oa - np.nanmean(oa)
                            r = compute_acc(fa_a, oa_a)
                            if np.isfinite(r):
                                accs.append(r)

            accs_by_scale.append(np.mean(accs) if accs else np.nan)

        ax.plot(range(len(scales)), accs_by_scale, 'o-', lw=2.5, ms=9,
                color=wcol, label=wname, zorder=5)

    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(list(scales.keys()), fontsize=10)
    ax.set_ylabel("ACC")
    ax.set_xlabel("Spatial Averaging Scale")
    ax.set_title("Scale-Dependent Skill: ACC vs Spatial Averaging\n"
                 "(JJAS 2000–2019)", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylim(-0.1, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("Skill increases\nwith spatial scale →", xy=(3, 0.85),
                fontsize=10, fontstyle='italic', color='#666666',
                ha='center')
    savefig(fig, "fig10_scale_dependent_skill.png")


# =========================================================================== #
#  FIGURE 11: MSE Decomposition (Murphy 1988)
# =========================================================================== #
def fig11(data):
    print("\nFig 11: MSE Decomposition...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())
    rbox = {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}

    bias_sq = []
    var_term = []
    corr_term = []

    for wname in weeks:
        fcsts_all = []
        obs_all = []
        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]
            obs = all_era5[mmdd][wname]
            fcst = np.nanmean(ens, axis=0)
            f_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
            o_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
            fcsts_all.extend(f_am.tolist())
            obs_all.extend(o_am.tolist())

        f_arr = np.array(fcsts_all)
        o_arr = np.array(obs_all)
        mask = np.isfinite(f_arr) & np.isfinite(o_arr)
        f_arr = f_arr[mask]
        o_arr = o_arr[mask]

        f_mean = np.mean(f_arr)
        o_mean = np.mean(o_arr)
        f_std = np.std(f_arr)
        o_std = np.std(o_arr)

        if o_std > 0 and f_std > 0:
            r = np.corrcoef(f_arr, o_arr)[0, 1]
        else:
            r = 0

        # Murphy decomposition: MSE = (f_mean - o_mean)^2 + (f_std - o_std)^2 + 2*f_std*o_std*(1 - r)
        b2 = (f_mean - o_mean)**2
        v2 = (f_std - o_std)**2
        c2 = 2 * f_std * o_std * (1 - r)

        bias_sq.append(b2)
        var_term.append(v2)
        corr_term.append(c2)

    x = np.arange(len(weeks))
    w = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, bias_sq, w, label='Bias² (Mean error)', color='#d6604d', edgecolor='white')
    ax.bar(x, var_term, w, bottom=bias_sq, label='Variance mismatch', color='#4393c3',
           edgecolor='white')
    bottom2 = np.array(bias_sq) + np.array(var_term)
    ax.bar(x, corr_term, w, bottom=bottom2, label='Correlation term (1−r)',
           color='#92c5de', edgecolor='white')

    total = np.array(bias_sq) + np.array(var_term) + np.array(corr_term)
    for i in range(len(weeks)):
        pct_b = bias_sq[i] / total[i] * 100 if total[i] > 0 else 0
        pct_v = var_term[i] / total[i] * 100 if total[i] > 0 else 0
        pct_c = corr_term[i] / total[i] * 100 if total[i] > 0 else 0
        ax.text(i, total[i] + 0.1, f"{pct_b:.0f}/{pct_v:.0f}/{pct_c:.0f}%",
                ha='center', fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(weeks)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("MSE component (mm²/day²)")
    ax.set_title("MSE Decomposition (Murphy 1988) — All-India Weekly TP\n"
                 "Bias²/Variance/Correlation components", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "fig11_mse_decomposition.png")


# =========================================================================== #
#  FIGURE 12: Useful Skill Horizon
# =========================================================================== #
def fig12(data):
    print("\nFig 12: Useful Skill Horizon by Region...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    weeks = list(WEEK_DEFS.keys())
    regions = {"All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E), "color": "k"}}
    regions.update(IMD_REGIONS)

    horizons = {}
    acc_profiles = {}

    for rname, rbox in regions.items():
        acc_by_week = []
        for wname in weeks:
            accs = []
            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]
                obs = all_era5[mmdd][wname]
                fcst = np.nanmean(ens, axis=0)
                f_am = region_mean_np(fcst, lats, lons, rbox["lat"], rbox["lon"])
                o_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
                fa = f_am - np.nanmean(f_am)
                oa = o_am - np.nanmean(o_am)
                r = compute_acc(fa, oa)
                if np.isfinite(r):
                    accs.append(r)
            acc_by_week.append(np.mean(accs) if accs else np.nan)

        acc_profiles[rname] = acc_by_week
        # Find horizon (last week with ACC >= 0.5)
        horizon = 0
        for wi, acc in enumerate(acc_by_week):
            if acc >= 0.5:
                horizon = wi + 1
            else:
                break
        horizons[rname] = horizon

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) Horizontal bar chart
    ax = axes[0]
    rnames = list(horizons.keys())
    hvals = [horizons[r] for r in rnames]
    colors_bar = [regions[r]["color"] for r in rnames]

    y_pos = np.arange(len(rnames))
    bars = ax.barh(y_pos, hvals, color=colors_bar, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rnames, fontsize=10)
    ax.set_xlabel("Useful Skill Horizon (weeks with ACC ≥ 0.5)")
    ax.set_title("(a) Useful Skill Horizon by Region", fontsize=12)
    ax.set_xlim(0, max(hvals) + 1.5)
    for i, v in enumerate(hvals):
        ax.text(v + 0.1, i, f"W{v}", fontsize=11, fontweight='bold',
                va='center', color=colors_bar[i])
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) ACC profiles
    ax = axes[1]
    for rname, rbox in regions.items():
        ax.plot(range(1, len(weeks)+1), acc_profiles[rname], 'o-', lw=2, ms=7,
                color=rbox["color"], label=rname)
    ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.7, label="ACC=0.5 threshold")
    ax.set_xticks(range(1, len(weeks)+1))
    ax.set_xticklabels(weeks)
    ax.set_xlabel("Lead Week")
    ax.set_ylabel("ACC")
    ax.set_title("(b) ACC Profiles by Region", fontsize=12)
    ax.set_ylim(-0.3, 1.0)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Useful Forecast Skill Horizon — ECMWF S2S Weekly TP\n"
                 "(JJAS 2000–2019)", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig12_skill_horizon.png")


# =========================================================================== #
#  FIGURE 13: ROC Curves for Extreme Events
# =========================================================================== #
def fig13(data):
    print("\nFig 13: ROC Curves...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    rbox = {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}

    # Compute 90th percentile from ERA5
    all_obs = []
    for mmdd in all_era5:
        for wname in all_era5[mmdd]:
            obs = all_era5[mmdd][wname]
            obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])
            all_obs.extend(obs_am.tolist())
    p90 = np.nanpercentile(all_obs, 90)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors_w = {'W1': '#2171b5', 'W2': '#41ab5d', 'W3': '#fd8d3c'}

    for wname, wcol in colors_w.items():
        fcst_probs = []
        obs_binary = []

        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]
            obs = all_era5[mmdd][wname]

            # Ensemble probability of exceeding p90
            ens_am = np.array([region_mean_np(ens[m], lats, lons, rbox["lat"], rbox["lon"])
                              for m in range(N_ENSEMBLE)])  # (11, 20)
            obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])  # (20,)

            for yr in range(N_YEARS):
                if np.isfinite(obs_am[yr]):
                    prob = np.mean(ens_am[:, yr] > p90)
                    fcst_probs.append(prob)
                    obs_binary.append(1.0 if obs_am[yr] > p90 else 0.0)

        fp_arr = np.array(fcst_probs)
        ob_arr = np.array(obs_binary)

        # Compute ROC curve
        thresholds = np.arange(0, 1.01, 1.0 / N_ENSEMBLE)
        tprs = []
        fprs = []
        for thresh in thresholds:
            pred = (fp_arr >= thresh).astype(float)
            tp = np.sum((pred == 1) & (ob_arr == 1))
            fp = np.sum((pred == 1) & (ob_arr == 0))
            fn = np.sum((pred == 0) & (ob_arr == 1))
            tn = np.sum((pred == 0) & (ob_arr == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)

        # Sort by FPR
        sorted_pairs = sorted(zip(fprs, tprs))
        fprs_s = [p[0] for p in sorted_pairs]
        tprs_s = [p[1] for p in sorted_pairs]

        # AUC (trapezoidal)
        auc = np.trapz(tprs_s, fprs_s)

        ax.plot(fprs_s, tprs_s, 'o-', lw=2.5, ms=6, color=wcol,
                label=f"{wname} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="No skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Hit Rate)")
    ax.set_title("ROC Curves: Detecting Extreme Wet Weeks (>90th pctl)\n"
                 "(All-India, JJAS 2000–2019)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, lw=0.5)
    savefig(fig, "fig13_roc_curves.png")


# =========================================================================== #
#  FIGURE 14: Spread-Skill Relationship
# =========================================================================== #
def fig14(data):
    print("\nFig 14: Spread-Skill Relationship...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    rbox = {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (wname, lbl) in enumerate([('W1', '(a) Week 1'), ('W3', '(b) Week 3')]):
        ax = axes[idx]
        spreads = []
        errors = []

        for mmdd in all_ens:
            if wname not in all_ens[mmdd]:
                continue
            ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
            obs = all_era5[mmdd][wname]  # (20, nlat, nlon)

            ens_am = np.array([region_mean_np(ens[m], lats, lons, rbox["lat"], rbox["lon"])
                              for m in range(N_ENSEMBLE)])  # (11, 20)
            obs_am = region_mean_np(obs, lats, lons, rbox["lat"], rbox["lon"])  # (20,)
            fcst_am = np.mean(ens_am, axis=0)

            for yr in range(N_YEARS):
                if np.isfinite(obs_am[yr]) and np.isfinite(fcst_am[yr]):
                    spread = np.std(ens_am[:, yr])
                    err = abs(fcst_am[yr] - obs_am[yr])
                    spreads.append(spread)
                    errors.append(err)

        spreads = np.array(spreads)
        errors = np.array(errors)

        ax.scatter(spreads, errors, alpha=0.4, s=30, color='#2171b5',
                   edgecolors='white', lw=0.3)

        # Fit line
        mask = np.isfinite(spreads) & np.isfinite(errors)
        if mask.sum() > 5:
            slope, intercept, r, p, _ = linregress(spreads[mask], errors[mask])
            x_fit = np.linspace(0, spreads[mask].max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
                    label=f"r={r:.2f}, p={p:.3f}")

        # Perfect spread-skill line
        max_val = max(spreads.max(), errors.max()) if len(spreads) > 0 else 5
        ax.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5,
                label="Spread = Error")

        ax.set_xlabel("Ensemble Spread (σ, mm/day)")
        ax.set_ylabel("|Forecast Error| (mm/day)")
        ax.set_title(f"{lbl}: Spread-Skill", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("ECMWF Ensemble Spread-Skill Relationship — All-India Weekly TP\n"
                 "(JJAS 2000–2019)", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig14_spread_skill.png")


# =========================================================================== #
#  FIGURE 15: Interannual Skill
# =========================================================================== #
def fig15(data):
    print("\nFig 15: Interannual Skill...")
    all_ens = data['all_ens']
    all_era5 = data['all_era5']
    lats = data['lats']
    lons = data['lons']

    regions_sub = {
        "All India": {"lat": (LAT_S, LAT_N), "lon": (LON_W, LON_E), "color": "k"},
        "Central India": IMD_REGIONS["Central India"],
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax_idx, (rname, rbox) in enumerate(regions_sub.items()):
        ax = axes[ax_idx]

        # For W1: compute JJAS-mean for each year
        wname = 'W1'
        year_fcst_mean = np.full(N_YEARS, np.nan)
        year_fcst_spread = np.full(N_YEARS, np.nan)
        year_obs_mean = np.full(N_YEARS, np.nan)

        for yi, year in enumerate(YEARS):
            fcst_vals = []
            ens_vals = []
            obs_vals = []

            for mmdd in all_ens:
                if wname not in all_ens[mmdd]:
                    continue
                ens = all_ens[mmdd][wname]  # (11, 20, nlat, nlon)
                obs = all_era5[mmdd][wname]

                fcst = np.nanmean(ens, axis=0)  # ensemble mean
                f_am = region_mean_np(fcst[yi:yi+1], lats, lons, rbox["lat"], rbox["lon"])
                o_am = region_mean_np(obs[yi:yi+1], lats, lons, rbox["lat"], rbox["lon"])

                # Spread
                ens_am = np.array([region_mean_np(ens[m, yi:yi+1], lats, lons,
                                                  rbox["lat"], rbox["lon"])
                                  for m in range(N_ENSEMBLE)])

                if np.isfinite(f_am[0]) and np.isfinite(o_am[0]):
                    fcst_vals.append(f_am[0])
                    obs_vals.append(o_am[0])
                    ens_vals.append(np.std(ens_am))

            if fcst_vals:
                year_fcst_mean[yi] = np.mean(fcst_vals)
                year_obs_mean[yi] = np.mean(obs_vals)
                year_fcst_spread[yi] = np.mean(ens_vals)

        # Plot
        x = np.array(YEARS)
        ax.bar(x, year_obs_mean, 0.6, color='#bdd7e7', edgecolor='#6baed6',
               label='ERA5', zorder=2)
        ax.errorbar(x, year_fcst_mean, yerr=year_fcst_spread,
                    fmt='o', ms=7, color='#d6604d', capsize=3, lw=1.5,
                    label='ECMWF W1 (±spread)', zorder=5)

        # Correlation
        mask = np.isfinite(year_fcst_mean) & np.isfinite(year_obs_mean)
        if mask.sum() >= 5:
            r = np.corrcoef(year_fcst_mean[mask], year_obs_mean[mask])[0, 1]
            rmse = np.sqrt(np.mean((year_fcst_mean[mask] - year_obs_mean[mask])**2))
            ax.text(0.02, 0.95, f"r = {r:.2f}, RMSE = {rmse:.2f} mm/day",
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='top', bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.9))

        lbl = "(a)" if ax_idx == 0 else "(b)"
        ax.set_title(f"{lbl} {rname}", fontsize=12)
        ax.set_ylabel("JJAS-mean W1 TP (mm/day)")
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Year")
    fig.suptitle("Interannual Variability: ECMWF W1 Forecast vs ERA5\n"
                 "(JJAS 2000–2019)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig15_interannual_skill.png")


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    print("=" * 70)
    print("ECMWF S2S VERIFICATION AGAINST ERA5")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Init dates: {JJAS_INITS}")
    print(f"Years: {YEARS[0]}-{YEARS[-1]} (n={N_YEARS})")
    print(f"Weeks: {list(WEEK_DEFS.keys())}")
    print()

    # Load all data
    data = compute_all_data()

    if data['lats'] is None:
        print("ERROR: No data loaded!")
        return

    # Generate all figures
    figures = [
        ("Fig 01", fig01),
        ("Fig 02", fig02),
        ("Fig 03", fig03),
        ("Fig 04", fig04),
        ("Fig 05", fig05),
        ("Fig 06", fig06),
        ("Fig 07", fig07),
        ("Fig 08", fig08),
        ("Fig 09", fig09),
        ("Fig 10", fig10),
        ("Fig 11", fig11),
        ("Fig 12", fig12),
        ("Fig 13", fig13),
        ("Fig 14", fig14),
        ("Fig 15", fig15),
    ]

    success = []
    failed = []

    for name, func in figures:
        try:
            func(data)
            success.append(name)
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Succeeded: {len(success)}/{len(figures)}")
    for s in success:
        print(f"  ✓ {s}")
    if failed:
        print(f"\nFailed: {len(failed)}/{len(figures)}")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")

    # List output files
    print(f"\nOutput files in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f}: {size:.0f} KB")


if __name__ == "__main__":
    main()
