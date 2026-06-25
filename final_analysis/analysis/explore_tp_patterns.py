#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_tp_patterns.py — ERA5 Precipitation Pattern Explorer over India
=======================================================================
Generates a comprehensive set of figures to understand TP spatial and
temporal patterns from the WeatherBench2 ERA5 zarr (1.5°, 1959–2023).

Figures produced (saved to ./tp_exploration/):
  1. JJAS & JFM mean climatology maps (side-by-side)
  2. Monthly climatology cycle (12 panels)
  3. Seasonal cycle — area-averaged All-India daily TP time series (clim)
  4. Interannual variability — JJAS-mean rainfall by year + trend
  5. Monsoon onset/withdrawal evolution (pentad-mean latitude-time Hovmöller)
  6. Wet/dry year composites — anomaly maps
  7. Extreme precipitation frequency — days > 10 mm/day climatology
  8. Regional seasonal cycle — 4 IMD homogeneous regions

Usage:
    python explore_tp_patterns.py            # all figures
    python explore_tp_patterns.py --fig 1 3  # only figures 1 and 3
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
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# --------------------------------------------------------------------------- #
# paths & constants
# --------------------------------------------------------------------------- #
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")

# India bounding box (generous, for maps)
LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0

# IMD homogeneous rainfall regions (approx lat/lon boxes)
IMD_REGIONS = {
    "Northwest India":     {"lat": (25.0, 36.0), "lon": (68.0, 80.0)},
    "Central India":       {"lat": (20.0, 28.0), "lon": (74.0, 86.0)},
    "South Peninsula":     {"lat": ( 8.0, 20.0), "lon": (74.0, 82.0)},
    "East & NE India":     {"lat": (20.0, 30.0), "lon": (86.0, 98.0)},
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp_exploration")
os.makedirs(OUT_DIR, exist_ok=True)

# Use a long climatology baseline (1991–2020 WMO standard)
CLIM_Y0, CLIM_Y1 = 1991, 2020


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_wb2_tp_india(y0=None, y1=None):
    """Open WB2 total_precipitation_24hr, crop to India, convert to mm/day.

    Uses the pre-aggregated 24hr field (one value per day) instead of
    resampling the 6-hourly data, which is ~4x faster on zarr.

    Returns an xarray DataArray (time, lat, lon) with units mm/day,
    latitude ascending, eagerly loaded into memory for fast operations.
    """
    print("  [1/3] Opening zarr...")
    ds = xr.open_zarr(WB2_ZARR)

    # Use 24hr total directly — much faster than resampling 6hr
    tp24 = ds["total_precipitation_24hr"]  # (time, lon, lat), units=m per 24h

    # crop to India box (lat ascending in WB2)
    tp24 = tp24.sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))

    # time slice
    t0 = f"{y0}-01-01" if y0 else None
    t1 = f"{y1}-12-31" if y1 else None
    if t0 or t1:
        tp24 = tp24.sel(time=slice(t0, t1))

    # Only keep one sample per day (the 24hr field is still on the 6-hourly
    # time axis — each day has 4 identical copies). Take 00:00 UTC timestamps.
    tp24 = tp24.sel(time=tp24["time.hour"] == 6)

    print(f"  [2/3] Loading into memory ({tp24.sizes})...")
    # convert m → mm and load eagerly (India subset is small: ~50 MB)
    tp_daily = (tp24 * 1000.0).load()  # mm/day, in memory

    # transpose to (time, lat, lon) for convenience
    tp_daily = tp_daily.transpose("time", "latitude", "longitude")
    print(f"  [3/3] Loaded! shape={tp_daily.shape}, mem≈{tp_daily.nbytes/1e6:.0f} MB")
    return tp_daily


def cos_weights(lat):
    """Cosine-latitude area weights, normalized to mean=1."""
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": lat})


def area_mean(da, lat_dim="latitude"):
    """Cosine-weighted area mean."""
    w = cos_weights(da[lat_dim])
    return da.weighted(w).mean(dim=[lat_dim, "longitude"])


def region_mean(da, lat_range, lon_range):
    """Extract a lat/lon box and compute area mean."""
    sub = da.sel(latitude=slice(lat_range[0], lat_range[1]),
                 longitude=slice(lon_range[0], lon_range[1]))
    return area_mean(sub)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {path}")


# Custom colormap for precipitation
def precip_cmap():
    """A rich blue-green-yellow-red precipitation colormap."""
    colors = [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
        "#6baed6", "#4292c6", "#2171b5",
        "#41ab5d", "#78c679", "#c2e699",
        "#fee08b", "#fdae61", "#f46d43",
        "#d73027", "#a50026", "#67001f",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip", colors, N=256)


def anomaly_cmap():
    """Brown-white-teal diverging colormap for precip anomalies."""
    colors = [
        "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5",
        "#c7eae5", "#80cdc1", "#35978f", "#01665e",
    ]
    return mcolors.LinearSegmentedColormap.from_list("precip_anom", colors, N=256)


# plt style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})


# =========================================================================== #
#  FIGURE 1: JJAS & JFM mean climatology maps
# =========================================================================== #
def fig1_seasonal_clim(tp):
    print("Fig 1: Seasonal climatology maps...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    jjas = clim.sel(time=clim["time.month"].isin([6, 7, 8, 9])).groupby("time.dayofyear").mean()
    jjas_mean = jjas.mean("dayofyear")

    jfm = clim.sel(time=clim["time.month"].isin([1, 2, 3])).groupby("time.dayofyear").mean()
    jfm_mean = jfm.mean("dayofyear")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             subplot_kw={"aspect": "auto"})
    cmap = precip_cmap()

    for ax, data, title, vmax in zip(
        axes,
        [jjas_mean, jfm_mean],
        ["JJAS (Jun–Sep) Mean Precip", "JFM (Jan–Mar) Mean Precip"],
        [18, 5],
    ):
        im = ax.pcolormesh(
            data.longitude, data.latitude, data.values,
            cmap=cmap, vmin=0, vmax=vmax, shading="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label("mm / day")

        # add region boxes
        for rname, rbox in IMD_REGIONS.items():
            lat0, lat1 = rbox["lat"]
            lon0, lon1 = rbox["lon"]
            ax.plot([lon0, lon1, lon1, lon0, lon0],
                    [lat0, lat0, lat1, lat1, lat0],
                    color="k", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle(f"ERA5 Precipitation Climatology ({CLIM_Y0}–{CLIM_Y1})", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig01_seasonal_clim.png")


# =========================================================================== #
#  FIGURE 2: Monthly climatology cycle (12 panels)
# =========================================================================== #
def fig2_monthly_clim(tp):
    print("Fig 2: Monthly climatology panels...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    monthly = clim.groupby("time.month").mean("time")

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), subplot_kw={"aspect": "equal"})
    cmap = precip_cmap()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for i, ax in enumerate(axes.flat):
        m = i + 1
        data = monthly.sel(month=m)
        vmax = 18 if m in [6, 7, 8, 9] else 6
        im = ax.pcolormesh(
            data.longitude, data.latitude, data.values,
            cmap=cmap, vmin=0, vmax=vmax, shading="auto",
        )
        ax.set_title(month_names[i], fontsize=12, fontweight="bold")
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        ax.tick_params(labelsize=8)
        if i % 4 != 0:
            ax.set_yticklabels([])
        if i < 8:
            ax.set_xticklabels([])
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    fig.suptitle(f"ERA5 Monthly Mean Precipitation ({CLIM_Y0}–{CLIM_Y1})",
                 y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig02_monthly_clim.png")


# =========================================================================== #
#  FIGURE 3: Annual cycle — area-averaged daily TP
# =========================================================================== #
def fig3_seasonal_cycle(tp):
    print("Fig 3: All-India seasonal cycle...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))
    daily_ai = area_mean(clim)
    doy_clim = daily_ai.groupby("time.dayofyear")

    mean = doy_clim.mean()
    p10 = doy_clim.quantile(0.1)
    p90 = doy_clim.quantile(0.9)

    # smooth with 15-day rolling
    mean_s = mean.rolling(dayofyear=15, center=True, min_periods=1).mean()
    p10_s = p10.rolling(dayofyear=15, center=True, min_periods=1).mean()
    p90_s = p90.rolling(dayofyear=15, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    doy = mean_s.dayofyear.values
    ax.fill_between(doy, p10_s.values, p90_s.values,
                    alpha=0.25, color="#2171b5", label="10th–90th percentile")
    ax.plot(doy, mean_s.values, color="#2171b5", lw=2.5, label="Mean")

    # mark JJAS
    ax.axvspan(152, 273, alpha=0.08, color="green", label="JJAS window")
    # mark JFM
    ax.axvspan(1, 90, alpha=0.08, color="orange", label="JFM window")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Precipitation (mm/day)")
    ax.set_title(f"All-India Daily Precipitation — Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(1, 366)
    ax.set_ylim(bottom=0)

    # month ticks
    month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_lbl = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_doy)
    ax.set_xticklabels(month_lbl)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig03_seasonal_cycle.png")


# =========================================================================== #
#  FIGURE 4: Interannual JJAS rainfall variability
# =========================================================================== #
def fig4_interannual(tp):
    print("Fig 4: Interannual JJAS variability...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    yearly = area_mean(jjas).resample(time="YE").mean()

    years = yearly["time.year"].values
    vals = yearly.values

    # long-term mean
    lt_mean = float(np.nanmean(vals))
    # anomalies
    anom = vals - lt_mean

    fig, ax = plt.subplots(figsize=(15, 5))
    colors = ["#d73027" if a < 0 else "#2171b5" for a in anom]
    ax.bar(years, anom, color=colors, width=0.8, edgecolor="white", lw=0.3)
    ax.axhline(0, color="k", lw=0.8)

    # add trend
    mask = ~np.isnan(vals)
    z = np.polyfit(years[mask], vals[mask], 1)
    trend_line = np.polyval(z, years)
    ax.plot(years, trend_line - lt_mean, color="k", ls="--", lw=1.5,
            label=f"Trend: {z[0]*10:.2f} mm/day per decade")

    ax.set_xlabel("Year")
    ax.set_ylabel("JJAS Precipitation Anomaly (mm/day)")
    ax.set_title("All-India JJAS Mean Precipitation — Interannual Variability (ERA5)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    fig.tight_layout()
    savefig(fig, "fig04_interannual_jjas.png")


# =========================================================================== #
#  FIGURE 5: Hovmöller — monsoon onset/progression (lon-averaged lat vs time)
# =========================================================================== #
def fig5_hovmoller(tp):
    print("Fig 5: Hovmöller (monsoon onset/progression)...")
    # average over 2002–2019 JJAS (same as your study period)
    sub = tp.sel(time=slice("2002", "2019"))
    # keep only Apr–Oct to see onset and withdrawal
    sub = sub.sel(time=sub["time.month"].isin([4, 5, 6, 7, 8, 9, 10]))
    # average over Indian longitudes (70–90°E)
    lon_avg = sub.sel(longitude=slice(70, 90)).mean("longitude")
    # pentad mean (5-day)
    pentad = lon_avg.resample(time="5D").mean()
    # climatological pentad (group by day-of-year pentad index)
    pentad_doy = pentad.groupby("time.dayofyear").mean("time")

    fig, ax = plt.subplots(figsize=(12, 7))
    doys = pentad_doy.dayofyear.values
    lats = pentad_doy.latitude.values
    cmap = precip_cmap()

    im = ax.pcolormesh(doys, lats, pentad_doy.values.T,
                       cmap=cmap, vmin=0, vmax=14, shading="auto")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlabel("Day of Year")
    ax.set_title("Monsoon Progression — Pentad-Mean Precipitation (70–90°E avg, 2002–2019)")

    # month ticks
    month_doy = [91, 121, 152, 182, 213, 244, 274, 305]
    month_lbl = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    ax.set_xticks(month_doy)
    ax.set_xticklabels(month_lbl)
    ax.set_ylim(LAT_S, LAT_N)

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("mm / day")
    fig.tight_layout()
    savefig(fig, "fig05_hovmoller_onset.png")


# =========================================================================== #
#  FIGURE 6: Wet vs Dry year composites (anomaly maps)
# =========================================================================== #
def fig6_wet_dry_composites(tp):
    print("Fig 6: Wet/Dry year composites...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    yearly_map = jjas.groupby("time.year").mean("time")  # (year, lat, lon)
    yearly_ai = area_mean(jjas).resample(time="YE").mean()

    years = yearly_ai["time.year"].values
    vals = yearly_ai.values
    lt_mean = float(np.nanmean(vals))
    anom_ts = vals - lt_mean
    std_val = float(np.nanstd(anom_ts))

    # wet years: > +0.5σ, dry years: < -0.5σ
    wet_yrs = years[anom_ts > 0.5 * std_val]
    dry_yrs = years[anom_ts < -0.5 * std_val]

    clim_map = yearly_map.mean("year")
    wet_composite = yearly_map.sel(year=wet_yrs).mean("year") - clim_map
    dry_composite = yearly_map.sel(year=dry_yrs).mean("year") - clim_map

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"aspect": "equal"})
    cmap = anomaly_cmap()
    vmax = 4

    for ax, data, title, yrs in zip(
        axes,
        [wet_composite, dry_composite],
        [f"Wet Years (n={len(wet_yrs)})", f"Dry Years (n={len(dry_yrs)})"],
        [wet_yrs, dry_yrs],
    ):
        im = ax.pcolormesh(
            data.longitude, data.latitude, data.values,
            cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label("Anomaly (mm/day)")
        # annotate years
        yr_str = ", ".join(str(y) for y in sorted(yrs))
        ax.annotate(yr_str, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=7, color="gray", va="bottom")

    fig.suptitle("JJAS Precipitation Anomaly Composites (ERA5, > ±0.5σ)", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig06_wet_dry_composites.png")


# =========================================================================== #
#  FIGURE 7: Extreme precip frequency — days > 10 mm/day
# =========================================================================== #
def fig7_extreme_frequency(tp):
    print("Fig 7: Extreme precip frequency...")
    jjas = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    jjas = jjas.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    # fraction of JJAS days exceeding 10 mm/day
    heavy = (jjas > 10.0).groupby("time.year").mean("time")  # fraction per year
    heavy_clim = heavy.mean("year") * 100  # convert to percentage

    # also > 20 mm/day
    very_heavy = (jjas > 20.0).groupby("time.year").mean("time")
    vh_clim = very_heavy.mean("year") * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"aspect": "equal"})

    for ax, data, title, vmax, cmap_name in zip(
        axes,
        [heavy_clim, vh_clim],
        ["> 10 mm/day", "> 20 mm/day"],
        [40, 20],
        ["YlOrRd", "magma_r"],
    ):
        im = ax.pcolormesh(
            data.longitude, data.latitude, data.values,
            cmap=cmap_name, vmin=0, vmax=vmax, shading="auto",
        )
        ax.set_title(f"JJAS Days with {title}")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlim(LON_W, LON_E)
        ax.set_ylim(LAT_S, LAT_N)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label("% of JJAS days")

    fig.suptitle(f"Extreme Precipitation Frequency ({CLIM_Y0}–{CLIM_Y1})", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig07_extreme_frequency.png")


# =========================================================================== #
#  FIGURE 8: Regional seasonal cycles — 4 IMD regions
# =========================================================================== #
def fig8_regional_cycles(tp):
    print("Fig 8: Regional seasonal cycles...")
    clim = tp.sel(time=slice(f"{CLIM_Y0}", f"{CLIM_Y1}"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    colors = ["#d73027", "#2171b5", "#41ab5d", "#ff7f00"]

    for ax, (rname, rbox), color in zip(axes.flat, IMD_REGIONS.items(), colors):
        rmean = region_mean(clim, rbox["lat"], rbox["lon"])
        doy_clim = rmean.groupby("time.dayofyear")
        mean = doy_clim.mean()
        p25 = doy_clim.quantile(0.25)
        p75 = doy_clim.quantile(0.75)

        # smooth
        k = 15
        mean_s = mean.rolling(dayofyear=k, center=True, min_periods=1).mean()
        p25_s = p25.rolling(dayofyear=k, center=True, min_periods=1).mean()
        p75_s = p75.rolling(dayofyear=k, center=True, min_periods=1).mean()

        doy = mean_s.dayofyear.values
        ax.fill_between(doy, p25_s.values, p75_s.values, alpha=0.2, color=color)
        ax.plot(doy, mean_s.values, color=color, lw=2.5)
        ax.set_title(rname, fontsize=12, fontweight="bold")
        ax.set_xlim(1, 366)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

        # month ticks
        month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_lbl = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax.set_xticks(month_doy)
        ax.set_xticklabels(month_lbl)

    for ax in axes[-1]:
        ax.set_xlabel("Month")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precipitation (mm/day)")

    fig.suptitle(f"Regional Precipitation Seasonal Cycle ({CLIM_Y0}–{CLIM_Y1})",
                 y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig08_regional_cycles.png")


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(description="ERA5 TP Pattern Explorer")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()
    figs = set(args.fig) if args.fig else set(range(1, 9))

    print(f"Opening WB2 ERA5 zarr and building daily TP over India...")
    print(f"  zarr: {WB2_ZARR}")
    print(f"  India box: {LAT_S}–{LAT_N}°N, {LON_W}–{LON_E}°E")

    # Load data — we load ALL years lazily, each figure subsets what it needs.
    # Actual compute only happens inside each figure function via .load() / .values
    tp = load_wb2_tp_india()
    print(f"  Daily TP loaded: {tp.sizes}")
    print(f"  Time range: {str(tp.time.values[0])[:10]} → {str(tp.time.values[-1])[:10]}")
    print(f"  Output dir: {OUT_DIR}\n")

    fig_funcs = {
        1: fig1_seasonal_clim,
        2: fig2_monthly_clim,
        3: fig3_seasonal_cycle,
        4: fig4_interannual,
        5: fig5_hovmoller,
        6: fig6_wet_dry_composites,
        7: fig7_extreme_frequency,
        8: fig8_regional_cycles,
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

    print(f"\nDone! All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
