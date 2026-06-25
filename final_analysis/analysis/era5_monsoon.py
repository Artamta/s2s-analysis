#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis/era5_monsoon.py  —  ERA5 observational analysis of the Indian summer
                             monsoon (JJAS) and its long-term cycles.
================================================================================
A self-contained, PARALLELIZED observational study built straight from the
WeatherBench2 ERA5 zarr (1.5 deg, 6-hourly, 1959-2023). It characterises the
mean monsoon, its annual cycle, its interannual variability, the climate-change
trends, the intraseasonal (MISO) northward propagation, and the leading EOF.

This script only READS the final_analysis verification core (grid / regions /
metrics / plotting style); it modifies nothing under core/ or jjas/. The heavy
per-year seasonal reductions are run in parallel with a ProcessPoolExecutor:
each worker opens the zarr lazily and loads ONLY its own year's India-box slice,
so the work scales linearly to all 64 years (run overnight with no --quick).

Figures (PNGs into analysis/figs/era5_monsoon/)
-----------------------------------------------
  1. clim_maps_jjas        JJAS climatology maps: rainfall, z500, t2m over India.
  2. annual_cycle          12-month climatological all-India-land mean rain + t2m.
  3. interannual_timeseries JJAS all-India rain & t2m time series + trend + sigma.
  4. trend_maps            per-grid JJAS rainfall / t2m linear trend (per decade).
  5. miso_hovmoller        Jun1-Sep30 latitude-time Hovmoller of rain anomaly.
  6. interannual_std_map   per-grid std of JJAS-mean rainfall across years.
  7. eof1_rainfall         leading EOF of detrended JJAS rainfall + PC time series.

Usage
-----
  # fast sanity run (recent decade, a couple of minutes):
  python era5_monsoon.py --quick

  # full overnight run (every complete JJAS, 1959-2022):
  python era5_monsoon.py --years 1959-2022 --workers 16

Run inside the s2s-hind conda env (cartopy + shapely + pyproj + scipy).
================================================================================
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FA_ROOT = os.path.dirname(HERE)
sys.path.insert(0, FA_ROOT)                               # final_analysis/ on path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from core.config import GridSpec, Physics
from core.grid import build_grid_context, make_grid
from core.metrics import cos_latitude_weights
from core.regions import _LCC, _WGS84

# ---------------------------------------------------------------- constants ---
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
SOI_SHP = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
REGION_MASK_NC = "/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc"
G = Physics().G

# India map extent (lon_w, lon_e, lat_s, lat_n) for set_extent.
INDIA_EXTENT = (65.0, 100.0, 5.0, 38.0)
# Generous load box (n, s, w, e) -- a little padding for clean interpolation edges.
LOAD_BOX = dict(s=3.0, n=40.0, w=60.0, e=102.0)
# India longitude band for the MISO Hovmoller (averaged over these lons).
MISO_LON_BAND = (70.0, 90.0)

# Big, presentation-friendly fonts.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 18, "figure.titleweight": "bold",
})


# ===================================================== parallel per-year work ==
def _open_box():
    """Open the WB2 zarr lazily, pre-sliced to the India load box (cheap)."""
    ds = xr.open_zarr(WB2_ZARR)
    return ds.sel(latitude=slice(LOAD_BOX["s"], LOAD_BOX["n"]),
                  longitude=slice(LOAD_BOX["w"], LOAD_BOX["e"]))


def _to_latlon(da):
    """Rename WB2 latitude/longitude -> lat/lon and order dims (lat, lon[, time])."""
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    lead = [d for d in ("time", "lat", "lon") if d in da.dims]
    return da.transpose(*lead)


def year_jjas_fields(year):
    """WORKER: JJAS (Jun1-Sep30) seasonal-MEAN rainfall (mm/day) and t2m (K) for
       one year on the NATIVE WB2 India-box grid. Returns (year, tp2d, t2m2d) as
       plain numpy arrays + coords so the result pickles cheaply back to master."""
    ds = _open_box()
    sub = ds.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
    tp = (sub["total_precipitation_6hr"].resample(time="1D").sum() * 1000.0).mean("time")
    t2 = sub["2m_temperature"].resample(time="1D").mean().mean("time")
    tp = _to_latlon(tp).load()
    t2 = _to_latlon(t2).load()
    return (year, tp.values.astype("float32"), t2.values.astype("float32"),
            tp["lat"].values, tp["lon"].values)


def year_jjas_daily_rain(year, lon_band=MISO_LON_BAND):
    """WORKER (MISO): daily JJAS rainfall (mm/day) for one year, averaged over the
       India longitude band -> a (time, lat) Hovmoller slab. Returns numpy arrays."""
    ds = _open_box()
    sub = ds.sel(time=slice(f"{year}-06-01", f"{year}-09-30"),
                 longitude=slice(lon_band[0], lon_band[1]))
    tp = (sub["total_precipitation_6hr"].resample(time="1D").sum() * 1000.0)
    tp = _to_latlon(tp).mean("lon").load()                # (time, lat)
    return year, tp.values.astype("float32"), tp["time"].values, tp["lat"].values


def z500_clim_field(years):
    """JJAS-mean z500 (gpm) climatology over `years`, native India-box grid.
       Done serially (one streamed reduction) -- z500 is only needed once."""
    ds = _open_box()
    pieces = []
    for y in years:
        z = ds["geopotential"].sel(level=500,
                                   time=slice(f"{y}-06-01", f"{y}-09-30"))
        z = (z.resample(time="1D").mean() / G).mean("time")
        pieces.append(_to_latlon(z).load())
    return sum(pieces) / len(pieces)


# ============================================================ grid + masking ===
def build_context():
    """Grid context on the 1.5 deg verification grid (land mask + IMD all-India
       mask + cosine weights), reusing the core helpers."""
    gs = GridSpec(lat0=38.0, lat1=5.0, lon0=65.0, lon1=100.0, dgrid=1.5)
    return build_grid_context(gs, REGION_MASK_NC)


def regrid(arr2d, src_lat, src_lon, GC):
    """Bilinear-interp a native (lat, lon) field onto the 1.5 deg verification grid
       and mask ocean (NaN). `arr2d` is numpy; src_lat/src_lon its native coords."""
    da = xr.DataArray(arr2d, dims=["lat", "lon"],
                      coords={"lat": src_lat, "lon": src_lon})
    out = da.interp(lat=GC["lat"], lon=GC["lon"], method="linear")
    return out.where(GC["land"])


def india_land_mean(da, GC):
    """Cosine-weighted mean over India LAND points (ocean already NaN)."""
    w = GC["weights"]
    masked = da.where(GC["land"])
    return float(masked.weighted(w).mean(["lat", "lon"]).item())


# ============================================================ map furniture ====
def state_geoms():
    """Survey-of-India state polygons reprojected LCC -> WGS84 for cartopy."""
    tr = Transformer.from_crs(_LCC, _WGS84, always_xy=True)
    return [shp_transform(tr.transform, rec.geometry)
            for rec in shpreader.Reader(SOI_SHP).records()]


def base_ax(ax, geoms):
    """Shared India basemap furniture: extent, coast, borders, SOI states, labels."""
    ax.set_extent(INDIA_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, color="0.25")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, color="0.4")
    ax.add_geometries(geoms, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.3", linewidth=0.45)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.7",
                      linestyle=":", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 10}
    return ax


def pcolor(ax, da, GC, **kw):
    """pcolormesh of a (lat, lon) DataArray on the India GeoAxes."""
    return ax.pcolormesh(GC["lon"], GC["lat"], da.transpose("lat", "lon").values,
                         transform=ccrs.PlateCarree(), shading="auto", **kw)


# =============================================================== computation ====
def compute_year_stack(years, workers):
    """Parallel JJAS-mean rainfall (mm/day) and t2m (K) for every year, regridded
       onto the verification grid. Returns (tp_stack, t2m_stack) as (year, lat, lon)
       DataArrays plus the year coordinate."""
    GC = build_context()
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(year_jjas_fields, y): y for y in years}
        for f in as_completed(futs):
            y, tp, t2, slat, slon = f.result()
            results[y] = (regrid(tp, slat, slon, GC), regrid(t2, slat, slon, GC))
            print(f"    year {y} done ({len(results)}/{len(years)})", flush=True)
    print(f"  per-year reductions ({len(years)} yrs, {workers} workers): "
          f"{time.time() - t0:.1f}s", flush=True)
    tp_stack = xr.concat([results[y][0] for y in years], dim="year")
    t2_stack = xr.concat([results[y][1] for y in years], dim="year")
    tp_stack["year"] = list(years)
    t2_stack["year"] = list(years)
    return GC, tp_stack, t2_stack


def linregress_da(stack, x):
    """Per-grid OLS slope and intercept of `stack` (year, lat, lon) vs scalar x.
       Returns (slope, intercept) DataArrays in units-per-(x unit)."""
    x = np.asarray(x, dtype=float)
    xm = x.mean()
    xa = xr.DataArray(x - xm, dims="year", coords={"year": stack["year"]})
    ya = stack - stack.mean("year")
    slope = (xa * ya).sum("year") / (xa ** 2).sum("year")
    intercept = stack.mean("year") - slope * xm
    return slope, intercept


# ================================================================= figure 1 ====
def fig_clim_maps(GC, geoms, tp_clim, t2_clim, z_clim, years, out):
    """JJAS climatology maps: rainfall (YlGnBu), z500 (filled+contours), t2m (coolwarm)."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    span = f"{years[0]}-{years[-1]}"

    base_ax(axes[0], geoms)
    vmax = max(15.0, float(np.ceil(np.nanpercentile(tp_clim.values, 99))))
    m0 = pcolor(axes[0], tp_clim, GC, cmap="YlGnBu", vmin=0, vmax=vmax)
    cb0 = fig.colorbar(m0, ax=axes[0], orientation="horizontal", pad=0.06,
                       shrink=0.9, extend="max")
    cb0.set_label("rainfall (mm/day)")
    axes[0].set_title("JJAS mean rainfall")

    base_ax(axes[1], geoms)
    m1 = pcolor(axes[1], z_clim, GC, cmap="viridis")
    cs = axes[1].contour(GC["lon"], GC["lat"], z_clim.transpose("lat", "lon").values,
                         colors="k", linewidths=0.7, levels=8,
                         transform=ccrs.PlateCarree())
    axes[1].clabel(cs, inline=True, fontsize=8, fmt="%d")
    cb1 = fig.colorbar(m1, ax=axes[1], orientation="horizontal", pad=0.06, shrink=0.9)
    cb1.set_label("z500 (gpm)")
    axes[1].set_title("JJAS mean 500 hPa height")

    base_ax(axes[2], geoms)
    t2c = t2_clim - 273.15
    tlo = float(np.nanpercentile(t2c.values, 2))
    thi = float(np.nanpercentile(t2c.values, 98))
    m2 = pcolor(axes[2], t2c, GC, cmap="coolwarm", vmin=tlo, vmax=thi)
    cb2 = fig.colorbar(m2, ax=axes[2], orientation="horizontal", pad=0.06,
                       shrink=0.9, extend="both")
    cb2.set_label("2 m temperature (degC)")
    axes[2].set_title("JJAS mean 2 m temperature")

    fig.suptitle(f"Indian summer monsoon (JJAS) climatology  -  ERA5 {span}", y=1.02)
    path = os.path.join(out, "1_clim_maps_jjas.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================= figure 2 ====
def fig_annual_cycle(GC, years, workers, out):
    """12-month climatological all-India-land mean rainfall + t2m (two panels)."""
    months = compute_monthly_clim(GC, years, workers)
    mnum = np.arange(1, 13)
    labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    rain = np.array([months[m][0] for m in mnum])
    temp = np.array([months[m][1] for m in mnum]) - 273.15

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    jjas = [6, 7, 8, 9]
    ax1.bar(mnum, rain, color="#1f6f8b", width=0.65, zorder=3)
    for m in jjas:
        ax1.bar(m, rain[m - 1], color="#0072B2", width=0.65, zorder=4)
    ax1.set_ylabel("rainfall (mm/day)")
    ax1.set_title("All-India land-mean annual cycle: rainfall")
    ax1.grid(axis="y", alpha=0.3, ls=":")
    ax1.axvspan(5.5, 9.5, color="0.9", zorder=0)
    ax1.text(7.5, ax1.get_ylim()[1] * 0.92, "MONSOON\n(JJAS)", ha="center",
             va="top", fontsize=12, color="#0072B2", weight="bold")

    ax2.plot(mnum, temp, "-o", color="#D55E00", lw=2.6, ms=8, zorder=3)
    ax2.set_ylabel("2 m temperature (degC)")
    ax2.set_xlabel("month")
    ax2.set_title("All-India land-mean annual cycle: 2 m temperature")
    ax2.grid(alpha=0.3, ls=":")
    ax2.axvspan(5.5, 9.5, color="0.9", zorder=0)
    ax2.set_xticks(mnum)
    ax2.set_xticklabels(labels)

    fig.suptitle(f"Annual cycle over India  -  ERA5 {years[0]}-{years[-1]}", y=1.0)
    path = os.path.join(out, "2_annual_cycle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _month_mean_worker(args):
    """WORKER: India-box monthly-mean rainfall (mm/day) and t2m (K) for one month
       across all `years` (returns per-month all-year mean fields as numpy)."""
    month, years = args
    ds = _open_box()
    tp_pieces, t2_pieces = [], []
    for y in years:
        sub = ds.sel(time=(ds.time.dt.year == y) & (ds.time.dt.month == month))
        tp = (sub["total_precipitation_6hr"].resample(time="1D").sum() * 1000.0).mean("time")
        t2 = sub["2m_temperature"].resample(time="1D").mean().mean("time")
        tp_pieces.append(_to_latlon(tp).load())
        t2_pieces.append(_to_latlon(t2).load())
    tp = (sum(tp_pieces) / len(tp_pieces))
    t2 = (sum(t2_pieces) / len(t2_pieces))
    return (month, tp.values.astype("float32"), t2.values.astype("float32"),
            tp["lat"].values, tp["lon"].values)


def compute_monthly_clim(GC, years, workers):
    """{month: (rain_landmean mm/day, t2m_landmean K)} climatology, parallel over months."""
    out = {}
    with ProcessPoolExecutor(max_workers=min(workers, 12)) as ex:
        futs = {ex.submit(_month_mean_worker, (m, years)): m for m in range(1, 13)}
        for f in as_completed(futs):
            m, tp, t2, slat, slon = f.result()
            rda = regrid(tp, slat, slon, GC)
            tda = regrid(t2, slat, slon, GC)
            out[m] = (india_land_mean(rda, GC), india_land_mean(tda, GC))
    return out


# ================================================================= figure 3 ====
def fig_interannual(GC, tp_stack, t2_stack, years, out):
    """JJAS all-India-land mean rainfall & t2m time series + trend + +/-1 sigma band;
       strong/weak monsoon years marked."""
    yrs = np.array(years, dtype=float)
    rain = np.array([india_land_mean(tp_stack.sel(year=y), GC) for y in years])
    temp = np.array([india_land_mean(t2_stack.sel(year=y), GC) for y in years]) - 273.15

    def _trend(y):
        s, i = np.polyfit(yrs, y, 1)
        return s, i

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # --- rainfall ---
    rm, rs = rain.mean(), rain.std()
    sr, ir = _trend(rain)
    ax1.axhspan(rm - rs, rm + rs, color="#0072B2", alpha=0.12, zorder=0,
                label="+/-1 sigma")
    ax1.axhline(rm, color="0.4", ls="--", lw=1.2, zorder=1)
    ax1.plot(yrs, rain, "-o", color="#0072B2", lw=2.0, ms=5, zorder=3, label="JJAS rain")
    ax1.plot(yrs, sr * yrs + ir, color="k", lw=2.2, ls="-", zorder=4,
             label=f"trend {sr * 10:+.2f} mm/day/decade")
    strong = yrs[rain >= rm + rs]
    weak = yrs[rain <= rm - rs]
    ax1.scatter(strong, rain[rain >= rm + rs], s=90, facecolor="none",
                edgecolor="green", lw=2, zorder=5, label="strong (>+1s)")
    ax1.scatter(weak, rain[rain <= rm - rs], s=90, facecolor="none",
                edgecolor="red", lw=2, zorder=5, label="weak (<-1s)")
    ax1.set_ylabel("JJAS rainfall (mm/day)")
    ax1.set_title("Interannual variability of all-India JJAS rainfall")
    ax1.grid(alpha=0.3, ls=":")
    ax1.legend(ncol=3, loc="upper center", fontsize=10)

    # --- temperature ---
    tm, ts = temp.mean(), temp.std()
    st, it = _trend(temp)
    ax2.axhspan(tm - ts, tm + ts, color="#D55E00", alpha=0.12, zorder=0)
    ax2.axhline(tm, color="0.4", ls="--", lw=1.2, zorder=1)
    ax2.plot(yrs, temp, "-o", color="#D55E00", lw=2.0, ms=5, zorder=3, label="JJAS t2m")
    ax2.plot(yrs, st * yrs + it, color="k", lw=2.2, zorder=4,
             label=f"warming {st * 10:+.2f} degC/decade")
    ax2.set_ylabel("JJAS 2 m temp (degC)")
    ax2.set_xlabel("year")
    ax2.set_title("Interannual variability of all-India JJAS 2 m temperature")
    ax2.grid(alpha=0.3, ls=":")
    ax2.legend(loc="upper left", fontsize=11)

    fig.suptitle(f"India JJAS interannual variability & trend  -  ERA5 "
                 f"{years[0]}-{years[-1]}", y=1.0)
    path = os.path.join(out, "3_interannual_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, (sr, st)


# ================================================================= figure 4 ====
def fig_trend_maps(GC, geoms, tp_stack, t2_stack, years, out):
    """Per-grid JJAS rainfall & t2m linear trend (per decade) over India."""
    yrs = np.array(years, dtype=float)
    sr, _ = linregress_da(tp_stack, yrs)
    st, _ = linregress_da(t2_stack, yrs)
    sr = (sr * 10.0).where(GC["land"])         # per decade
    st = (st * 10.0).where(GC["land"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    base_ax(axes[0], geoms)
    rmax = max(0.5, float(np.nanpercentile(np.abs(sr.values), 98)))
    m0 = pcolor(axes[0], sr, GC, cmap="BrBG", vmin=-rmax, vmax=rmax)
    cb0 = fig.colorbar(m0, ax=axes[0], orientation="horizontal", pad=0.06,
                       shrink=0.9, extend="both")
    cb0.set_label("rainfall trend (mm/day per decade)")
    axes[0].set_title("JJAS rainfall trend")

    base_ax(axes[1], geoms)
    tmax = max(0.1, float(np.nanpercentile(np.abs(st.values), 98)))
    m1 = pcolor(axes[1], st, GC, cmap="RdBu_r", vmin=-tmax, vmax=tmax)
    cb1 = fig.colorbar(m1, ax=axes[1], orientation="horizontal", pad=0.06,
                       shrink=0.9, extend="both")
    cb1.set_label("temperature trend (degC per decade)")
    axes[1].set_title("JJAS 2 m temperature trend")

    fig.suptitle(f"Long-term JJAS trends over India  -  ERA5 {years[0]}-{years[-1]}",
                 y=1.02)
    path = os.path.join(out, "4_trend_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================= figure 5 ====
def fig_miso_hovmoller(GC, years, year, workers, out):
    """Latitude-time Hovmoller of daily rainfall ANOMALY over the India lon band
       for one monsoon year -- northward-propagating active/break (MISO) bands.

       The daily climatology is built from ALL `years` (so the anomaly is a true
       intraseasonal anomaly), computed in parallel across years."""
    # target year slab
    _, slab, t_year, lat = year_jjas_daily_rain(year)
    n_t, n_lat = slab.shape

    # daily climatology across all years (parallel), aligned by day-of-season index
    clim_years = [y for y in years if y != year] or years
    accum = np.zeros((n_t, n_lat), dtype="float64")
    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(year_jjas_daily_rain, y): y for y in clim_years}
        for f in as_completed(futs):
            _, s, _, _ = f.result()
            if s.shape[0] >= n_t and s.shape[1] == n_lat:
                accum += s[:n_t, :]
                cnt += 1
    clim = accum / max(cnt, 1)
    anom = slab - clim

    # smooth lightly in time (3-day) to bring out the propagating envelope
    k = np.ones(3) / 3.0
    anom_s = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, anom)

    days = pd.to_datetime(t_year)
    fig, ax = plt.subplots(figsize=(12, 7))
    amax = float(np.nanpercentile(np.abs(anom_s), 97))
    mesh = ax.pcolormesh(np.arange(n_t), lat, anom_s.T, cmap="RdBu",
                         vmin=-amax, vmax=amax, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, extend="both")
    cb.set_label("daily rainfall anomaly (mm/day)")
    # monthly tick marks
    ticks, labs = [], []
    for i, d in enumerate(days):
        if d.day == 1:
            ticks.append(i)
            labs.append(d.strftime("%b 1"))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    ax.set_ylabel("latitude (degN)")
    ax.set_xlabel(f"day of {year} monsoon season")
    ax.set_title(f"MISO Hovmoller: daily rainfall anomaly over {MISO_LON_BAND[0]:.0f}-"
                 f"{MISO_LON_BAND[1]:.0f}E\n{year}  (northward-propagating "
                 f"active/break bands)")
    path = os.path.join(out, f"5_miso_hovmoller_{year}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================= figure 6 ====
def fig_interannual_std(GC, geoms, tp_stack, years, out):
    """Per-grid standard deviation of JJAS-mean rainfall across years."""
    sd = tp_stack.std("year").where(GC["land"])
    fig, ax = plt.subplots(figsize=(7.5, 7),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    base_ax(ax, geoms)
    vmax = max(1.0, float(np.nanpercentile(sd.values, 99)))
    mesh = pcolor(ax, sd, GC, cmap="magma_r", vmin=0, vmax=vmax)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85,
                      pad=0.03, extend="max")
    cb.set_label("std of JJAS-mean rainfall (mm/day)")
    ax.set_title(f"Interannual rainfall variability\nERA5 {years[0]}-{years[-1]} "
                 f"(JJAS std)")
    path = os.path.join(out, "6_interannual_std_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================= figure 7 ====
def fig_eof1(GC, geoms, tp_stack, years, out):
    """Leading EOF of DETRENDED JJAS rainfall over India: spatial pattern + PC.
       Cosine-latitude area weighting applied before the SVD."""
    yrs = np.array(years, dtype=float)
    slope, intercept = linregress_da(tp_stack, yrs)
    trend = slope * xr.DataArray(yrs, dims="year", coords={"year": tp_stack["year"]}) + intercept
    anom = (tp_stack - trend).where(GC["land"])           # detrended anomaly

    # area weighting (sqrt of cos-lat) and land mask -> 2D matrix (time x space)
    w = np.sqrt(np.cos(np.deg2rad(GC["lat"])))
    wda = xr.DataArray(w, dims="lat", coords={"lat": GC["lat"]})
    Aw = (anom * wda)
    land2d = GC["land"].values
    flat = Aw.values.reshape(len(years), -1)
    valid = np.isfinite(flat).all(axis=0)
    M = flat[:, valid]
    M = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    var_frac = (S ** 2) / (S ** 2).sum()

    # reconstruct EOF1 spatial pattern back onto the grid (un-weight for display)
    eof1 = np.full(flat.shape[1], np.nan)
    eof1[valid] = Vt[0]
    eof1_grid = eof1.reshape(len(GC["lat"]), len(GC["lon"]))
    eof1_da = xr.DataArray(eof1_grid, dims=["lat", "lon"],
                           coords={"lat": GC["lat"], "lon": GC["lon"]})
    eof1_da = (eof1_da / np.cos(np.deg2rad(xr.DataArray(GC["lat"], dims="lat",
                                                        coords={"lat": GC["lat"]}))) ** 0.5)
    pc1 = U[:, 0] * S[0]
    # sign convention: make the PC trend / spatial mean positive for readability
    if np.nanmean(eof1_da.values) < 0:
        eof1_da = -eof1_da
        pc1 = -pc1

    fig = plt.figure(figsize=(15, 6.2))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    base_ax(ax0, geoms)
    emax = float(np.nanpercentile(np.abs(eof1_da.values), 98))
    mesh = pcolor(ax0, eof1_da.where(GC["land"]), GC, cmap="RdBu_r",
                  vmin=-emax, vmax=emax)
    cb = fig.colorbar(mesh, ax=ax0, orientation="horizontal", pad=0.06,
                      shrink=0.9, extend="both")
    cb.set_label("EOF1 loading")
    ax0.set_title(f"EOF1 of detrended JJAS rainfall\n({var_frac[0] * 100:.1f}% variance)")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.axhline(0, color="0.4", lw=1.0)
    colors = ["#0072B2" if v >= 0 else "#D55E00" for v in pc1]
    ax1.bar(yrs, pc1, color=colors, width=0.8)
    ax1.set_xlabel("year")
    ax1.set_ylabel("PC1 amplitude")
    ax1.set_title("EOF1 principal component (PC1)")
    ax1.grid(axis="y", alpha=0.3, ls=":")

    fig.suptitle(f"Leading EOF of detrended JJAS rainfall over India  -  ERA5 "
                 f"{years[0]}-{years[-1]}", y=1.02)
    path = os.path.join(out, "7_eof1_rainfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, var_frac[0]


# =================================================================== driver =====
def parse_years(s):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted(int(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser(description="ERA5 JJAS monsoon observational analysis.")
    ap.add_argument("--years", default="1959-2022",
                    help="year range 'START-END' (full run: 1959-2022)")
    ap.add_argument("--quick", action="store_true",
                    help="fast test window (2010-2019) for a couple-minute run")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "era5_monsoon"),
                    help="output directory for PNGs")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel worker processes for per-year reductions")
    ap.add_argument("--miso-year", type=int, default=2019,
                    help="monsoon year for the MISO Hovmoller")
    a = ap.parse_args()

    years = parse_years("2010-2019") if a.quick else parse_years(a.years)
    os.makedirs(a.out, exist_ok=True)
    miso_year = a.miso_year if a.miso_year in years else years[-1]

    t_start = time.time()
    print(f"ERA5 JJAS monsoon analysis | years {years[0]}-{years[-1]} "
          f"({len(years)} yrs) | workers {a.workers} | out {a.out}", flush=True)

    # --- heavy parallel reductions (rain + t2m JJAS-mean per year) -------------
    GC, tp_stack, t2_stack = compute_year_stack(years, a.workers)
    tp_clim = tp_stack.mean("year")
    t2_clim = t2_stack.mean("year")
    print(f"  climatology India land-mean: rain {india_land_mean(tp_clim, GC):.2f} "
          f"mm/day, t2m {india_land_mean(t2_clim, GC) - 273.15:.2f} degC", flush=True)

    # --- z500 climatology (serial, single pass) --------------------------------
    print("  z500 climatology ...", flush=True)
    z_native = z500_clim_field(years)
    z_clim = regrid(z_native.values, z_native["lat"].values, z_native["lon"].values, GC)

    geoms = state_geoms()
    created = []
    created.append(fig_clim_maps(GC, geoms, tp_clim, t2_clim, z_clim, years, a.out))
    print("  [1/7] climatology maps", flush=True)

    created.append(fig_annual_cycle(GC, years, a.workers, a.out))
    print("  [2/7] annual cycle", flush=True)

    p, (sr, st) = fig_interannual(GC, tp_stack, t2_stack, years, a.out)
    created.append(p)
    print(f"  [3/7] interannual time series  (rain trend {sr * 10:+.3f} mm/day/dec, "
          f"warming {st * 10:+.3f} degC/dec)", flush=True)

    created.append(fig_trend_maps(GC, geoms, tp_stack, t2_stack, years, a.out))
    print("  [4/7] trend maps", flush=True)

    created.append(fig_miso_hovmoller(GC, years, miso_year, a.workers, a.out))
    print(f"  [5/7] MISO Hovmoller ({miso_year})", flush=True)

    created.append(fig_interannual_std(GC, geoms, tp_stack, years, a.out))
    print("  [6/7] interannual std map", flush=True)

    p, vf = fig_eof1(GC, geoms, tp_stack, years, a.out)
    created.append(p)
    print(f"  [7/7] EOF1 ({vf * 100:.1f}% variance)", flush=True)

    dt = time.time() - t_start
    print(f"\nDone in {dt:.1f}s. Created {len(created)} figures:")
    for c in created:
        print("  ", c)


if __name__ == "__main__":
    main()
