#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/plots/monsoon_maps.py  —  Spatial monsoon maps over India (JJAS).
================================================================================
The visual backbone of the JJAS monsoon paper: publication-quality cartopy maps
on the India extent, with Survey-of-India state boundaries overlaid, built by
REUSING the final_analysis verification core (truth / grid / adapters). Nothing
under core/ or jjas/ outside this folder is modified.

Figures produced (per `--years`):
  1. Observed JJAS rainfall per year  — small-multiples, JJAS (Jun-Sep) seasonal
     MEAN rainfall (mm/day) from WeatherBench2 ERA5. ("monsoon every year")
  2. JJAS rainfall anomaly per year   — each year's JJAS mean minus the multi-year
     JJAS climatology (diverging BrBG; wet/dry years).
  3. ECMWF forecast bias map          — ONE year, ONE lead window (Week 2): ECMWF
     reforecast week-mean rainfall minus the matching ERA5 obs window, on the
     common grid (diverging). cfgrib opens are slow (~70 s), so just one example.
  4. (optional, --skill) grid-point skill map — temporal correlation of the ECMWF
     weekly forecast vs ERA5 across the JJAS init dates of one year, per cell.

Usage
-----
  python monsoon_maps.py --years 2017-2019 --out figs/maps/
  python monsoon_maps.py --years 2017-2019 --no-ecmwf      # obs maps only (fast)
  python monsoon_maps.py --years 2019 --skill              # add the skill map

Run inside the s2s-hind conda env (cartopy + shapely + pyproj available).
================================================================================
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
JJAS_DIR = os.path.dirname(HERE)
FA_ROOT = os.path.dirname(JJAS_DIR)
sys.path.insert(0, FA_ROOT)                                # final_analysis/ on path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from core import Physics, get_adapter
from core.grid import build_grid_context, to_grid
from core.aggregate import valid_dates_for
from core.regions import _LCC, _WGS84           # SOI LCC -> WGS84 proj strings
from core.truth import open_truth_wb2, truth_period_mean
from jjas.config import WB2_ZARR, build_config
from jjas import adapters_jjas                   # noqa: F401  (registers ECMWF adapters)

# India map extent (lon_w, lon_e, lat_s, lat_n) for PlateCarree set_extent.
INDIA_EXTENT = (65.0, 100.0, 5.0, 38.0)
# Week-2 lead window (1-based lead days) used for the ECMWF bias / skill maps.
WEEK2 = (8, 14)


# ============================================================ map furniture ===
def _state_geoms(shp_path):
    """Reproject every Survey-of-India state polygon from LCC to WGS84 (lon/lat).
       Returns a list of shapely geometries ready for cartopy add_geometries."""
    tr = Transformer.from_crs(_LCC, _WGS84, always_xy=True)
    geoms = []
    for rec in shpreader.Reader(shp_path).records():
        geoms.append(shp_transform(tr.transform, rec.geometry))
    return geoms


def _base_ax(ax, state_geoms):
    """Apply the shared India basemap furniture to one GeoAxes: extent, coastline,
       state boundaries (SOI), and gridline labels."""
    ax.set_extent(INDIA_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, color="0.3")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, color="0.45")
    ax.add_geometries(state_geoms, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor="0.35", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.7",
                      linestyle=":", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return ax


def _grid_da(arr, GC):
    """Wrap a (lat, lon) ndarray/DataArray as a DataArray with the grid coords,
       always in (lat, lon) dim order so pcolormesh gets proper geographic axes.
       to_grid() can return (lon, lat) order depending on the source orientation,
       so transpose defensively."""
    if isinstance(arr, xr.DataArray):
        if set(arr.dims) >= {"lat", "lon"}:
            return arr.transpose("lat", "lon")
        return arr
    return xr.DataArray(arr, dims=["lat", "lon"],
                        coords={"lat": GC["lat"], "lon": GC["lon"]})


def _pcolor(ax, da, GC, **kw):
    """pcolormesh of a (lat, lon) field on the India GeoAxes."""
    da = _grid_da(da, GC)
    return ax.pcolormesh(GC["lon"], GC["lat"], da.values,
                         transform=ccrs.PlateCarree(), shading="auto", **kw)


def _panel_grid(n):
    """ncols, nrows for a tidy small-multiples layout of n panels."""
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    return ncols, nrows


# ============================================================== obs loading ===
def jjas_mean(tp_daily, year, GC, land_only=True):
    """JJAS (Jun 1 - Sep 30) seasonal-MEAN rainfall (mm/day) for `year`, regridded
       onto the verification grid and ocean-masked (land via core land mask)."""
    season = tp_daily.sel(time=slice(f"{year}-06-01", f"{year}-09-30")).mean("time")
    g = to_grid(season, GC)            # interp to grid + ocean -> NaN
    return g


def load_obs(years, GC):
    """{year: JJAS-mean DataArray} from WB2 ERA5, opened once across all years."""
    y0, y1 = min(years), max(years)
    t = open_truth_wb2(WB2_ZARR, Physics(), f"{y0}-06-01", f"{y1}-09-30")
    return {y: jjas_mean(t["tp_daily"], y, GC) for y in years}


# ================================================================ figure 1 ====
def fig_observed(obs, years, GC, state_geoms, out):
    """Small-multiples of observed JJAS-mean rainfall, one panel per year."""
    ncols, nrows = _panel_grid(len(years))
    vmax = float(np.nanpercentile(np.stack([obs[y].values for y in years]), 99))
    vmax = max(5.0, np.ceil(vmax))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    axes = axes.ravel()
    mesh = None
    for ax, y in zip(axes, years):
        _base_ax(ax, state_geoms)
        mesh = _pcolor(ax, obs[y], GC, cmap="YlGnBu", vmin=0, vmax=vmax)
        mu = float(np.nanmean(obs[y].values))
        ax.set_title(f"JJAS {y}   (mean {mu:.1f} mm/day)", fontsize=10)
    for ax in axes[len(years):]:
        ax.axis("off")
    cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="vertical",
                        shrink=0.7, pad=0.02, extend="max")
    cbar.set_label("JJAS mean rainfall (mm/day)")
    fig.suptitle("Observed JJAS monsoon rainfall (ERA5 / WeatherBench2)",
                 fontsize=13, y=0.99)
    path = os.path.join(out, "obs_jjas_rainfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================ figure 2 ====
def fig_anomaly(obs, years, GC, state_geoms, out, clim):
    """Small-multiples of JJAS-mean anomaly (year minus multi-year climatology)."""
    ncols, nrows = _panel_grid(len(years))
    anoms = {y: obs[y] - clim for y in years}
    amax = float(np.nanpercentile(np.abs(np.stack([anoms[y].values for y in years])), 98))
    amax = max(1.0, np.ceil(amax))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)
    axes = axes.ravel()
    mesh = None
    for ax, y in zip(axes, years):
        _base_ax(ax, state_geoms)
        mesh = _pcolor(ax, anoms[y], GC, cmap="BrBG", vmin=-amax, vmax=amax)
        mu = float(np.nanmean(anoms[y].values))
        ax.set_title(f"JJAS {y} anomaly   (mean {mu:+.1f} mm/day)", fontsize=10)
    for ax in axes[len(years):]:
        ax.axis("off")
    cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="vertical",
                        shrink=0.7, pad=0.02, extend="both")
    cbar.set_label("JJAS rainfall anomaly (mm/day)")
    fig.suptitle("JJAS rainfall anomaly vs multi-year climatology (ERA5)",
                 fontsize=13, y=0.99)
    path = os.path.join(out, "anomaly_jjas_rainfall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================ figure 3 ====
def fig_ecmwf_bias(year, GC, state_geoms, out, clim_years):
    """ONE ECMWF reforecast bias map: Week-2 week-mean rainfall for the first JJAS
       init of `year`, minus the ERA5 obs averaged over the SAME valid window.

       Returns (path, runtime_seconds). cfgrib opens dominate the runtime."""
    t0 = time.time()
    cfg = build_config(year, dgrid=GC["lat"][0] - GC["lat"][1] if len(GC["lat"]) > 1 else 1.5)
    spec = cfg.model("ECMWF")
    ds, de = WEEK2

    # First JJAS init whose ECMWF Week-2 cube exists for this hindcast year.
    fc_mu = init_used = None
    for init in cfg.init_dates:
        cube = get_adapter("ecmwf_reforecast")(init, "TP", spec, Physics())
        if cube is not None and cube.has_week(de):
            fc_mu, _ = cube.weekly(ds, de, GC)          # ens-mean week-2 rate (mm/day)
            init_used = init
            break
    if fc_mu is None:
        print("  [ECMWF bias] no usable init found; skipping.")
        return None, time.time() - t0

    # ERA5 obs over the matching Week-2 valid window for the same init.
    valid = valid_dates_for(init_used, ds, de, end=f"{year}-10-15")
    truth = open_truth_wb2(WB2_ZARR, Physics(),
                           valid[0], valid[-1])
    obs_mu = truth_period_mean("TP", truth, valid, GC)    # mm/day on grid
    bias = _grid_da(fc_mu, GC) - _grid_da(obs_mu, GC)

    bmax = float(np.nanpercentile(np.abs(bias.values), 98))
    bmax = max(2.0, np.ceil(bmax))
    fig, ax = plt.subplots(figsize=(5.2, 4.8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    _base_ax(ax, state_geoms)
    mesh = _pcolor(ax, bias, GC, cmap="RdBu", vmin=-bmax, vmax=bmax)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85,
                        pad=0.03, extend="both")
    cbar.set_label("Forecast - obs rainfall (mm/day)")
    mb = float(np.nanmean(bias.values))
    ax.set_title(f"ECMWF Week-2 rainfall bias\ninit {init_used} "
                 f"(valid {valid[0]}..{valid[-1]})\nmean bias {mb:+.2f} mm/day",
                 fontsize=10)
    path = os.path.join(out, f"ecmwf_bias_week2_{year}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, time.time() - t0


# ================================================================ figure 4 ====
def fig_skill(year, GC, state_geoms, out):
    """Grid-point temporal correlation of the ECMWF Week-2 ens-mean forecast vs
       ERA5 across all JJAS init dates of `year`. Slow (one cfgrib open per init)."""
    t0 = time.time()
    cfg = build_config(year, dgrid=GC["lat"][0] - GC["lat"][1] if len(GC["lat"]) > 1 else 1.5)
    spec = cfg.model("ECMWF")
    ds, de = WEEK2
    truth = open_truth_wb2(WB2_ZARR, Physics(), f"{year}-06-01", f"{year}-10-15")

    fcs, obss = [], []
    for init in cfg.init_dates:
        cube = get_adapter("ecmwf_reforecast")(init, "TP", spec, Physics())
        if cube is None or not cube.has_week(de):
            continue
        fc_mu, _ = cube.weekly(ds, de, GC)
        valid = valid_dates_for(init, ds, de, end=f"{year}-10-15")
        if len(valid) < (de - ds + 1):
            continue
        obs_mu = truth_period_mean("TP", truth, valid, GC)
        if obs_mu is None:
            continue
        fcs.append(_grid_da(fc_mu, GC))
        obss.append(_grid_da(obs_mu, GC))
    if len(fcs) < 4:
        print(f"  [skill] only {len(fcs)} usable inits; skipping skill map.")
        return None, time.time() - t0

    F = xr.concat(fcs, dim="init")
    O = xr.concat(obss, dim="init")
    Fa = F - F.mean("init")
    Oa = O - O.mean("init")
    num = (Fa * Oa).mean("init")
    den = np.sqrt((Fa ** 2).mean("init") * (Oa ** 2).mean("init"))
    corr = num / den

    fig, ax = plt.subplots(figsize=(5.2, 4.8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    _base_ax(ax, state_geoms)
    mesh = _pcolor(ax, corr, GC, cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85,
                        pad=0.03, extend="neither")
    cbar.set_label("Temporal correlation (forecast vs ERA5)")
    mc = float(np.nanmean(corr.values))
    ax.set_title(f"ECMWF Week-2 rainfall skill (corr) {year}\n"
                 f"{len(fcs)} JJAS inits  (India mean r={mc:+.2f})", fontsize=10)
    path = os.path.join(out, f"ecmwf_skill_week2_{year}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, time.time() - t0


# =================================================================== driver ===
def parse_years(s):
    """'2017-2019' or '2019' or '2015,2017,2019' -> sorted list of ints."""
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted(int(x) for x in s.split(","))


def main():
    ap = argparse.ArgumentParser(description="JJAS spatial monsoon maps over India.")
    ap.add_argument("--years", default="2017-2019",
                    help="year range '2017-2019' / list '2015,2017' / single '2019'")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "maps"),
                    help="output directory for PNGs")
    ap.add_argument("--dgrid", type=float, default=1.5, help="common-grid spacing (deg)")
    ap.add_argument("--clim", default="2002-2019",
                    help="JJAS climatology window for the anomaly maps "
                         "('requested' = use the --years span)")
    ap.add_argument("--no-ecmwf", action="store_true",
                    help="skip the (slow) ECMWF bias map")
    ap.add_argument("--skill", action="store_true",
                    help="also build the per-gridpoint ECMWF skill map (very slow)")
    ap.add_argument("--ecmwf-year", type=int, default=None,
                    help="year for the single ECMWF bias / skill map (default: last)")
    a = ap.parse_args()

    years = parse_years(a.years)
    os.makedirs(a.out, exist_ok=True)
    print(f"Years: {years}   out: {a.out}")

    cfg = build_config(years[-1], a.dgrid)
    GC = build_grid_context(cfg.grid, cfg.paths.region_mask_nc)
    state_geoms = _state_geoms(cfg.paths.soi_shapefile)
    print(f"Grid {len(GC['lat'])}x{len(GC['lon'])} @ {a.dgrid} deg; "
          f"{len(state_geoms)} SOI state polygons loaded.")

    # --- observed JJAS rainfall (fast, the priority) -------------------------
    obs = load_obs(years, GC)
    for y in years:
        print(f"  obs JJAS {y}: India-mean {float(np.nanmean(obs[y].values)):.2f} "
              f"mm/day, max {float(np.nanmax(obs[y].values)):.1f} mm/day")
    created = [fig_observed(obs, years, GC, state_geoms, a.out)]

    # --- climatology for anomaly maps ----------------------------------------
    if a.clim == "requested":
        clim_years = years
        clim = sum(obs[y] for y in years) / len(years)
    else:
        c0, c1 = (int(x) for x in a.clim.split("-"))
        clim_years = list(range(c0, c1 + 1))
        ct = open_truth_wb2(WB2_ZARR, Physics(), f"{c0}-06-01", f"{c1}-09-30")
        clim_fields = [jjas_mean(ct["tp_daily"], y, GC) for y in clim_years]
        clim = sum(clim_fields) / len(clim_fields)
    print(f"  JJAS climatology over {clim_years[0]}-{clim_years[-1]} "
          f"({len(clim_years)} yrs): India-mean {float(np.nanmean(clim.values)):.2f} mm/day")
    created.append(fig_anomaly(obs, years, GC, state_geoms, a.out, clim))

    # --- ECMWF bias map (slow, secondary) ------------------------------------
    ec_year = a.ecmwf_year or years[-1]
    if not a.no_ecmwf:
        print(f"  ECMWF Week-2 bias map for {ec_year} (cfgrib opens are slow) ...")
        path, dt = fig_ecmwf_bias(ec_year, GC, state_geoms, a.out, clim_years)
        print(f"    ECMWF bias map step took {dt:.1f}s")
        if path:
            created.append(path)

    # --- optional skill map ---------------------------------------------------
    if a.skill:
        print(f"  ECMWF Week-2 skill map for {ec_year} (very slow) ...")
        path, dt = fig_skill(ec_year, GC, state_geoms, a.out)
        print(f"    skill map step took {dt:.1f}s")
        if path:
            created.append(path)

    print("\nCreated:")
    for p in created:
        print(" ", p)


if __name__ == "__main__":
    main()
