#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis/monsoon_dynamics.py  —  ERA5 monsoon CIRCULATION & TELECONNECTIONS.
================================================================================
A self-contained, PARALLELIZED observational study of the DYNAMICS of the Indian
summer monsoon (JJAS) and its large-scale teleconnections, built straight from
the WeatherBench2 ERA5 zarr (1.5 deg, 6-hourly, 1959-2023).

This is the CIRCULATION + TELECONNECTION companion to analysis/era5_monsoon.py
(which owns the rainfall/temperature climatology, annual cycle, interannual,
trends and MISO Hovmoller). The two scripts share NO outputs. This one only
READS the final_analysis verification core (regions / projection strings) and
modifies nothing under core/ or jjas/.

Heavy per-year seasonal reductions (winds, SST, rainfall, IVT, MSLP) are run in
parallel with a ProcessPoolExecutor: each worker opens the zarr lazily and loads
ONLY its own year's JJAS slice over the domains it needs, so the work scales
linearly to all 64 years -- run overnight with no --quick.

Figures (PNGs into analysis/figs/monsoon_dynamics/)
---------------------------------------------------
  1. circulation_jjas      850 hPa winds (Somali jet + cross-eq flow, shaded
                           wind speed) and 200 hPa winds (Tropical Easterly Jet)
                           over the monsoon domain, India + coastlines overlaid.
  2. webster_yang_index    Webster-Yang index (U850 - U200, 0-20N 40-110E),
                           JJAS-mean per year, with mean + linear trend, strong/
                           weak years marked.
  3. enso_teleconnection   Nino-3.4 (SST anom, 5S-5N 170-120W) vs all-India JJAS
                           rainfall (ISMR) scatter + correlation, and a composite
                           map of JJAS rainfall anomaly for El Nino minus La Nina.
  4. iod_teleconnection    Dipole Mode Index (West 50-70E 10S-10N minus East
                           90-110E 10S-0) vs ISMR scatter + correlation.
  5. ivt_mslp_jjas         JJAS IVT magnitude (shaded) + 850 hPa moisture-flux
                           vectors over the monsoon domain, and MSLP showing the
                           monsoon trough.
  6. sst_ismr_corr_map     per-grid correlation of tropical Indo-Pacific JJAS SST
                           anomaly with ISMR (the teleconnection footprint).

Usage
-----
  # fast sanity run (recent 30 yrs, a few minutes):
  python monsoon_dynamics.py --quick

  # full overnight run (every complete JJAS, 1959-2022):
  python monsoon_dynamics.py --years 1959-2022 --workers 16

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
from scipy import stats

from core.regions import _LCC, _WGS84

# ---------------------------------------------------------------- constants ---
WB2_ZARR = ("/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
SOI_SHP = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
G = 9.80665                                               # gravity (m/s^2)
# Pressure levels (hPa) used for the vertical integral of moisture transport.
# WB2's stored `integrated_vapor_transport` is in opaque/normalised units (it is
# ~80x too small vs the physical value), so we compute IVT = (1/g) integral of
# q*V dp ourselves -> proper kg/m/s. These are the levels present in the zarr.
IVT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
# numpy>=2 renamed trapz -> trapezoid; support both (avoid eager getattr default).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# --- domains (lat south..north, lon west..east in 0-360 convention) ---------
# Monsoon circulation domain for the wind / IVT maps.
MON = dict(s=-10.0, n=35.0, w=30.0, e=120.0)
# All-India rainfall (ISMR) land box.
ISMR = dict(s=5.0, n=38.0, w=65.0, e=100.0)
# Webster-Yang index averaging box: 0-20N, 40-110E.
WY = dict(s=0.0, n=20.0, w=40.0, e=110.0)
# Nino-3.4 SST box: 5S-5N, 170-120W = 190-240E.
NINO34 = dict(s=-5.0, n=5.0, w=190.0, e=240.0)
# IOD Dipole-Mode-Index boxes (Saji et al. 1999).
DMI_W = dict(s=-10.0, n=10.0, w=50.0, e=70.0)             # west pole
DMI_E = dict(s=-10.0, n=0.0,  w=90.0, e=110.0)            # east pole
# Tropical Indo-Pacific box for the per-grid SST-ISMR correlation map.
INDOPAC = dict(s=-30.0, n=30.0, w=40.0, e=290.0)

# Map extents (lon_w, lon_e, lat_s, lat_n) for set_extent.
MON_EXTENT = (MON["w"], MON["e"], MON["s"], MON["n"])
INDIA_EXTENT = (65.0, 100.0, 5.0, 38.0)
INDOPAC_EXTENT = (INDOPAC["w"], INDOPAC["e"], INDOPAC["s"], INDOPAC["n"])

# Strong/weak Webster-Yang years are flagged at +/- this many sigma.
WY_SIGMA = 1.0
# El Nino / La Nina classification threshold on the Nino-3.4 index (deg C).
ENSO_THR = 0.5

# Big, presentation-friendly fonts.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 18, "figure.titleweight": "bold",
})


# =============================================================== zarr helpers ==
def _open(box=None):
    """Open the WB2 zarr lazily, optionally pre-sliced to a lat/lon `box`
       (dict s,n,w,e). Latitude is ascending (-90..90), longitude 0..358.5."""
    ds = xr.open_zarr(WB2_ZARR)
    if box is not None:
        ds = ds.sel(latitude=slice(box["s"], box["n"]),
                    longitude=slice(box["w"], box["e"]))
    return ds


def _ll(da):
    """Rename WB2 latitude/longitude -> lat/lon, order dims (..., lat, lon)."""
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    lead = [d for d in da.dims if d not in ("lat", "lon")]
    return da.transpose(*lead, "lat", "lon")


def _box_mean(da, box):
    """Cosine-latitude weighted area mean over a lat/lon box (da already sliced
       or global). Returns a DataArray reduced over (lat, lon)."""
    sub = da.sel(lat=slice(box["s"], box["n"]), lon=slice(box["w"], box["e"]))
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(["lat", "lon"])


def _jjas(ds, year):
    """Slice a dataset/dataarray to JJAS (Jun 1 - Sep 30) of one year."""
    return ds.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))


# ==================================================== parallel per-year workers ==
# Each worker returns plain numpy arrays + coords so results pickle cheaply.

def year_indices(year):
    """WORKER: all the scalar JJAS indices for one year ->
       (year, U850box, U200box, nino34_sst, dmiW_sst, dmiE_sst, ismr_rain).
       SST values are RAW JJAS-mean (deg C); anomalies are removed in master."""
    ds = _open()
    sub = _jjas(ds, year)
    # --- Webster-Yang zonal winds over 0-20N, 40-110E -----------------------
    u = _ll(sub["u_component_of_wind"])
    u850 = float(_box_mean(u.sel(level=850).mean("time"), WY))
    u200 = float(_box_mean(u.sel(level=200).mean("time"), WY))
    # --- SST boxes (sea_surface_temperature is in Kelvin) -------------------
    sst = _ll(sub["sea_surface_temperature"]).mean("time") - 273.15
    nino = float(_box_mean(sst, NINO34))
    dmi_w = float(_box_mean(sst, DMI_W))
    dmi_e = float(_box_mean(sst, DMI_E))
    # --- ISMR all-India JJAS rainfall (mm/day), land+ocean box mean ---------
    tp = (_ll(sub["total_precipitation_6hr"]).resample(time="1D").sum() * 1000.0
          ).mean("time")
    ismr = float(_box_mean(tp, ISMR))
    return year, u850, u200, nino, dmi_w, dmi_e, ismr


def year_rain_field(year, box=ISMR):
    """WORKER: JJAS-mean rainfall (mm/day) field over the ISMR box, native grid."""
    ds = _open(box)
    tp = (_ll(_jjas(ds, year)["total_precipitation_6hr"]).resample(time="1D").sum()
          * 1000.0).mean("time").load()
    return year, tp.values.astype("float32"), tp["lat"].values, tp["lon"].values


def year_sst_field(year, box=INDOPAC):
    """WORKER: JJAS-mean SST (deg C) field over the Indo-Pacific box, native grid."""
    ds = _open(box)
    sst = (_ll(_jjas(ds, year)["sea_surface_temperature"]).mean("time") - 273.15
           ).load()
    return year, sst.values.astype("float32"), sst["lat"].values, sst["lon"].values


def _ivt_components(sub):
    """Vertically-integrated water-vapour transport (kg m^-1 s^-1) of a JJAS-mean
       column: Qu, Qv = (1/g) integral of q*u, q*v dp over IVT_LEVELS. Returns two
       (lat, lon) DataArrays. Integral sign handled for descending levels."""
    q = _ll(sub["specific_humidity"].sel(level=IVT_LEVELS)).mean("time")
    u = _ll(sub["u_component_of_wind"].sel(level=IVT_LEVELS)).mean("time")
    v = _ll(sub["v_component_of_wind"].sel(level=IVT_LEVELS)).mean("time")
    p = np.asarray(IVT_LEVELS, dtype=float) * 100.0       # hPa -> Pa
    qu = (q * u).transpose("level", "lat", "lon").values
    qv = (q * v).transpose("level", "lat", "lon").values
    Qu = -_trapz(qu, p, axis=0) / G                       # levels descend -> negate
    Qv = -_trapz(qv, p, axis=0) / G
    coords = {"lat": q["lat"].values, "lon": q["lon"].values}
    Qu = xr.DataArray(Qu, dims=["lat", "lon"], coords=coords)
    Qv = xr.DataArray(Qv, dims=["lat", "lon"], coords=coords)
    return Qu, Qv


def year_circ_fields(year, box=MON):
    """WORKER: JJAS-mean circulation fields over the monsoon domain (native grid):
       U850, V850, speed850, U200, V200, speed200, MSLP (hPa), and the
       vertically-integrated moisture transport Qu, Qv (kg/m/s) for IVT.
       Returns a dict of numpy arrays + the (lat, lon) coords."""
    ds = _open(box)
    sub = _jjas(ds, year)
    u = _ll(sub["u_component_of_wind"]).mean("time")
    v = _ll(sub["v_component_of_wind"]).mean("time")
    u850, v850 = u.sel(level=850), v.sel(level=850)
    u200, v200 = u.sel(level=200), v.sel(level=200)
    Qu, Qv = _ivt_components(sub)
    mslp = _ll(sub["mean_sea_level_pressure"]).mean("time") / 100.0   # Pa -> hPa
    out = dict(
        u850=u850.values.astype("float32"), v850=v850.values.astype("float32"),
        u200=u200.values.astype("float32"), v200=v200.values.astype("float32"),
        ivt_u=Qu.values.astype("float32"), ivt_v=Qv.values.astype("float32"),
        mslp=mslp.values.astype("float32"),
        lat=u850["lat"].values, lon=u850["lon"].values)
    out["year"] = year
    return out


# ==================================================================== drivers ==
def compute_indices(years, workers):
    """Parallel scalar indices for every year. Returns a tidy DataFrame indexed
       by year with raw box means, plus derived anomaly indices (Nino3.4, DMI)
       computed against the multi-year JJAS climatology, and the Webster-Yang
       index U850 - U200 (raw, no anomaly needed)."""
    rows = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(year_indices, y): y for y in years}
        for f in as_completed(futs):
            y, u850, u200, nino, dw, de, ismr = f.result()
            rows[y] = dict(u850=u850, u200=u200, sst_nino34=nino,
                           sst_dmiW=dw, sst_dmiE=de, ismr=ismr)
            print(f"    indices year {y} ({len(rows)}/{len(years)})", flush=True)
    print(f"  scalar indices ({len(years)} yrs, {workers} workers): "
          f"{time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "year"
    # Webster-Yang index (raw shear; positive -> strong monsoon).
    df["wy"] = df["u850"] - df["u200"]
    # ENSO / IOD as SST ANOMALIES about the multi-year JJAS climatology.
    df["nino34"] = df["sst_nino34"] - df["sst_nino34"].mean()
    df["dmi"] = (df["sst_dmiW"] - df["sst_dmiW"].mean()) - \
                (df["sst_dmiE"] - df["sst_dmiE"].mean())
    # ISMR anomaly (mm/day) for the composite / scatter axes.
    df["ismr_anom"] = df["ismr"] - df["ismr"].mean()
    return df


def compute_field_stack(worker, years, workers, label):
    """Generic parallel per-year FIELD stack. `worker` returns
       (year, arr2d, lat, lon). Returns (stack (year,lat,lon) DataArray)."""
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, y): y for y in years}
        for f in as_completed(futs):
            y, arr, lat, lon = f.result()
            res[y] = xr.DataArray(arr, dims=["lat", "lon"],
                                  coords={"lat": lat, "lon": lon})
            print(f"    {label} year {y} ({len(res)}/{len(years)})", flush=True)
    print(f"  {label} fields ({len(years)} yrs, {workers} workers): "
          f"{time.time() - t0:.1f}s", flush=True)
    stack = xr.concat([res[y] for y in years], dim="year")
    stack["year"] = list(years)
    return stack


def circulation_climatology(years, workers):
    """Parallel JJAS circulation climatology (mean over `years`) on the monsoon
       domain. Returns a dict of (lat, lon) DataArrays."""
    keys = ("u850", "v850", "u200", "v200", "ivt_u", "ivt_v", "mslp")
    acc, n = None, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(year_circ_fields, y): y for y in years}
        for f in as_completed(futs):
            d = f.result()
            if acc is None:
                lat, lon = d["lat"], d["lon"]
                acc = {k: np.zeros((len(lat), len(lon)), dtype="float64") for k in keys}
            for k in keys:
                acc[k] += d[k]
            n += 1
            print(f"    circ year {d['year']} ({n}/{len(years)})", flush=True)
    print(f"  circulation fields ({len(years)} yrs, {workers} workers): "
          f"{time.time() - t0:.1f}s", flush=True)
    clim = {k: xr.DataArray(acc[k] / n, dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon}) for k in keys}
    clim["speed850"] = np.hypot(clim["u850"], clim["v850"])
    clim["speed200"] = np.hypot(clim["u200"], clim["v200"])
    clim["ivt"] = np.hypot(clim["ivt_u"], clim["ivt_v"])
    return clim


# ================================================================ map furniture ==
def state_geoms():
    """Survey-of-India state polygons reprojected LCC -> WGS84 for cartopy."""
    tr = Transformer.from_crs(_LCC, _WGS84, always_xy=True)
    return [shp_transform(tr.transform, rec.geometry)
            for rec in shpreader.Reader(SOI_SHP).records()]


def base_ax(ax, extent, india_geoms=None, india_states=False):
    """Shared basemap furniture: extent, coastline, country borders, gridlabels.
       Optionally overlay Survey-of-India state polygons (only sensible when the
       extent is over India)."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, color="0.2")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, color="0.45")
    if india_states and india_geoms is not None:
        ax.add_geometries(india_geoms, crs=ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.3", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.7",
                      linestyle=":", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 10}
    return ax


def _shade(ax, da, **kw):
    """pcolormesh of a (lat, lon) DataArray (native grid)."""
    return ax.pcolormesh(da["lon"].values, da["lat"].values,
                         da.transpose("lat", "lon").values,
                         transform=ccrs.PlateCarree(), shading="auto", **kw)


def _box_outline(ax, box, color, label=None):
    """Draw a rectangle for an index averaging box on a map (PlateCarree)."""
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle((box["w"], box["s"]), box["e"] - box["w"],
                              box["n"] - box["s"], fill=False, edgecolor=color,
                              linewidth=2.0, transform=ccrs.PlateCarree(), zorder=6)
    ax.add_patch(rect)
    if label:
        ax.text(box["w"] + 0.5, box["n"] - 1.5, label, color=color, fontsize=9,
                fontweight="bold", transform=ccrs.PlateCarree(), zorder=7)


# ==================================================================== figure 1 ==
def fig_circulation(clim, geoms, years, out):
    """850 hPa winds (Somali jet) + 200 hPa winds (TEJ), shaded by wind speed."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    span = f"{years[0]}-{years[-1]}"
    skip = 2                                              # quiver thinning stride

    def _panel(ax, u, v, spd, title, vmax, cmap):
        base_ax(ax, MON_EXTENT, geoms, india_states=True)
        m = _shade(ax, spd, cmap=cmap, vmin=0, vmax=vmax)
        lon, lat = spd["lon"].values, spd["lat"].values
        ax.quiver(lon[::skip], lat[::skip],
                  u.transpose("lat", "lon").values[::skip, ::skip],
                  v.transpose("lat", "lon").values[::skip, ::skip],
                  transform=ccrs.PlateCarree(), scale=350, width=0.0025,
                  color="k", alpha=0.75)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")       # equator
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.06,
                          shrink=0.92, extend="max")
        cb.set_label("wind speed (m s$^{-1}$)")
        ax.set_title(title)
        return spd

    _panel(axes[0], clim["u850"], clim["v850"], clim["speed850"],
           "850 hPa wind: Somali low-level jet &\ncross-equatorial flow",
           vmax=float(np.ceil(np.nanmax(clim["speed850"].values))), cmap="YlGnBu")
    _panel(axes[1], clim["u200"], clim["v200"], clim["speed200"],
           "200 hPa wind: Tropical Easterly Jet",
           vmax=float(np.ceil(np.nanmax(clim["speed200"].values))), cmap="YlOrRd")

    fig.suptitle(f"JJAS mean monsoon circulation, ERA5 {span}", y=1.0)
    fig.tight_layout()
    p = os.path.join(out, "circulation_jjas.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    jet = float(np.nanmax(clim["speed850"].sel(
        lat=slice(0, 15), lon=slice(40, 60)).values))
    return p, jet


# ==================================================================== figure 2 ==
def fig_webster_yang(df, out):
    """Webster-Yang index time series with mean, linear trend, strong/weak years."""
    yrs = df.index.values.astype(int)
    wy = df["wy"].values
    mean, sd = wy.mean(), wy.std()
    sl, ic, r, p, se = stats.linregress(yrs, wy)
    trend = ic + sl * yrs
    strong = wy >= mean + WY_SIGMA * sd
    weak = wy <= mean - WY_SIGMA * sd

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.plot(yrs, wy, "-o", color="#1f4e79", lw=1.8, ms=5, label="WY index (U850-U200)")
    ax.axhline(mean, color="0.4", ls="--", lw=1.3, label=f"mean = {mean:.1f} m/s")
    ax.plot(yrs, trend, color="crimson", lw=2.2,
            label=f"trend = {sl*10:+.2f} m/s/decade (p={p:.2f})")
    ax.scatter(yrs[strong], wy[strong], s=140, facecolors="none",
               edgecolors="green", linewidths=2.2, zorder=5,
               label=f"strong (>+{WY_SIGMA:.0f}$\\sigma$)")
    ax.scatter(yrs[weak], wy[weak], s=140, facecolors="none",
               edgecolors="darkorange", linewidths=2.2, zorder=5,
               label=f"weak (<-{WY_SIGMA:.0f}$\\sigma$)")
    ax.set_xlabel("year")
    ax.set_ylabel("Webster-Yang index (m s$^{-1}$)")
    ax.set_title("Webster-Yang monsoon index (JJAS mean U850-U200, 0-20N 40-110E)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, framealpha=0.9, loc="best")
    fig.tight_layout()
    p_ = os.path.join(out, "webster_yang_index.png")
    fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p_, dict(mean=mean, trend_decade=sl * 10, p=p,
                    strong=yrs[strong].tolist(), weak=yrs[weak].tolist())


# ==================================================================== figure 3 ==
def fig_enso(df, rain_stack, geoms, years, out):
    """Nino3.4 vs ISMR scatter + correlation, and El Nino minus La Nina rainfall
       anomaly composite."""
    nino = df["nino34"].values
    ismr = df["ismr"].values
    r, p = stats.pearsonr(nino, ismr)
    sl, ic, *_ = stats.linregress(nino, ismr)

    # Composite: El Nino (nino34 > +thr) minus La Nina (nino34 < -thr).
    yrs = df.index.values.astype(int)
    en = yrs[df["nino34"].values > ENSO_THR]
    ln = yrs[df["nino34"].values < -ENSO_THR]
    clim = rain_stack.mean("year")
    comp = None
    if len(en) and len(ln):
        comp = (rain_stack.sel(year=list(en)).mean("year")
                - rain_stack.sel(year=list(ln)).mean("year"))

    fig = plt.figure(figsize=(20, 7.4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.scatter(nino, ismr, s=70, c=nino, cmap="RdBu_r", edgecolors="k",
                linewidths=0.5, vmin=-2, vmax=2, zorder=4)
    xx = np.linspace(nino.min(), nino.max(), 50)
    ax0.plot(xx, ic + sl * xx, color="k", lw=2.0,
             label=f"r = {r:+.2f} (p={p:.3f})\nslope = {sl:+.2f} mm/day per $^\\circ$C")
    ax0.axhline(ismr.mean(), color="0.6", ls=":", lw=1)
    ax0.axvline(0, color="0.6", ls=":", lw=1)
    ax0.set_xlabel("Nino-3.4 SST anomaly ($^\\circ$C)")
    ax0.set_ylabel("All-India JJAS rainfall (mm/day)")
    ax0.set_title("ENSO-monsoon: Nino-3.4 vs ISMR")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best", framealpha=0.9)

    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    base_ax(ax1, INDIA_EXTENT, geoms, india_states=True)
    if comp is not None:
        vmax = float(np.nanpercentile(np.abs(comp.values), 98))
        m = _shade(ax1, comp, cmap="BrBG", vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(m, ax=ax1, orientation="horizontal", pad=0.06,
                          shrink=0.9, extend="both")
        cb.set_label("rainfall anomaly (mm/day)")
        ax1.set_title(f"JJAS rainfall: El Nino - La Nina\n"
                      f"({len(en)} EN yrs - {len(ln)} LN yrs)")
    else:
        ax1.set_title("El Nino - La Nina composite\n(insufficient years in subset)")

    fig.suptitle(f"ENSO teleconnection, ERA5 {years[0]}-{years[-1]}", y=1.0)
    fig.tight_layout()
    p_ = os.path.join(out, "enso_teleconnection.png")
    fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p_, dict(r=r, p=p, slope=sl, n_elnino=len(en), n_lanina=len(ln),
                    elnino_years=en.tolist(), lanina_years=ln.tolist())


# ==================================================================== figure 4 ==
def fig_iod(df, out):
    """Dipole Mode Index vs ISMR scatter + correlation."""
    dmi = df["dmi"].values
    ismr = df["ismr"].values
    r, p = stats.pearsonr(dmi, ismr)
    sl, ic, *_ = stats.linregress(dmi, ismr)

    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.scatter(dmi, ismr, s=70, c=dmi, cmap="PuOr_r", edgecolors="k",
               linewidths=0.5, zorder=4)
    xx = np.linspace(dmi.min(), dmi.max(), 50)
    ax.plot(xx, ic + sl * xx, color="k", lw=2.0,
            label=f"r = {r:+.2f} (p={p:.3f})\nslope = {sl:+.2f} mm/day per $^\\circ$C")
    ax.axhline(ismr.mean(), color="0.6", ls=":", lw=1)
    ax.axvline(0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("Dipole Mode Index (DMI, $^\\circ$C)")
    ax.set_ylabel("All-India JJAS rainfall (mm/day)")
    ax.set_title("IOD-monsoon: DMI vs ISMR")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    p_ = os.path.join(out, "iod_teleconnection.png")
    fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p_, dict(r=r, p=p, slope=sl)


# ==================================================================== figure 5 ==
def fig_ivt_mslp(clim, geoms, years, out):
    """JJAS IVT magnitude (shaded) + 850 hPa moisture-flux vectors, and MSLP
       showing the monsoon trough."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.2),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    span = f"{years[0]}-{years[-1]}"
    skip = 2

    # --- IVT magnitude + vertically-integrated moisture-flux vectors --------
    ax = axes[0]
    base_ax(ax, MON_EXTENT, geoms, india_states=True)
    ivt = clim["ivt"]
    vmax = float(np.ceil(np.nanpercentile(ivt.values, 99) / 50) * 50)
    m = _shade(ax, ivt, cmap="GnBu", vmin=0, vmax=vmax)
    lon, lat = clim["ivt_u"]["lon"].values, clim["ivt_u"]["lat"].values
    ax.quiver(lon[::skip], lat[::skip],
              clim["ivt_u"].transpose("lat", "lon").values[::skip, ::skip],
              clim["ivt_v"].transpose("lat", "lon").values[::skip, ::skip],
              transform=ccrs.PlateCarree(), scale=6000, width=0.0025,
              color="k", alpha=0.75)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.06, shrink=0.92,
                      extend="max")
    cb.set_label("IVT magnitude (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title("JJAS moisture transport: vertically-\nintegrated vapour transport (IVT)")

    # --- MSLP: monsoon trough -----------------------------------------------
    ax = axes[1]
    base_ax(ax, MON_EXTENT, geoms, india_states=True)
    mslp = clim["mslp"]
    lo, hi = np.nanpercentile(mslp.values, [1, 99])
    m2 = _shade(ax, mslp, cmap="viridis", vmin=lo, vmax=hi)
    cs = ax.contour(mslp["lon"].values, mslp["lat"].values,
                    mslp.transpose("lat", "lon").values, colors="k",
                    linewidths=0.7, levels=12, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d")
    cb2 = fig.colorbar(m2, ax=ax, orientation="horizontal", pad=0.06, shrink=0.92)
    cb2.set_label("mean sea-level pressure (hPa)")
    ax.set_title("JJAS mean sea-level pressure:\nthe monsoon trough")

    fig.suptitle(f"JJAS moisture pathways & monsoon trough, ERA5 {span}", y=1.0)
    fig.tight_layout()
    p_ = os.path.join(out, "ivt_mslp_jjas.png")
    fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ivt_india = float(_box_mean(ivt, dict(s=8, n=25, w=70, e=90)))
    return p_, ivt_india


# ==================================================================== figure 6 ==
def fig_sst_ismr_corr(sst_stack, df, years, out):
    """Per-grid correlation of tropical Indo-Pacific JJAS SST anomaly with ISMR."""
    ismr = xr.DataArray(df["ismr"].values, dims="year",
                        coords={"year": df.index.values})
    ismr = ismr.sel(year=sst_stack["year"])
    sst_an = sst_stack - sst_stack.mean("year")
    ismr_an = ismr - ismr.mean("year")
    n = sst_stack["year"].size
    num = (sst_an * ismr_an).sum("year")
    den = np.sqrt((sst_an ** 2).sum("year") * (ismr_an ** 2).sum("year"))
    r = (num / den).where(den > 0)
    # two-sided p-value (t = r*sqrt((n-2)/(1-r^2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
    pmask = xr.apply_ufunc(lambda tt: 2 * stats.t.sf(np.abs(tt), n - 2), t) < 0.10

    fig = plt.figure(figsize=(18, 7.6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    base_ax(ax, INDOPAC_EXTENT)
    m = ax.pcolormesh(r["lon"].values, r["lat"].values,
                      r.transpose("lat", "lon").values, cmap="RdBu_r",
                      vmin=-0.8, vmax=0.8, shading="auto",
                      transform=ccrs.PlateCarree())
    # stipple significant cells
    LON, LAT = np.meshgrid(r["lon"].values, r["lat"].values)
    sig = pmask.transpose("lat", "lon").values
    ax.scatter(LON[sig], LAT[sig], s=1.5, color="k", alpha=0.4,
               transform=ccrs.PlateCarree())
    _box_outline(ax, NINO34, "k", "Nino3.4")
    cb = fig.colorbar(m, ax=ax, orientation="vertical", pad=0.02, shrink=0.8,
                      extend="both")
    cb.set_label("correlation r (SST anom vs ISMR)")
    ax.set_title(f"JJAS SST-ISMR teleconnection (ERA5 {years[0]}-{years[-1]}, "
                 f"n={n}; stipple p<0.10)")
    fig.tight_layout()
    p_ = os.path.join(out, "sst_ismr_corr_map.png")
    fig.savefig(p_, dpi=150, bbox_inches="tight")
    plt.close(fig)
    nino_r = float(_box_mean(r, NINO34))
    return p_, nino_r


# ========================================================================= main ==
def parse_years(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser(description="ERA5 monsoon dynamics & teleconnections")
    ap.add_argument("--years", default="1959-2022",
                    help="inclusive year range, e.g. 1990-2019")
    ap.add_argument("--quick", action="store_true",
                    help="fast subset: 1990-2019 (overrides --years)")
    ap.add_argument("--workers", type=int, default=8,
                    help="ProcessPoolExecutor workers for per-year reductions")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "monsoon_dynamics"),
                    help="output directory for PNGs")
    args = ap.parse_args()

    years = parse_years("1990-2019" if args.quick else args.years)
    os.makedirs(args.out, exist_ok=True)
    print(f"[monsoon_dynamics] years {years[0]}-{years[-1]} ({len(years)}), "
          f"{args.workers} workers, out={args.out}", flush=True)
    T0 = time.time()
    geoms = state_geoms()

    # ---- scalar indices (WY, Nino3.4, DMI, ISMR) ----
    print("[1/4] scalar JJAS indices ...", flush=True)
    df = compute_indices(years, args.workers)
    df.to_csv(os.path.join(args.out, "indices.csv"))

    # ---- field stacks (rainfall for composite, SST for corr map) ----
    print("[2/4] rainfall & SST field stacks ...", flush=True)
    rain_stack = compute_field_stack(year_rain_field, years, args.workers, "rain")
    sst_stack = compute_field_stack(year_sst_field, years, args.workers, "sst")

    # ---- circulation climatology ----
    print("[3/4] circulation climatology ...", flush=True)
    clim = circulation_climatology(years, args.workers)

    # ---- figures ----
    print("[4/4] figures ...", flush=True)
    paths, stats_out = {}, {}
    paths["circulation"], jet = fig_circulation(clim, geoms, years, args.out)
    paths["webster_yang"], wy = fig_webster_yang(df, args.out)
    paths["enso"], enso = fig_enso(df, rain_stack, geoms, years, args.out)
    paths["iod"], iod = fig_iod(df, args.out)
    paths["ivt_mslp"], ivt_india = fig_ivt_mslp(clim, geoms, years, args.out)
    paths["sst_corr"], nino_r = fig_sst_ismr_corr(sst_stack, df, years, args.out)

    dt = time.time() - T0
    print("\n" + "=" * 70)
    print(f"DONE in {dt:.1f}s  ({len(years)} yrs, {args.workers} workers)")
    print("=" * 70)
    print(f"  Somali-jet peak speed     : {jet:.1f} m/s   (expect ~15+)")
    print(f"  Webster-Yang mean         : {wy['mean']:.1f} m/s "
          f"(trend {wy['trend_decade']:+.2f}/dec, p={wy['p']:.2f})")
    print(f"    WY strong years         : {wy['strong']}")
    print(f"    WY weak years           : {wy['weak']}")
    print(f"  Nino3.4-ISMR correlation  : r={enso['r']:+.2f} "
          f"(p={enso['p']:.3f})   (expect negative ~ -0.5)")
    print(f"    El Nino yrs ({enso['n_elnino']}): {enso['elnino_years']}")
    print(f"    La Nina yrs ({enso['n_lanina']}): {enso['lanina_years']}")
    print(f"  DMI-ISMR correlation      : r={iod['r']:+.2f} (p={iod['p']:.3f})")
    print(f"  IVT into core India       : {ivt_india:.0f} kg/m/s")
    print(f"  Nino3.4-box mean r (map)  : {nino_r:+.2f}")
    print("-" * 70)
    for k, p in paths.items():
        print(f"  fig {k:14s}: {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
