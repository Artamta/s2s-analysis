#!/usr/bin/env python3
"""
make_forecast_animation.py
==========================
Publication-quality animated GIFs of FuXi-S2S 42-day forecast over India.
Designed for website display — dark theme, self-explanatory panels.

GIFs produced (in forecast_animation/):
  1. tp_animation.gif        — Precipitation forecast | Precipitation anomaly
  2. t2m_animation.gif       — 2m Temperature | T2m anomaly
  3. z500_wind_animation.gif — Z500 geopotential | Z500 anomaly + wind anomaly
  4. llj_animation.gif       — 850 hPa wind speed + streamlines (monsoon LLJ)
  5. olr_animation.gif       — OLR (convection proxy) | OLR anomaly
  6. moisture_animation.gif  — Precipitable water (TCWV) | TCWV anomaly

Climatology: WeatherBench2 ERA5 1990-2019 (tp, t2m, z500, tcwv, u850, v850)
             OLR anomaly vs model day-1 (WB2 has no ttr)

Usage
-----
  python make_forecast_animation.py                    # all GIFs
  python make_forecast_animation.py --mode tp          # one GIF only
  python make_forecast_animation.py --fps 4 --members 0
"""

import argparse
import datetime
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import xarray as xr
from PIL import Image

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
RAW_DIR  = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw/20260618")
OUT_DIR  = Path("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/forecast_animation")
SOI_SHP  = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"

INIT_DATE   = datetime.date(2026, 6, 18)
TOTAL_STEPS = 42
DEFAULT_FPS = 3

LAT_MIN, LAT_MAX = 5,  38
LON_MIN, LON_MAX = 65, 100

# ── SOI CRS ───────────────────────────────────────────────────────────────────
SOI_CRS = ccrs.LambertConformal(
    central_longitude=80.0, central_latitude=24.0,
    standard_parallels=(12.472944, 35.172806),
    false_easting=4000000.0, false_northing=4000000.0,
)
PROJ = ccrs.PlateCarree()

# ── CUSTOM COLORMAPS ──────────────────────────────────────────────────────────
# Monsoon rainfall — white→light blue→green→dark blue
TP_COLORS = ["#f7fbff","#c6dbef","#6baed6","#2171b5","#084594","#041c4a"]
TP_CMAP   = LinearSegmentedColormap.from_list("tp_monsoon", TP_COLORS, N=256)

# OLR — reversed (low OLR = deep convection = should be dark/purple)
OLR_CMAP  = "Purples_r"

# Wind speed — calm to strong (for LLJ)
LLJ_CMAP  = LinearSegmentedColormap.from_list(
    "llj", ["#f0f9e8","#bae4bc","#7bccc4","#43a2ca","#0868ac","#084081"], N=256)

# Moisture
TCWV_CMAP = "YlGnBu"

# Diverging anomaly
DIV_CMAP  = "RdBu_r"
BRG_CMAP  = "BrBG"        # for precip anomaly (brown=dry, green=wet)


# ── LOAD SOI SHAPEFILE ────────────────────────────────────────────────────────
def load_soi():
    try:
        reader = shpreader.Reader(SOI_SHP)
        geoms  = [r.geometry.simplify(5000) for r in reader.records()]
        print(f"  SOI: {len(geoms)} state polygons")
        return geoms
    except Exception as e:
        print(f"  WARNING: SOI failed ({e}), using cartopy")
        return None


def add_india_map(ax, soi_geoms, title, subtitle=None):
    """Base India map with ocean, land, borders, gridlines, title."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),  color="#1a3a5c", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),   color="#2d2d2d", zorder=0)

    if soi_geoms:
        ax.add_geometries(soi_geoms, crs=SOI_CRS,
                          facecolor="none", edgecolor="#aaaaaa",
                          linewidth=0.55, zorder=4)
    else:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       linewidth=0.35, edgecolor="#aaaaaa", zorder=4)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.9, edgecolor="#dddddd", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.7, edgecolor="#cccccc", linestyle="--", zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="#666666",
                      alpha=0.6, zorder=3, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.MultipleLocator(10)
    gl.ylocator   = mticker.MultipleLocator(10)
    gl.xlabel_style = {"size": 7, "color": "#aaaaaa"}
    gl.ylabel_style = {"size": 7, "color": "#aaaaaa"}

    # Panel title
    ax.set_title(title, fontsize=10, fontweight="bold", color="white",
                 pad=5, loc="center")
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5,
                color="#bbbbbb", style="italic")


def add_colorbar(fig, cf, ax, label, fmt="%.1f", ticks=None):
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.92, aspect=28)
    cb.set_label(label, fontsize=8, color="#cccccc")
    cb.ax.tick_params(labelsize=7, colors="#aaaaaa")
    cb.outline.set_edgecolor("#555555")
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def outline(ax, text, x, y, fs=8, color="white", bg="black", **kw):
    """Text with outline for readability on any background."""
    t = ax.text(x, y, text, fontsize=fs, color=color,
                transform=ax.transAxes, **kw)
    t.set_path_effects([pe.withStroke(linewidth=2, foreground=bg)])
    return t


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_step(step, channels, members):
    data = {ch: [] for ch in channels}
    lat = lon = None
    for mem in members:
        f = RAW_DIR / f"member/{mem:02d}/{step:02d}.nc"
        if not f.exists():
            continue
        da = xr.open_dataarray(str(f))
        lat = da.lat.values
        lon = da.lon.values
        for ch in channels:
            if ch in da.channel.values:
                data[ch].append(da.sel(channel=ch).squeeze(drop=True).values)
    out = {ch: np.array(v).mean(axis=0) for ch, v in data.items() if v}
    return out, lat, lon


def india_box(arr, lat, lon):
    lmask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    omask = (lon >= LON_MIN) & (lon <= LON_MAX)
    return arr[np.ix_(lmask, omask)], lat[lmask], lon[omask]


# ── WB2 CLIMATOLOGY CACHE ─────────────────────────────────────────────────────
WB2_MAP = {
    "t2m":  "t2m",
    "z500": "z500",
    "tp":   "tp24",    # WB2 tp24 in metres → ×1000 = mm
    "tcwv": "tcwv",
    "u850": "u850",
    "v850": "v850",
}


def build_climo_cache(lat, lon):
    print("  Pre-fetching WB2 1990-2019 climatology for all 42 lead days ...")
    try:
        from earth2studio.data import WB2Climatology
        wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr",
                             verbose=False)
    except Exception as e:
        print(f"  WARNING: WB2 unavailable ({e})")
        return None

    wb2_vars = list(dict.fromkeys(WB2_MAP.values()))
    cache    = {}

    for step in range(1, TOTAL_STEPS + 1):
        valid  = INIT_DATE + datetime.timedelta(days=step)
        t      = datetime.datetime(2001, valid.month, valid.day, 0)
        da     = wb2(t, wb2_vars)
        wb2_lat = da.lat.values
        wb2_lon = da.lon.values

        step_c = {}
        for model_ch, wb2_var in WB2_MAP.items():
            arr = da.sel(variable=wb2_var).squeeze().values.astype(np.float32)
            if wb2_var == "tp24":
                arr = arr * 1000.0          # m → mm
            if wb2_var == "z500":
                arr = arr / 9.80665         # m²/s² → gpm
            da_xr   = xr.DataArray(arr, dims=["lat","lon"],
                                   coords={"lat": wb2_lat, "lon": wb2_lon})
            step_c[model_ch] = da_xr.interp(lat=lat, lon=lon,
                                            method="linear").values
        cache[step] = step_c
        if step % 14 == 0:
            print(f"    climo cached through lead day {step}")

    print("  Climatology ready.\n")
    return cache


# ── FRAME FUNCTIONS ───────────────────────────────────────────────────────────

def frame_tp(step, data, climo, lat, lon, soi, valid_date):
    tp, lat_i, lon_i = india_box(data["tp"], lat, lon)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: forecast ────────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="Precipitation Forecast",
                  subtitle="Daily mean rainfall rate (mm/day)")
    levs = [0, 0.5, 1, 2, 4, 6, 8, 12, 18, 25]
    cf   = ax.contourf(lon_i, lat_i, tp, levels=levs,
                       cmap=TP_CMAP, transform=PROJ, extend="max", zorder=1)
    add_colorbar(fig, cf, ax, "mm / day", ticks=levs)
    outline(ax, "Higher values = more rainfall", 0.02, 0.02, fs=7, color="#aaddff")

    # ── Right: anomaly ────────────────────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="Precipitation Anomaly vs Climatology",
                  subtitle="Departure from ERA5 1990–2019 daily mean (mm/day)")
    if climo and step in climo and "tp" in climo[step]:
        tp_c = india_box(climo[step]["tp"], lat, lon)[0]
        anom = tp - tp_c
        lim  = max(2.5, float(np.nanpercentile(np.abs(anom), 97)))
        levs_a = np.linspace(-lim, lim, 21)
        cf2  = ax.contourf(lon_i, lat_i, anom, levels=levs_a,
                           cmap=BRG_CMAP, transform=PROJ, extend="both", zorder=1)
        add_colorbar(fig, cf2, ax, "mm/day anomaly")
        outline(ax, "Green = wetter than normal   Brown = drier than normal",
                0.02, 0.02, fs=7, color="#aaffaa")
    return fig


def frame_t2m(step, data, climo, lat, lon, soi, valid_date):
    t2m_K, lat_i, lon_i = india_box(data["t2m"], lat, lon)
    t2m_C = t2m_K - 273.15

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: absolute T2m ────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="2m Temperature Forecast",
                  subtitle="Near-surface air temperature (°C)")
    levs = np.arange(-4, 44, 2)
    cf   = ax.contourf(lon_i, lat_i, t2m_C, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, t2m_C, levels=levs[::3],
                      colors="white", linewidths=0.3, alpha=0.4,
                      transform=PROJ, zorder=2)
    ax.clabel(cs, fmt="%d°", fontsize=6, inline=True, colors="white")
    add_colorbar(fig, cf, ax, "°C")
    outline(ax, "Blue = cool   Red = warm", 0.02, 0.02, fs=7, color="#ffddaa")

    # ── Right: anomaly ────────────────────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="T2m Anomaly vs Climatology",
                  subtitle="Departure from ERA5 1990–2019 daily mean (°C)")
    if climo and step in climo and "t2m" in climo[step]:
        t2m_c = india_box(climo[step]["t2m"], lat, lon)[0] - 273.15
        anom  = t2m_C - t2m_c
        lim   = max(1.5, float(np.nanpercentile(np.abs(anom), 97)))
        cf2   = ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        cs2   = ax.contour(lon_i, lat_i, anom, levels=[-2,-1,1,2],
                           colors="white", linewidths=0.3, alpha=0.5,
                           transform=PROJ, zorder=2)
        ax.clabel(cs2, fmt="%+.0f°", fontsize=6, colors="white")
        add_colorbar(fig, cf2, ax, "°C anomaly")
        outline(ax, "Red = warmer than normal   Blue = cooler than normal",
                0.02, 0.02, fs=7, color="#ffaaaa")
    return fig


def frame_z500(step, data, climo, lat, lon, soi, valid_date):
    z_gpm, lat_i, lon_i = india_box(data["z500"] / 9.80665, lat, lon)
    u850,  _,     _     = india_box(data["u850"],            lat, lon)
    v850,  _,     _     = india_box(data["v850"],            lat, lon)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: Z500 + wind ─────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="Z500 Geopotential + 850 hPa Wind",
                  subtitle="Mid-level circulation (gpm) and low-level wind (m/s)")
    levs = np.arange(5740, 5920, 10)
    cf   = ax.contourf(lon_i, lat_i, z_gpm, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, z_gpm, levels=levs[::2],
                      colors="white", linewidths=0.35, alpha=0.5,
                      transform=PROJ, zorder=2)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white", inline=True)
    skip = 2
    ax.quiver(lon_i[::skip], lat_i[::skip],
              u850[::skip,::skip], v850[::skip,::skip],
              transform=PROJ, scale=200, width=0.003,
              color="#ffffff", alpha=0.85, zorder=6,
              headwidth=4, headlength=4)
    add_colorbar(fig, cf, ax, "gpm")
    outline(ax, "Arrows = 850 hPa wind direction & speed", 0.02, 0.02, fs=7)

    # ── Right: anomaly ────────────────────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="Z500 Anomaly + Wind Anomaly vs Climatology",
                  subtitle="Departure from ERA5 1990–2019 daily mean")
    if climo and step in climo and "z500" in climo[step]:
        zc    = india_box(climo[step]["z500"], lat, lon)[0]
        uc    = india_box(climo[step]["u850"], lat, lon)[0]
        vc    = india_box(climo[step]["v850"], lat, lon)[0]
        zanom = z_gpm - zc
        uanom = u850 - uc
        vanom = v850 - vc
        lim   = max(10, float(np.nanpercentile(np.abs(zanom), 95)))
        cf2   = ax.contourf(lon_i, lat_i, zanom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        cs2   = ax.contour(lon_i, lat_i, zanom,
                           levels=[-20,-10,10,20],
                           colors="white", linewidths=0.3, alpha=0.4,
                           transform=PROJ, zorder=2)
        ax.clabel(cs2, fmt="%+.0f", fontsize=6, colors="white")
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  uanom[::skip,::skip], vanom[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="#ffffff", alpha=0.8, zorder=6,
                  headwidth=4, headlength=4)
        add_colorbar(fig, cf2, ax, "gpm anomaly")
        outline(ax, "Red = high pressure anomaly   Blue = low pressure anomaly",
                0.02, 0.02, fs=7, color="#aaaaff")
    return fig


def frame_llj(step, data, climo, lat, lon, soi, valid_date):
    """850 hPa wind speed + streamlines — monsoon low-level jet."""
    u850, lat_i, lon_i = india_box(data["u850"], lat, lon)
    v850, _,     _     = india_box(data["v850"], lat, lon)
    wspd = np.sqrt(u850**2 + v850**2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: wind speed ──────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="850 hPa Wind Speed (Monsoon Low-Level Jet)",
                  subtitle="Low-level wind speed (m/s) — key monsoon circulation driver")
    levs = np.arange(0, 22, 2)
    cf   = ax.contourf(lon_i, lat_i, wspd, levels=levs,
                       cmap=LLJ_CMAP, transform=PROJ, extend="max", zorder=1)
    # Streamlines — needs regular grid (already is)
    try:
        ax.streamplot(lon_i, lat_i, u850, v850,
                      transform=PROJ, color="#ffffff",
                      linewidth=0.6, density=1.5, arrowsize=0.8,
                      zorder=5)
    except Exception:
        skip = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u850[::skip,::skip], v850[::skip,::skip],
                  transform=PROJ, scale=180, width=0.003,
                  color="white", alpha=0.8, zorder=5)
    add_colorbar(fig, cf, ax, "m/s", ticks=levs)
    outline(ax, "Strong westerlies over Arabian Sea = active monsoon LLJ",
            0.02, 0.02, fs=7, color="#aaffaa")

    # ── Right: wind speed anomaly ─────────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="850 hPa Wind Speed Anomaly vs Climatology",
                  subtitle="Departure from ERA5 1990–2019 daily mean (m/s)")
    if climo and step in climo and "u850" in climo[step]:
        uc   = india_box(climo[step]["u850"], lat, lon)[0]
        vc   = india_box(climo[step]["v850"], lat, lon)[0]
        wc   = np.sqrt(uc**2 + vc**2)
        anom = wspd - wc
        lim  = max(2, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        # Wind anomaly vectors
        uanom = u850 - uc
        vanom = v850 - vc
        skip  = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  uanom[::skip,::skip], vanom[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="white", alpha=0.8, zorder=5,
                  headwidth=4, headlength=4)
        add_colorbar(fig, cf2, ax, "m/s anomaly")
        outline(ax, "Green/teal = stronger winds than normal (enhanced monsoon)",
                0.02, 0.02, fs=7, color="#aaffaa")
    return fig


def frame_olr(step, data, climo_day1, lat, lon, soi, valid_date):
    """OLR (ttr) — proxy for deep convection. Anomaly vs model day-1."""
    # ttr in model is W/m² (negative = outgoing, divide by -1 for OLR convention)
    ttr, lat_i, lon_i = india_box(data["ttr"], lat, lon)
    olr = -ttr   # positive OLR = energy leaving atmosphere

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: OLR ─────────────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="Outgoing Longwave Radiation (OLR)",
                  subtitle="Proxy for deep convection — low OLR = active convection/rainfall (W/m²)")
    levs = np.arange(100, 320, 20)
    cf   = ax.contourf(lon_i, lat_i, olr, levels=levs,
                       cmap=OLR_CMAP, transform=PROJ, extend="both", zorder=1)
    add_colorbar(fig, cf, ax, "W/m²", ticks=levs[::2])
    outline(ax, "Dark purple = intense convection/heavy rain   Light = clear sky",
            0.02, 0.02, fs=7, color="#ddaaff")

    # ── Right: OLR anomaly vs day-1 ──────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="OLR Anomaly vs Day-1 Forecast",
                  subtitle="Departure from Day-1 OLR (W/m²) — negative = more convection than day-1")
    if climo_day1 is not None:
        olr_d1 = -climo_day1
        anom   = olr - olr_d1
        lim    = max(20, float(np.nanpercentile(np.abs(anom), 97)))
        cf2    = ax.contourf(lon_i, lat_i, anom,
                             levels=np.linspace(-lim, lim, 21),
                             cmap="PuOr_r", transform=PROJ, extend="both", zorder=1)
        add_colorbar(fig, cf2, ax, "W/m² anomaly")
        outline(ax, "Purple = more convection than initial day   Orange = less convection",
                0.02, 0.02, fs=7, color="#ddaaff")
    return fig


def frame_moisture(step, data, climo, lat, lon, soi, valid_date):
    """Total column water vapour (TCWV) — atmospheric moisture reservoir."""
    tcwv, lat_i, lon_i = india_box(data["tcwv"], lat, lon)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="#121212")

    # ── Left: TCWV ────────────────────────────────────────────────────────────
    ax = axes[0]
    add_india_map(ax, soi,
                  title="Total Column Water Vapour (TCWV)",
                  subtitle="Total atmospheric moisture content (kg/m²) — fuel for monsoon rainfall")
    levs = np.arange(10, 75, 5)
    cf   = ax.contourf(lon_i, lat_i, tcwv, levels=levs,
                       cmap=TCWV_CMAP, transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, tcwv, levels=[40, 50, 60],
                      colors="white", linewidths=0.4, alpha=0.5, transform=PROJ)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white")
    add_colorbar(fig, cf, ax, "kg/m²", ticks=levs[::2])
    outline(ax, "Deep blue = very moist atmosphere → potential for heavy rain",
            0.02, 0.02, fs=7, color="#aaddff")

    # ── Right: TCWV anomaly ───────────────────────────────────────────────────
    ax = axes[1]
    add_india_map(ax, soi,
                  title="TCWV Anomaly vs Climatology",
                  subtitle="Departure from ERA5 1990–2019 daily mean (kg/m²)")
    if climo and step in climo and "tcwv" in climo[step]:
        tcwv_c = india_box(climo[step]["tcwv"], lat, lon)[0]
        anom   = tcwv - tcwv_c
        lim    = max(3, float(np.nanpercentile(np.abs(anom), 97)))
        cf2    = ax.contourf(lon_i, lat_i, anom,
                             levels=np.linspace(-lim, lim, 21),
                             cmap=BRG_CMAP, transform=PROJ, extend="both", zorder=1)
        add_colorbar(fig, cf2, ax, "kg/m² anomaly")
        outline(ax, "Green = more moisture than normal   Brown = drier atmosphere",
                0.02, 0.02, fs=7, color="#aaffaa")
    return fig


# ── FIGURE FOOTER + HEADER ────────────────────────────────────────────────────
def add_header_footer(fig, step, valid_date):
    init_str  = f"FuXi-S2S  ·  Initialised: {INIT_DATE.strftime('%d %b %Y')}"
    lead_str  = (f"Lead Day {step:02d}  ·  Valid: {valid_date.strftime('%d %b %Y')}"
                 f"  ·  Ensemble Control Run")
    foot_str  = ("FuXi-S2S deep learning model  ·  "
                 "Anomalies vs WeatherBench2 ERA5 1990–2019 climatology  ·  "
                 "Grid: 1.5° global")

    fig.text(0.5, 0.985, init_str, ha="center", va="top",
             fontsize=10, color="#aaaaaa", fontweight="bold")
    fig.text(0.5, 0.965, lead_str, ha="center", va="top",
             fontsize=13, color="#f0c040", fontweight="bold")
    fig.text(0.5, 0.012, foot_str, ha="center", va="bottom",
             fontsize=7, color="#666666", style="italic")


def render_frame(fig, step, valid_date):
    add_header_footer(fig, step, valid_date)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.10, wspace=0.06)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return img


# ── MODE CONFIG ───────────────────────────────────────────────────────────────
MODES = {
    "tp":       dict(channels=["tp"],                              label="Precipitation"),
    "t2m":      dict(channels=["t2m"],                            label="Temperature"),
    "z500":     dict(channels=["z500","u850","v850"],              label="Z500 + Wind"),
    "llj":      dict(channels=["u850","v850"],                     label="LLJ / 850hPa Wind"),
    "olr":      dict(channels=["ttr"],                             label="OLR / Convection"),
    "moisture": dict(channels=["tcwv"],                            label="Moisture (TCWV)"),
}

FRAME_FNS = {
    "tp":       frame_tp,
    "t2m":      frame_t2m,
    "z500":     frame_z500,
    "llj":      frame_llj,
    "olr":      frame_olr,
    "moisture": frame_moisture,
}


def make_gif(mode, members, fps, climo, soi, olr_day1=None):
    cfg = MODES[mode]
    print(f"\n{'='*55}")
    print(f"  {cfg['label']} animation  ({TOTAL_STEPS} frames @ {fps} fps)")
    print(f"{'='*55}")

    frames = []
    fn = FRAME_FNS[mode]

    for step in range(1, TOTAL_STEPS + 1):
        data, lat, lon = load_step(step, cfg["channels"], members)
        if not data:
            continue

        valid = INIT_DATE + datetime.timedelta(days=step)

        if mode == "olr":
            fig = fn(step, data, olr_day1, lat, lon, soi, valid)
        else:
            fig = fn(step, data, climo, lat, lon, soi, valid)

        frames.append(render_frame(fig, step, valid))

        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{TOTAL_STEPS} done")

    if not frames:
        print("  No frames — check data path")
        return

    out = OUT_DIR / f"{mode}_animation.gif"
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {out}  ({out.stat().st_size/1024**2:.1f} MB)")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default=None, choices=list(MODES.keys()))
    parser.add_argument("--members", type=int, nargs="+", default=[0])
    parser.add_argument("--fps",     type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output  : {OUT_DIR}")
    print(f"Members : {args.members}  |  FPS: {args.fps}")

    soi = load_soi()

    # Load lat/lon reference from step 1
    _, lat, lon = load_step(1, ["t2m"], args.members)
    if lat is None:
        print("ERROR: no model output found in", RAW_DIR)
        sys.exit(1)

    # Build WB2 climo cache
    climo = build_climo_cache(lat, lon)

    # OLR day-1 reference for anomaly
    olr_day1_data, _, _ = load_step(1, ["ttr"], args.members)
    olr_day1 = india_box(olr_day1_data["ttr"], lat, lon)[0] if "ttr" in olr_day1_data else None

    modes = [args.mode] if args.mode else list(MODES.keys())
    for m in modes:
        make_gif(m, args.members, args.fps, climo, soi,
                 olr_day1=olr_day1 if m == "olr" else None)

    print(f"\nAll done!  GIFs → {OUT_DIR}")


if __name__ == "__main__":
    main()
