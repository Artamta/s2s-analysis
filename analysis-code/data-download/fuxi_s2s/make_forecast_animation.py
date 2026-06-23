#!/usr/bin/env python3
"""
make_forecast_animation.py
==========================
Publication-quality animated GIFs of FuXi-S2S 42-day forecast over India.
Dark theme, self-explanatory — designed for website display.

GIFs produced:
  tp_animation.gif        Precipitation forecast | anomaly vs ERA5 climo
  t2m_animation.gif       2m Temperature | anomaly vs ERA5 climo
  z500_animation.gif      Z500 geopotential + 850hPa wind | anomaly
  llj_animation.gif       850hPa wind speed + streamlines | anomaly
  olr_animation.gif       OLR (convection) | anomaly vs model day-1
  moisture_animation.gif  TCWV (precipitable water) | anomaly vs ERA5 climo

Climatology: WeatherBench2 ERA5 1990-2019
             OLR anomaly uses model day-1 as reference (WB2 has no ttr)

Usage
-----
  # basic — uses defaults from CONFIG dict below
  python make_forecast_animation.py

  # single mode
  python make_forecast_animation.py --mode tp

  # different date / paths
  python make_forecast_animation.py \\
      --date 20260618 \\
      --raw_dir /storage/raj.ayush/All_Model_Data/fuxi/test/raw \\
      --out_dir /path/to/output \\
      --fps 3 --members 0
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
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import xarray as xr
from PIL import Image

warnings.filterwarnings("ignore")

# ── DEFAULT CONFIG (override via CLI args) ────────────────────────────────────
CONFIG = dict(
    date       = "20260618",
    raw_dir    = "/storage/raj.ayush/All_Model_Data/fuxi/test/raw",
    out_dir    = ("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/"
                  "data-download/fuxi_s2s/forecast_animation"),
    soi_shp    = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp",
    members    = [0],
    fps        = 3,
    total_steps= 42,
    lat_min    = 5,   lat_max = 38,
    lon_min    = 65,  lon_max = 100,
)

# SOI shapefile CRS (Lambert Conformal Conic, Survey of India)
SOI_CRS = ccrs.LambertConformal(
    central_longitude=80.0, central_latitude=24.0,
    standard_parallels=(12.472944, 35.172806),
    false_easting=4000000.0, false_northing=4000000.0,
)
PROJ = ccrs.PlateCarree()

# ── COLORMAPS ─────────────────────────────────────────────────────────────────
# Precipitation: black→deep-green→teal→cyan→white at high end
# Vivid even at low values (0.1–2 mm/day), unlike pale blues
TP_CMAP = LinearSegmentedColormap.from_list("tp_vivid", [
    "#0d0d0d",   # near-zero: almost black
    "#005824",   # 0.3 mm: deep forest green
    "#238b45",   # 0.8 mm: medium green
    "#41ae76",   # 1.5 mm: teal-green
    "#66c2a4",   # 2.5 mm: teal
    "#2ca25f",   # 4 mm: saturated green (re-deepen)
    "#006d2c",   # 6 mm: dark green
    "#00441b",   # 8+ mm: very dark
    "#084594",   # 12+ mm: deep blue (very heavy)
    "#041c4a",   # 20+ mm: midnight blue
], N=256)

# OLR: low OLR (dark convective cloud) → dark purple; high OLR (clear) → yellow
OLR_CMAP = LinearSegmentedColormap.from_list("olr_conv", [
    "#2d004b",   # deep convection
    "#7b3294",
    "#c2a5cf",
    "#f7f7f7",   # neutral
    "#a6dba0",
    "#008837",
    "#f4a582",
    "#d73027",   # subsidence/clear sky
], N=256)

# 850 hPa wind speed: white-calm → deep-blue-strong
LLJ_CMAP = LinearSegmentedColormap.from_list("llj", [
    "#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#084594", "#08306b",
], N=256)

TCWV_CMAP = "YlGnBu"
DIV_CMAP  = "RdBu_r"
PRECIP_ANOM_CMAP = "BrBG"


# ── SHAPEFILE ─────────────────────────────────────────────────────────────────
def load_soi(shp_path):
    try:
        reader = shpreader.Reader(shp_path)
        geoms  = [r.geometry.simplify(5000) for r in reader.records()]
        print(f"  SOI: {len(geoms)} state polygons loaded")
        return geoms
    except Exception as e:
        print(f"  WARNING: SOI shapefile failed ({e}) — using cartopy states")
        return None


# ── BASE MAP ──────────────────────────────────────────────────────────────────
def base_map(ax, cfg, soi_geoms, title):
    """Draw base India map with borders, gridlines, and a clean title badge."""
    ax.set_extent([cfg["lon_min"], cfg["lon_max"],
                   cfg["lat_min"], cfg["lat_max"]], crs=PROJ)

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), color="#0d1b2a", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),  color="#1e1e1e", zorder=0)

    if soi_geoms:
        ax.add_geometries(soi_geoms, crs=SOI_CRS,
                          facecolor="none", edgecolor="#888888",
                          linewidth=0.5, zorder=4)
    else:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor="#888888", linewidth=0.35, zorder=4)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#dddddd", linewidth=0.9, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#bbbbbb", linewidth=0.65,
                   linestyle="--", zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.18, color="#444444",
                      alpha=0.7, linestyle=":", zorder=3)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(10)
    gl.ylocator = mticker.MultipleLocator(10)
    gl.xlabel_style = {"size": 7, "color": "#999999"}
    gl.ylabel_style = {"size": 7, "color": "#999999"}

    # Title badge — inside axes, top-centre, never overlaps figure header
    ax.set_title("")
    ax.text(0.5, 0.985, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#0a0a1a",
                      edgecolor="#3355aa", alpha=0.92, linewidth=1.0),
            zorder=10)


def colorbar(fig, cf, ax, label, ticks=None):
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.92, aspect=30)
    cb.set_label(label, fontsize=8.5, color="#cccccc", labelpad=4)
    cb.ax.tick_params(labelsize=7.5, colors="#aaaaaa", length=3)
    cb.outline.set_edgecolor("#444444")
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


# ── DATA I/O ──────────────────────────────────────────────────────────────────
def load_step(raw_dir, date_str, step, channels, members):
    """Load ensemble-mean for one lead step. Returns (data_dict, lat, lon)."""
    data = {ch: [] for ch in channels}
    lat = lon = None
    for mem in members:
        f = Path(raw_dir) / date_str / f"member/{mem:02d}/{step:02d}.nc"
        if not f.exists():
            continue
        da  = xr.open_dataarray(str(f))
        lat = da.lat.values
        lon = da.lon.values
        for ch in channels:
            if ch in da.channel.values:
                data[ch].append(da.sel(channel=ch).squeeze(drop=True).values)
    out = {ch: np.array(v).mean(axis=0) for ch, v in data.items() if v}
    return out, lat, lon


def box(arr, lat, lon, cfg):
    lm = (lat >= cfg["lat_min"]) & (lat <= cfg["lat_max"])
    om = (lon >= cfg["lon_min"]) & (lon <= cfg["lon_max"])
    return arr[np.ix_(lm, om)], lat[lm], lon[om]


# ── WB2 CLIMATOLOGY ───────────────────────────────────────────────────────────
WB2_MAP = {
    "t2m":  "t2m",
    "z500": "z500",
    "tp":   "tp24",   # WB2 tp24 in metres → ×1000 → mm/day
    "tcwv": "tcwv",
    "u850": "u850",
    "v850": "v850",
}


def build_climo_cache(init_date, total_steps, lat, lon):
    print("  Building WB2 1990-2019 climatology cache …")
    try:
        from earth2studio.data import WB2Climatology
        wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr",
                             verbose=False)
    except Exception as e:
        print(f"  WARNING: WB2 unavailable ({e}) — anomalies will be skipped")
        return None

    wb2_vars = list(dict.fromkeys(WB2_MAP.values()))
    cache    = {}

    for step in range(1, total_steps + 1):
        valid = init_date + datetime.timedelta(days=step)
        t     = datetime.datetime(2001, valid.month, valid.day, 0)
        da    = wb2(t, wb2_vars)
        wlat  = da.lat.values
        wlon  = da.lon.values

        step_c = {}
        for mch, wvar in WB2_MAP.items():
            arr = da.sel(variable=wvar).squeeze().values.astype(np.float32)
            if wvar == "tp24":
                arr = arr * 1000.0        # m → mm
            if wvar == "z500":
                arr = arr / 9.80665       # m²/s² → gpm
            xda  = xr.DataArray(arr, dims=["lat","lon"],
                                 coords={"lat": wlat, "lon": wlon})
            step_c[mch] = xda.interp(lat=lat, lon=lon, method="linear").values
        cache[step] = step_c

        if step % 14 == 0:
            print(f"    … cached lead days 1–{step}")

    print("  Climatology ready.\n")
    return cache


# ── FIGURE HEADER / FOOTER ────────────────────────────────────────────────────
# Each panel pair has a description dict:  {left_desc, right_desc, units_left, units_right}
# These go into the footer strip so the map faces stay clean.

PANEL_DESC = {
    "tp": dict(
        left ="Precipitation forecast  ·  daily mean rate  ·  mm/day",
        right="Precipitation anomaly vs ERA5 1990–2019 climatology  ·  "
              "green = wetter than normal,  brown = drier than normal  ·  mm/day",
    ),
    "t2m": dict(
        left ="2m temperature forecast  ·  near-surface air temperature  ·  °C",
        right="T2m anomaly vs ERA5 1990–2019 climatology  ·  "
              "red = warmer than normal,  blue = cooler than normal  ·  °C",
    ),
    "z500": dict(
        left ="500 hPa geopotential height + 850 hPa wind  ·  "
              "mid-troposphere circulation  ·  gpm  |  wind arrows m/s",
        right="Z500 anomaly vs ERA5 1990–2019 climatology  ·  "
              "red = high-pressure anomaly,  blue = low-pressure anomaly  ·  gpm",
    ),
    "llj": dict(
        left ="850 hPa wind speed + streamlines  ·  Monsoon Low-Level Jet  ·  m/s  "
              "|  strong westerlies over Arabian Sea = active LLJ",
        right="850 hPa wind speed anomaly vs ERA5 1990–2019  ·  "
              "blue = stronger winds,  red = weaker winds  ·  m/s",
    ),
    "olr": dict(
        left ="Outgoing Longwave Radiation (OLR)  ·  proxy for deep convection  ·  W/m²  "
              "|  low OLR (dark purple) = active convection/heavy rain",
        right="OLR anomaly vs model Day-1  ·  "
              "purple = more convection than Day-1,  orange = less  ·  W/m²",
    ),
    "moisture": dict(
        left ="Total Column Water Vapour (TCWV)  ·  atmospheric moisture  ·  kg/m²  "
              "|  high TCWV = fuel for heavy rainfall",
        right="TCWV anomaly vs ERA5 1990–2019 climatology  ·  "
              "green = more moisture than normal,  brown = drier  ·  kg/m²",
    ),
}


def add_frame_text(fig, mode, step, init_date, valid_date):
    """Write header + footer text onto figure. Called after subplots_adjust."""
    desc = PANEL_DESC[mode]

    # ── Header: model info (small, grey) + lead/valid (large, gold) ──────────
    fig.text(0.5, 0.980,
             f"FuXi-S2S   ·   Initialised: {init_date.strftime('%d %b %Y')}   ·   "
             f"Ensemble control run",
             ha="center", va="top", fontsize=8.5, color="#777777")
    fig.text(0.5, 0.960,
             f"Lead Day {step:02d}   ·   Valid:  "
             f"{valid_date.strftime('%A, %d %B %Y')}",
             ha="center", va="top", fontsize=15, color="#f0c040",
             fontweight="bold")

    # ── Footer: two-line description (left panel | right panel) ──────────────
    # Line 1: left panel description
    fig.text(0.5, 0.072, f"Left:   {desc['left']}",
             ha="center", va="top", fontsize=7.2, color="#999999", style="italic")
    # Line 2: right panel description
    fig.text(0.5, 0.053, f"Right:  {desc['right']}",
             ha="center", va="top", fontsize=7.2, color="#999999", style="italic")
    # Line 3: model/climo credit
    fig.text(0.5, 0.018,
             "FuXi-S2S deep learning S2S model  ·  "
             "Anomalies vs WeatherBench2 ERA5 1990–2019 climatology  ·  "
             "1.5° global grid",
             ha="center", va="bottom", fontsize=6.5, color="#555555")


def render_frame(fig, mode, step, init_date, valid_date):
    # Taller figure, generous margins so footer has room for 3 text lines
    plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.17, wspace=0.08)
    add_frame_text(fig, mode, step, init_date, valid_date)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return img


# ── FRAME BUILDERS ────────────────────────────────────────────────────────────

def frame_tp(step, data, climo, cfg, soi, init_date, valid_date):
    tp, lat_i, lon_i = box(data["tp"], *_latlon(data, cfg), cfg)

    fig, axes = _fig2()

    # Left: forecast
    ax = axes[0]
    base_map(ax, cfg, soi, "Precipitation  [mm/day]")
    levs = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    cf   = ax.contourf(lon_i, lat_i, tp, levels=levs,
                       cmap=TP_CMAP, transform=PROJ, extend="max", zorder=1)
    colorbar(fig, cf, ax, "mm / day", ticks=levs)

    # Right: anomaly
    ax = axes[1]
    base_map(ax, cfg, soi, "Precipitation Anomaly  [mm/day]")
    if climo and step in climo and "tp" in climo[step]:
        tp_c = box(climo[step]["tp"], *_latlon_from_climo(climo, step, "tp", cfg), cfg)[0]
        anom = tp - tp_c
        lim  = max(2.0, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap=PRECIP_ANOM_CMAP, transform=PROJ,
                           extend="both", zorder=1)
        colorbar(fig, cf2, ax, "mm/day")

    return render_frame(fig, "tp", step, init_date, valid_date)


def frame_t2m(step, data, climo, cfg, soi, init_date, valid_date):
    lat, lon    = _latlon(data, cfg)
    t2m_K, lat_i, lon_i = box(data["t2m"], lat, lon, cfg)
    t2m_C = t2m_K - 273.15

    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "2m Temperature  [°C]")
    levs = np.arange(-4, 46, 2)
    cf   = ax.contourf(lon_i, lat_i, t2m_C, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, t2m_C, levels=levs[::4],
                      colors="white", linewidths=0.3, alpha=0.35, transform=PROJ)
    ax.clabel(cs, fmt="%d°", fontsize=6, inline=True, colors="white")
    colorbar(fig, cf, ax, "°C")

    ax = axes[1]
    base_map(ax, cfg, soi, "T2m Anomaly vs Climatology  [°C]")
    if climo and step in climo and "t2m" in climo[step]:
        t2m_c = box(climo[step]["t2m"], lat, lon, cfg)[0] - 273.15
        anom  = t2m_C - t2m_c
        lim   = max(1.5, float(np.nanpercentile(np.abs(anom), 97)))
        cf2   = ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        colorbar(fig, cf2, ax, "°C")

    return render_frame(fig, "t2m", step, init_date, valid_date)


def frame_z500(step, data, climo, cfg, soi, init_date, valid_date):
    lat, lon  = _latlon(data, cfg)
    z_gpm, lat_i, lon_i = box(data["z500"] / 9.80665, lat, lon, cfg)
    u850  = box(data["u850"], lat, lon, cfg)[0]
    v850  = box(data["v850"], lat, lon, cfg)[0]

    fig, axes = _fig2()
    skip = 2

    ax = axes[0]
    base_map(ax, cfg, soi, "Z500  [gpm]  +  850 hPa Wind  [m/s]")
    levs = np.arange(5720, 5940, 10)
    cf   = ax.contourf(lon_i, lat_i, z_gpm, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, z_gpm, levels=levs[::2],
                      colors="white", linewidths=0.3, alpha=0.4, transform=PROJ)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white", inline=True)
    ax.quiver(lon_i[::skip], lat_i[::skip],
              u850[::skip,::skip], v850[::skip,::skip],
              transform=PROJ, scale=200, width=0.003,
              color="#ffffffcc", zorder=6, headwidth=4, headlength=4)
    colorbar(fig, cf, ax, "gpm")

    ax = axes[1]
    base_map(ax, cfg, soi, "Z500 Anomaly  [gpm]  +  Wind Anomaly")
    if climo and step in climo and "z500" in climo[step]:
        zc    = box(climo[step]["z500"], lat, lon, cfg)[0]
        uc    = box(climo[step]["u850"], lat, lon, cfg)[0]
        vc    = box(climo[step]["v850"], lat, lon, cfg)[0]
        zanom = z_gpm - zc
        lim   = max(10, float(np.nanpercentile(np.abs(zanom), 95)))
        cf2   = ax.contourf(lon_i, lat_i, zanom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  (u850-uc)[::skip,::skip], (v850-vc)[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="#ffffffbb", zorder=6, headwidth=4, headlength=4)
        colorbar(fig, cf2, ax, "gpm")

    return render_frame(fig, "z500", step, init_date, valid_date)


def frame_llj(step, data, climo, cfg, soi, init_date, valid_date):
    lat, lon  = _latlon(data, cfg)
    u850, lat_i, lon_i = box(data["u850"], lat, lon, cfg)
    v850 = box(data["v850"], lat, lon, cfg)[0]
    wspd = np.sqrt(u850**2 + v850**2)

    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "850 hPa Wind Speed  [m/s]  —  Monsoon LLJ")
    levs = np.arange(0, 24, 2)
    cf   = ax.contourf(lon_i, lat_i, wspd, levels=levs,
                       cmap=LLJ_CMAP, transform=PROJ, extend="max", zorder=1)
    try:
        ax.streamplot(lon_i, lat_i, u850, v850,
                      transform=PROJ, color="#ffffffaa",
                      linewidth=0.65, density=1.5, arrowsize=0.9, zorder=5)
    except Exception:
        skip = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u850[::skip,::skip], v850[::skip,::skip],
                  transform=PROJ, scale=180, width=0.003,
                  color="white", alpha=0.8, zorder=5)
    colorbar(fig, cf, ax, "m/s", ticks=levs)

    ax = axes[1]
    base_map(ax, cfg, soi, "850 hPa Wind Speed Anomaly  [m/s]")
    if climo and step in climo and "u850" in climo[step]:
        uc   = box(climo[step]["u850"], lat, lon, cfg)[0]
        vc   = box(climo[step]["v850"], lat, lon, cfg)[0]
        anom = wspd - np.sqrt(uc**2 + vc**2)
        lim  = max(2, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        skip = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  (u850-uc)[::skip,::skip], (v850-vc)[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="#ffffffbb", zorder=5, headwidth=4, headlength=4)
        colorbar(fig, cf2, ax, "m/s")

    return render_frame(fig, "llj", step, init_date, valid_date)


def frame_olr(step, data, olr_day1, cfg, soi, init_date, valid_date):
    lat, lon  = _latlon(data, cfg)
    ttr, lat_i, lon_i = box(data["ttr"], lat, lon, cfg)
    olr = -ttr   # W/m², positive = outgoing energy

    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "OLR  [W/m²]  —  Convection Proxy")
    levs = np.arange(100, 330, 20)
    cf   = ax.contourf(lon_i, lat_i, olr, levels=levs,
                       cmap=OLR_CMAP, transform=PROJ, extend="both", zorder=1)
    colorbar(fig, cf, ax, "W/m²", ticks=levs[::2])

    ax = axes[1]
    base_map(ax, cfg, soi, "OLR Anomaly vs Day-1  [W/m²]")
    if olr_day1 is not None:
        anom = olr - (-olr_day1)
        lim  = max(20, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap="PuOr_r", transform=PROJ, extend="both", zorder=1)
        colorbar(fig, cf2, ax, "W/m²")

    return render_frame(fig, "olr", step, init_date, valid_date)


def frame_moisture(step, data, climo, cfg, soi, init_date, valid_date):
    lat, lon  = _latlon(data, cfg)
    tcwv, lat_i, lon_i = box(data["tcwv"], lat, lon, cfg)

    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "Total Column Water Vapour  [kg/m²]")
    levs = np.arange(10, 78, 4)
    cf   = ax.contourf(lon_i, lat_i, tcwv, levels=levs,
                       cmap=TCWV_CMAP, transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, tcwv, levels=[40, 50, 60],
                      colors="white", linewidths=0.4, alpha=0.5, transform=PROJ)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white")
    colorbar(fig, cf, ax, "kg/m²", ticks=levs[::2])

    ax = axes[1]
    base_map(ax, cfg, soi, "TCWV Anomaly vs Climatology  [kg/m²]")
    if climo and step in climo and "tcwv" in climo[step]:
        tcwv_c = box(climo[step]["tcwv"], lat, lon, cfg)[0]
        anom   = tcwv - tcwv_c
        lim    = max(3, float(np.nanpercentile(np.abs(anom), 97)))
        cf2    = ax.contourf(lon_i, lat_i, anom,
                             levels=np.linspace(-lim, lim, 21),
                             cmap=PRECIP_ANOM_CMAP, transform=PROJ,
                             extend="both", zorder=1)
        colorbar(fig, cf2, ax, "kg/m²")

    return render_frame(fig, "moisture", step, init_date, valid_date)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _fig2():
    return plt.subplots(1, 2, figsize=(15, 7.5),
                        subplot_kw=dict(projection=PROJ),
                        facecolor="#0a0a0a")


def _latlon(data, cfg):
    """Extract lat/lon arrays from the first available channel in data."""
    # data values are plain numpy arrays — lat/lon stored externally; use cfg to find
    # They are actually just arrays — we stored them alongside in load_step return
    # This helper is called with the raw arrays from load_step
    raise RuntimeError("Use _ll() with explicit lat/lon from load_step")


def _ll_box(arr, lat, lon, cfg):
    return box(arr, lat, lon, cfg)


# ── MODE REGISTRY ─────────────────────────────────────────────────────────────
MODES = {
    "tp":       dict(channels=["tp"],             label="Precipitation"),
    "t2m":      dict(channels=["t2m"],            label="Temperature"),
    "z500":     dict(channels=["z500","u850","v850"], label="Z500+Wind"),
    "llj":      dict(channels=["u850","v850"],    label="LLJ/850hPa"),
    "olr":      dict(channels=["ttr"],            label="OLR/Convection"),
    "moisture": dict(channels=["tcwv"],           label="Moisture/TCWV"),
}


def make_gif(mode, cfg, climo, soi, init_date, olr_day1=None):
    label  = MODES[mode]["label"]
    chans  = MODES[mode]["channels"]
    fps    = cfg["fps"]
    nsteps = cfg["total_steps"]

    print(f"\n{'='*55}")
    print(f"  {label}  ({nsteps} frames @ {fps} fps)")
    print(f"{'='*55}")

    frames = []

    for step in range(1, nsteps + 1):
        data, lat, lon = load_step(cfg["raw_dir"], cfg["date"],
                                   step, chans, cfg["members"])
        if not data:
            print(f"  SKIP step {step} — no data")
            continue

        valid = init_date + datetime.timedelta(days=step)

        # Dispatch to the right frame builder
        # All frame functions now accept (step, data, climo_or_ref, cfg, soi, init, valid, lat, lon)
        if mode == "tp":
            img = _frame_tp(step, data, lat, lon, climo, cfg, soi, init_date, valid)
        elif mode == "t2m":
            img = _frame_t2m(step, data, lat, lon, climo, cfg, soi, init_date, valid)
        elif mode == "z500":
            img = _frame_z500(step, data, lat, lon, climo, cfg, soi, init_date, valid)
        elif mode == "llj":
            img = _frame_llj(step, data, lat, lon, climo, cfg, soi, init_date, valid)
        elif mode == "olr":
            img = _frame_olr(step, data, lat, lon, olr_day1, cfg, soi, init_date, valid)
        elif mode == "moisture":
            img = _frame_moisture(step, data, lat, lon, climo, cfg, soi, init_date, valid)

        frames.append(img)
        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")

    if not frames:
        print("  No frames generated — check data path")
        return

    out = Path(cfg["out_dir"]) / f"{mode}_animation.gif"
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {out}  ({out.stat().st_size/1024**2:.1f} MB)")


# ── INTERNAL FRAME FUNCTIONS (lat/lon explicit) ───────────────────────────────

def _frame_tp(step, data, lat, lon, climo, cfg, soi, init_date, valid_date):
    tp, lat_i, lon_i = box(data["tp"], lat, lon, cfg)
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "Precipitation  [mm/day]")
    levs = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    cf   = ax.contourf(lon_i, lat_i, tp, levels=levs,
                       cmap=TP_CMAP, transform=PROJ, extend="max", zorder=1)
    colorbar(fig, cf, ax, "mm / day", ticks=levs)

    ax = axes[1]
    base_map(ax, cfg, soi, "Precipitation Anomaly  [mm/day]")
    if climo and step in climo and "tp" in climo[step]:
        tp_c = box(climo[step]["tp"], lat, lon, cfg)[0]
        anom = tp - tp_c
        lim  = max(2.0, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap=PRECIP_ANOM_CMAP, transform=PROJ,
                           extend="both", zorder=1)
        colorbar(fig, cf2, ax, "mm/day")

    return render_frame(fig, "tp", step, init_date, valid_date)


def _frame_t2m(step, data, lat, lon, climo, cfg, soi, init_date, valid_date):
    t2m_K, lat_i, lon_i = box(data["t2m"], lat, lon, cfg)
    t2m_C = t2m_K - 273.15
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "2m Temperature  [°C]")
    levs = np.arange(-4, 46, 2)
    cf   = ax.contourf(lon_i, lat_i, t2m_C, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, t2m_C, levels=levs[::4],
                      colors="white", linewidths=0.3, alpha=0.35, transform=PROJ)
    ax.clabel(cs, fmt="%d°", fontsize=6, inline=True, colors="white")
    colorbar(fig, cf, ax, "°C")

    ax = axes[1]
    base_map(ax, cfg, soi, "T2m Anomaly vs Climatology  [°C]")
    if climo and step in climo and "t2m" in climo[step]:
        t2m_c = box(climo[step]["t2m"], lat, lon, cfg)[0] - 273.15
        anom  = t2m_C - t2m_c
        lim   = max(1.5, float(np.nanpercentile(np.abs(anom), 97)))
        cf2   = ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        colorbar(fig, cf2, ax, "°C")

    return render_frame(fig, "t2m", step, init_date, valid_date)


def _frame_z500(step, data, lat, lon, climo, cfg, soi, init_date, valid_date):
    z_gpm, lat_i, lon_i = box(data["z500"] / 9.80665, lat, lon, cfg)
    u850 = box(data["u850"], lat, lon, cfg)[0]
    v850 = box(data["v850"], lat, lon, cfg)[0]
    skip = 2
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "Z500  [gpm]  +  850 hPa Wind  [m/s]")
    levs = np.arange(5720, 5940, 10)
    cf   = ax.contourf(lon_i, lat_i, z_gpm, levels=levs,
                       cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, z_gpm, levels=levs[::2],
                      colors="white", linewidths=0.3, alpha=0.4, transform=PROJ)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white", inline=True)
    ax.quiver(lon_i[::skip], lat_i[::skip],
              u850[::skip,::skip], v850[::skip,::skip],
              transform=PROJ, scale=200, width=0.003,
              color="#ffffffcc", zorder=6, headwidth=4, headlength=4)
    colorbar(fig, cf, ax, "gpm")

    ax = axes[1]
    base_map(ax, cfg, soi, "Z500 Anomaly  [gpm]  +  Wind Anomaly  [m/s]")
    if climo and step in climo and "z500" in climo[step]:
        zc    = box(climo[step]["z500"], lat, lon, cfg)[0]
        uc    = box(climo[step]["u850"], lat, lon, cfg)[0]
        vc    = box(climo[step]["v850"], lat, lon, cfg)[0]
        zanom = z_gpm - zc
        lim   = max(10, float(np.nanpercentile(np.abs(zanom), 95)))
        cf2   = ax.contourf(lon_i, lat_i, zanom,
                            levels=np.linspace(-lim, lim, 21),
                            cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  (u850-uc)[::skip,::skip], (v850-vc)[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="#ffffffbb", zorder=6, headwidth=4, headlength=4)
        colorbar(fig, cf2, ax, "gpm")

    return render_frame(fig, "z500", step, init_date, valid_date)


def _frame_llj(step, data, lat, lon, climo, cfg, soi, init_date, valid_date):
    u850, lat_i, lon_i = box(data["u850"], lat, lon, cfg)
    v850 = box(data["v850"], lat, lon, cfg)[0]
    wspd = np.sqrt(u850**2 + v850**2)
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "850 hPa Wind Speed  [m/s]  —  Monsoon LLJ")
    levs = np.arange(0, 24, 2)
    cf   = ax.contourf(lon_i, lat_i, wspd, levels=levs,
                       cmap=LLJ_CMAP, transform=PROJ, extend="max", zorder=1)
    try:
        ax.streamplot(lon_i, lat_i, u850, v850,
                      transform=PROJ, color="#ffffffaa",
                      linewidth=0.65, density=1.5, arrowsize=0.9, zorder=5)
    except Exception:
        skip = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u850[::skip,::skip], v850[::skip,::skip],
                  transform=PROJ, scale=180, width=0.003, color="white",
                  alpha=0.8, zorder=5)
    colorbar(fig, cf, ax, "m/s", ticks=levs)

    ax = axes[1]
    base_map(ax, cfg, soi, "850 hPa Wind Speed Anomaly  [m/s]")
    if climo and step in climo and "u850" in climo[step]:
        uc   = box(climo[step]["u850"], lat, lon, cfg)[0]
        vc   = box(climo[step]["v850"], lat, lon, cfg)[0]
        anom = wspd - np.sqrt(uc**2 + vc**2)
        lim  = max(2, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap=DIV_CMAP, transform=PROJ, extend="both", zorder=1)
        skip = 2
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  (u850-uc)[::skip,::skip], (v850-vc)[::skip,::skip],
                  transform=PROJ, scale=80, width=0.003,
                  color="#ffffffbb", zorder=5, headwidth=4, headlength=4)
        colorbar(fig, cf2, ax, "m/s")

    return render_frame(fig, "llj", step, init_date, valid_date)


def _frame_olr(step, data, lat, lon, olr_day1, cfg, soi, init_date, valid_date):
    ttr, lat_i, lon_i = box(data["ttr"], lat, lon, cfg)
    olr = -ttr
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "OLR  [W/m²]  —  Convection Proxy")
    levs = np.arange(100, 330, 20)
    cf   = ax.contourf(lon_i, lat_i, olr, levels=levs,
                       cmap=OLR_CMAP, transform=PROJ, extend="both", zorder=1)
    colorbar(fig, cf, ax, "W/m²", ticks=levs[::2])

    ax = axes[1]
    base_map(ax, cfg, soi, "OLR Anomaly vs Day-1  [W/m²]")
    if olr_day1 is not None:
        anom = olr - (-olr_day1)
        lim  = max(20, float(np.nanpercentile(np.abs(anom), 97)))
        cf2  = ax.contourf(lon_i, lat_i, anom,
                           levels=np.linspace(-lim, lim, 21),
                           cmap="PuOr_r", transform=PROJ, extend="both", zorder=1)
        colorbar(fig, cf2, ax, "W/m²")

    return render_frame(fig, "olr", step, init_date, valid_date)


def _frame_moisture(step, data, lat, lon, climo, cfg, soi, init_date, valid_date):
    tcwv, lat_i, lon_i = box(data["tcwv"], lat, lon, cfg)
    fig, axes = _fig2()

    ax = axes[0]
    base_map(ax, cfg, soi, "Total Column Water Vapour  [kg/m²]")
    levs = np.arange(10, 78, 4)
    cf   = ax.contourf(lon_i, lat_i, tcwv, levels=levs,
                       cmap=TCWV_CMAP, transform=PROJ, extend="both", zorder=1)
    cs   = ax.contour(lon_i, lat_i, tcwv, levels=[40, 50, 60],
                      colors="white", linewidths=0.4, alpha=0.5, transform=PROJ)
    ax.clabel(cs, fmt="%d", fontsize=6, colors="white")
    colorbar(fig, cf, ax, "kg/m²", ticks=levs[::2])

    ax = axes[1]
    base_map(ax, cfg, soi, "TCWV Anomaly vs Climatology  [kg/m²]")
    if climo and step in climo and "tcwv" in climo[step]:
        tcwv_c = box(climo[step]["tcwv"], lat, lon, cfg)[0]
        anom   = tcwv - tcwv_c
        lim    = max(3, float(np.nanpercentile(np.abs(anom), 97)))
        cf2    = ax.contourf(lon_i, lat_i, anom,
                             levels=np.linspace(-lim, lim, 21),
                             cmap=PRECIP_ANOM_CMAP, transform=PROJ,
                             extend="both", zorder=1)
        colorbar(fig, cf2, ax, "kg/m²")

    return render_frame(fig, "moisture", step, init_date, valid_date)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Make FuXi-S2S forecast animation GIFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date",       default=CONFIG["date"],
                   help="Init date YYYYMMDD")
    p.add_argument("--raw_dir",    default=CONFIG["raw_dir"],
                   help="Parent dir containing <date>/member/ subdirs")
    p.add_argument("--out_dir",    default=CONFIG["out_dir"],
                   help="Output directory for GIF files")
    p.add_argument("--soi_shp",    default=CONFIG["soi_shp"],
                   help="Path to Survey of India STATE_BOUNDARY.shp")
    p.add_argument("--mode",       default=None,
                   choices=list(MODES.keys()),
                   help="Single mode; omit to run all")
    p.add_argument("--members",    type=int, nargs="+",
                   default=CONFIG["members"],
                   help="Ensemble member indices to average")
    p.add_argument("--fps",        type=int, default=CONFIG["fps"])
    p.add_argument("--steps",      type=int, default=CONFIG["total_steps"],
                   help="Number of lead days")
    p.add_argument("--lat_min",    type=float, default=CONFIG["lat_min"])
    p.add_argument("--lat_max",    type=float, default=CONFIG["lat_max"])
    p.add_argument("--lon_min",    type=float, default=CONFIG["lon_min"])
    p.add_argument("--lon_max",    type=float, default=CONFIG["lon_max"])
    return p.parse_args()


def main():
    args = parse_args()

    cfg = dict(
        date        = args.date,
        raw_dir     = args.raw_dir,
        out_dir     = args.out_dir,
        soi_shp     = args.soi_shp,
        members     = args.members,
        fps         = args.fps,
        total_steps = args.steps,
        lat_min     = args.lat_min,
        lat_max     = args.lat_max,
        lon_min     = args.lon_min,
        lon_max     = args.lon_max,
    )

    init_date = datetime.date(int(args.date[:4]),
                              int(args.date[4:6]),
                              int(args.date[6:8]))

    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"\nFuXi-S2S Animation Generator")
    print(f"  Init date : {init_date.strftime('%d %b %Y')}  ({args.date})")
    print(f"  Raw dir   : {cfg['raw_dir']}/{cfg['date']}/member/")
    print(f"  Output    : {cfg['out_dir']}")
    print(f"  Domain    : lat {cfg['lat_min']}–{cfg['lat_max']}  "
          f"lon {cfg['lon_min']}–{cfg['lon_max']}")
    print(f"  Members   : {cfg['members']}   FPS: {cfg['fps']}   "
          f"Steps: {cfg['total_steps']}\n")

    soi = load_soi(cfg["soi_shp"])

    # Get lat/lon reference grid from step 1
    ref_data, lat, lon = load_step(cfg["raw_dir"], cfg["date"],
                                   1, ["t2m"], cfg["members"])
    if lat is None:
        print(f"ERROR: no data at {cfg['raw_dir']}/{cfg['date']}/member/")
        sys.exit(1)

    climo = build_climo_cache(init_date, cfg["total_steps"], lat, lon)

    # OLR day-1 reference
    d1_data, d1_lat, d1_lon = load_step(cfg["raw_dir"], cfg["date"],
                                         1, ["ttr"], cfg["members"])
    olr_day1 = box(d1_data["ttr"], d1_lat, d1_lon, cfg)[0] \
               if "ttr" in d1_data else None

    modes = [args.mode] if args.mode else list(MODES.keys())
    for m in modes:
        make_gif(m, cfg, climo, soi, init_date,
                 olr_day1=olr_day1 if m == "olr" else None)

    print(f"\nAll done!  GIFs → {cfg['out_dir']}")


if __name__ == "__main__":
    main()
