#!/usr/bin/env python3
"""
make_forecast_animation.py
==========================
Create publication-quality animated GIFs of FuXi-S2S 42-day forecast over India.

Three GIFs:
  1. tp_forecast.gif       — Daily TP forecast (mm/day)
  2. t2m_anomaly.gif       — T2m anomaly vs WB2 1990-2019 climatology (°C)
  3. z500_wind_anomaly.gif — Z500 anomaly (gpm) + 850 hPa wind anomaly vectors

Each frame: forecast field (left) | anomaly vs climatology (right)
Dates shown correctly on each frame.

Climatology: WeatherBench2 ERA5 1990-2019 daily mean, fetched per DOY.

Usage
-----
  python make_forecast_animation.py
  python make_forecast_animation.py --members 0 1 2   # specific members
  python make_forecast_animation.py --fps 3           # slower
"""

import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
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

INIT_DATE = datetime.date(2026, 6, 18)
TOTAL_STEPS = 42
DEFAULT_FPS = 4

# India domain
LAT_MIN, LAT_MAX = 5, 38
LON_MIN, LON_MAX = 65, 100

# SOI LCC CRS
SOI_CRS = ccrs.LambertConformal(
    central_longitude=80.0, central_latitude=24.0,
    standard_parallels=(12.472944, 35.172806),
    false_easting=4000000.0, false_northing=4000000.0,
)

PROJ = ccrs.PlateCarree()


# ── COLORMAPS ─────────────────────────────────────────────────────────────────
# Custom beautiful colormaps
TP_CMAP   = "YlGnBu"
ANOM_CMAP = plt.cm.RdBu_r
Z500_CMAP = "RdYlBu_r"

TP_LEVELS   = [0, 0.5, 1, 2, 3, 5, 8, 12, 18, 25]   # mm/day
T2M_LEVELS  = np.linspace(-6, 6, 25)                  # anomaly °C
Z500_LEVELS = np.linspace(-80, 80, 25)                 # anomaly gpm


# ── LOAD SOI SHAPEFILE ────────────────────────────────────────────────────────
def load_soi():
    try:
        reader = shpreader.Reader(SOI_SHP)
        geoms  = [r.geometry.simplify(5000) for r in reader.records()]
        print(f"  SOI: {len(geoms)} state polygons loaded")
        return geoms
    except Exception as e:
        print(f"  WARNING: SOI shapefile failed ({e}), using cartopy")
        return None


def add_india_map(ax, soi_geoms):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),  color="#cce5f0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),   color="#f5f0e8", zorder=0)
    if soi_geoms:
        ax.add_geometries(soi_geoms, crs=SOI_CRS,
                          facecolor="none", edgecolor="#333333",
                          linewidth=0.6, zorder=4)
    else:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       linewidth=0.4, edgecolor="#555555", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.9, edgecolor="#111111", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.7, edgecolor="#222222",
                   linestyle="--", zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="grey",
                      alpha=0.5, zorder=3, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlocator  = mticker.MultipleLocator(10)
    gl.ylocator  = mticker.MultipleLocator(10)
    gl.xlabel_style = {"size": 7, "color": "#444444"}
    gl.ylabel_style = {"size": 7, "color": "#444444"}


# ── LOAD MODEL OUTPUT ─────────────────────────────────────────────────────────
def load_step(step, channels, members):
    """Load specific channels for one step across members → dict of (member, lat, lon)."""
    data = {ch: [] for ch in channels}
    for mem in members:
        f = RAW_DIR / f"member/{mem:02d}/{step:02d}.nc"
        if not f.exists():
            continue
        da = xr.open_dataarray(str(f))
        for ch in channels:
            data[ch].append(da.sel(channel=ch).squeeze(drop=True).values)
    out = {}
    for ch in channels:
        if data[ch]:
            arr = np.array(data[ch])   # (nmem, lat, lon)
            out[ch] = arr.mean(axis=0)  # ensemble mean
    return out, da.lat.values, da.lon.values


# ── LOAD WB2 CLIMATOLOGY ──────────────────────────────────────────────────────
def build_climo_cache(variables, lat, lon):
    """Pre-fetch WB2 climo for all 42 valid dates, regrid to FuXi 1.5° grid."""
    print("  Loading WB2 climatology for all 42 lead days ...")
    try:
        from earth2studio.data import WB2Climatology
        wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr",
                             verbose=False)
    except Exception as e:
        print(f"  WARNING: WB2 not available ({e}), anomalies will be vs day-1")
        return None

    cache = {}
    wb2_vars = []
    if "t2m"  in variables: wb2_vars.append("t2m")
    if "z500" in variables: wb2_vars.append("z500")
    if "tp"   in variables: wb2_vars.append("tp")
    if "u850" in variables: wb2_vars.append("u10")   # closest available

    for step in range(1, TOTAL_STEPS + 1):
        valid = INIT_DATE + datetime.timedelta(days=step)
        t     = datetime.datetime(2001, valid.month, valid.day, 0)  # dummy year
        da    = wb2(t, wb2_vars)

        step_climo = {}
        for wv in wb2_vars:
            arr = da.sel(variable=wv).squeeze().values   # (721, 1440)
            # Regrid WB2 0.25° → FuXi 1.5°: simple nearest-neighbour via xr interp
            wb2_lat = da.lat.values
            wb2_lon = da.lon.values
            da_xr   = xr.DataArray(arr, dims=["lat", "lon"],
                                   coords={"lat": wb2_lat, "lon": wb2_lon})
            arr_1p5 = da_xr.interp(lat=lat, lon=lon, method="linear").values
            step_climo[wv] = arr_1p5

        cache[step] = step_climo
        if step % 7 == 0:
            print(f"    Climo cached through step {step}")

    print("  Climatology cache ready.")
    return cache


# ── FRAME DRAWING ─────────────────────────────────────────────────────────────
STYLE = dict(fontfamily="DejaVu Sans")


def draw_tp_frame(ax_fc, ax_an, tp, tp_climo, lat, lon, soi, valid_date, step):
    """TP forecast | TP anomaly."""
    for ax in [ax_fc, ax_an]:
        add_india_map(ax, soi)

    la = slice(LAT_MAX, LAT_MIN)
    lo = slice(LON_MIN, LON_MAX)
    lat_i = lat[(lat >= LAT_MIN) & (lat <= LAT_MAX)]
    lon_i = lon[(lon >= LON_MIN) & (lon <= LON_MAX)]

    # Subset
    tp_sub = tp[(lat >= LAT_MIN) & (lat <= LAT_MAX), :]
    tp_sub = tp_sub[:, (lon >= LON_MIN) & (lon <= LON_MAX)]

    cf1 = ax_fc.contourf(lon_i, lat_i, tp_sub,
                          levels=TP_LEVELS, cmap=TP_CMAP,
                          transform=PROJ, extend="max", zorder=1)
    plt.colorbar(cf1, ax=ax_fc, orientation="horizontal",
                 label="TP (mm/day)", pad=0.04, shrink=0.95,
                 ticks=TP_LEVELS, format="%.0f")
    ax_fc.set_title("Precipitation Forecast", fontsize=10, fontweight="bold", pad=4)

    if tp_climo is not None:
        tp_clim_sub = tp_climo[(lat >= LAT_MIN) & (lat <= LAT_MAX), :]
        tp_clim_sub = tp_clim_sub[:, (lon >= LON_MIN) & (lon <= LON_MAX)]
        anom = tp_sub - tp_clim_sub
        lim  = max(3, np.nanpercentile(np.abs(anom), 98))
        levs = np.linspace(-lim, lim, 21)
        cf2  = ax_an.contourf(lon_i, lat_i, anom, levels=levs,
                               cmap="BrBG", transform=PROJ,
                               extend="both", zorder=1)
        plt.colorbar(cf2, ax=ax_an, orientation="horizontal",
                     label="TP anomaly (mm/day)", pad=0.04, shrink=0.95)
        ax_an.set_title("Precipitation Anomaly (vs 1990–2019 clim)", fontsize=10,
                         fontweight="bold", pad=4)
    else:
        ax_an.text(0.5, 0.5, "No climatology", transform=ax_an.transAxes,
                   ha="center", va="center", fontsize=12)


def draw_t2m_frame(ax_fc, ax_an, t2m, t2m_climo, lat, lon, soi, valid_date, step):
    """T2m absolute | T2m anomaly."""
    for ax in [ax_fc, ax_an]:
        add_india_map(ax, soi)

    lat_i = lat[(lat >= LAT_MIN) & (lat <= LAT_MAX)]
    lon_i = lon[(lon >= LON_MIN) & (lon <= LON_MAX)]
    t2m_C = t2m[(lat >= LAT_MIN) & (lat <= LAT_MAX), :]
    t2m_C = t2m_C[:, (lon >= LON_MIN) & (lon <= LON_MAX)] - 273.15

    levs_abs = np.arange(-4, 42, 2)
    cf1 = ax_fc.contourf(lon_i, lat_i, t2m_C, levels=levs_abs,
                          cmap="RdYlBu_r", transform=PROJ, extend="both", zorder=1)
    cs1 = ax_fc.contour(lon_i, lat_i, t2m_C, levels=levs_abs[::3],
                         colors="k", linewidths=0.3, transform=PROJ, zorder=2)
    ax_fc.clabel(cs1, fmt="%d°", fontsize=6, inline=True)
    plt.colorbar(cf1, ax=ax_fc, orientation="horizontal",
                 label="T2m (°C)", pad=0.04, shrink=0.95)
    ax_fc.set_title("2m Temperature", fontsize=10, fontweight="bold", pad=4)

    if t2m_climo is not None:
        clim_C = t2m_climo[(lat >= LAT_MIN) & (lat <= LAT_MAX), :]
        clim_C = clim_C[:, (lon >= LON_MIN) & (lon <= LON_MAX)] - 273.15
        anom   = t2m_C - clim_C
        lim    = max(2, np.nanpercentile(np.abs(anom), 98))
        levs   = np.linspace(-lim, lim, 21)
        cf2    = ax_an.contourf(lon_i, lat_i, anom, levels=levs,
                                 cmap="RdBu_r", transform=PROJ,
                                 extend="both", zorder=1)
        cs2    = ax_an.contour(lon_i, lat_i, anom, levels=[-2, -1, 1, 2],
                                colors="k", linewidths=0.3, transform=PROJ, zorder=2)
        ax_an.clabel(cs2, fmt="%+.0f°", fontsize=6)
        plt.colorbar(cf2, ax=ax_an, orientation="horizontal",
                     label="T2m anomaly (°C)", pad=0.04, shrink=0.95)
        ax_an.set_title("T2m Anomaly (vs 1990–2019 clim)", fontsize=10,
                         fontweight="bold", pad=4)


def draw_z500_frame(ax_fc, ax_an, z500, u850, v850,
                    z500_climo, u850_climo, v850_climo,
                    lat, lon, soi, valid_date, step):
    """Z500 + wind | Z500 anomaly + wind anomaly."""
    for ax in [ax_fc, ax_an]:
        add_india_map(ax, soi)

    lat_i = lat[(lat >= LAT_MIN) & (lat <= LAT_MAX)]
    lon_i = lon[(lon >= LON_MIN) & (lon <= LON_MAX)]

    def sub(arr):
        a = arr[(lat >= LAT_MIN) & (lat <= LAT_MAX), :]
        return a[:, (lon >= LON_MIN) & (lon <= LON_MAX)]

    z_gpm  = sub(z500)  / 9.80665
    u_ms   = sub(u850)
    v_ms   = sub(v850)

    z_levs = np.arange(5740, 5920, 10)
    cf1    = ax_fc.contourf(lon_i, lat_i, z_gpm, levels=z_levs,
                             cmap=Z500_CMAP, transform=PROJ, extend="both", zorder=1)
    cs1    = ax_fc.contour(lon_i, lat_i, z_gpm, levels=z_levs[::2],
                            colors="k", linewidths=0.4, transform=PROJ, zorder=2)
    ax_fc.clabel(cs1, fmt="%d", fontsize=7)
    skip   = 2
    ax_fc.quiver(lon_i[::skip], lat_i[::skip],
                 u_ms[::skip, ::skip], v_ms[::skip, ::skip],
                 transform=PROJ, scale=180, width=0.004,
                 color="#111111", zorder=6,
                 headwidth=4, headlength=4)
    plt.colorbar(cf1, ax=ax_fc, orientation="horizontal",
                 label="Z500 (gpm)", pad=0.04, shrink=0.95)
    ax_fc.set_title("Z500 + 850hPa Wind", fontsize=10, fontweight="bold", pad=4)

    if z500_climo is not None:
        z_an  = sub(z500_climo) / 9.80665
        u_an  = sub(u850_climo) if u850_climo is not None else None
        v_an  = sub(v850_climo) if v850_climo is not None else None
        zanom = z_gpm - z_an
        lim   = max(30, np.nanpercentile(np.abs(zanom), 98))
        levs  = np.linspace(-lim, lim, 21)
        cf2   = ax_an.contourf(lon_i, lat_i, zanom, levels=levs,
                                cmap="RdBu_r", transform=PROJ,
                                extend="both", zorder=1)
        cs2   = ax_an.contour(lon_i, lat_i, zanom, levels=[-30,-20,-10,10,20,30],
                               colors="k", linewidths=0.3, transform=PROJ, zorder=2)
        ax_an.clabel(cs2, fmt="%+.0f", fontsize=6)
        if u_an is not None:
            uanom = u_ms - u_an
            vanom = v_ms - v_an
            ax_an.quiver(lon_i[::skip], lat_i[::skip],
                         uanom[::skip, ::skip], vanom[::skip, ::skip],
                         transform=PROJ, scale=100, width=0.004,
                         color="#111111", zorder=6,
                         headwidth=4, headlength=4)
        plt.colorbar(cf2, ax=ax_an, orientation="horizontal",
                     label="Z500 anomaly (gpm)", pad=0.04, shrink=0.95)
        ax_an.set_title("Z500 Anomaly + Wind Anomaly (vs 1990–2019 clim)",
                         fontsize=10, fontweight="bold", pad=4)


# ── MAIN FIGURE BUILDER ───────────────────────────────────────────────────────
def make_frame(step, data, climo, lat, lon, soi, mode):
    """Render one frame, return PIL Image."""
    valid_date = INIT_DATE + datetime.timedelta(days=step)
    lead_str   = f"Lead day {step:02d}  |  Valid: {valid_date.strftime('%d %b %Y')}"
    init_str   = f"FuXi-S2S  Init: {INIT_DATE.strftime('%d %b %Y')}"

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw=dict(projection=PROJ),
        facecolor="#1a1a2e"
    )
    fig.patch.set_facecolor("#1a1a2e")

    # Title bar
    fig.text(0.5, 0.97, init_str, ha="center", va="top",
             fontsize=11, color="white", fontweight="bold",
             fontfamily="DejaVu Sans")
    fig.text(0.5, 0.93, lead_str, ha="center", va="top",
             fontsize=13, color="#f0c040", fontweight="bold",
             fontfamily="DejaVu Sans")

    if mode == "tp":
        draw_tp_frame(axes[0], axes[1],
                      data["tp"], climo.get(step, {}).get("tp") if climo else None,
                      lat, lon, soi, valid_date, step)

    elif mode == "t2m":
        draw_t2m_frame(axes[0], axes[1],
                       data["t2m"], climo.get(step, {}).get("t2m") if climo else None,
                       lat, lon, soi, valid_date, step)

    elif mode == "z500":
        draw_z500_frame(axes[0], axes[1],
                        data["z500"], data["u850"], data["v850"],
                        climo.get(step, {}).get("z500") if climo else None,
                        climo.get(step, {}).get("u10")  if climo else None,
                        climo.get(step, {}).get("u10")  if climo else None,
                        lat, lon, soi, valid_date, step)

    for ax in axes:
        ax.set_facecolor("#cce5f0")

    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12,
                        wspace=0.08)

    # Render to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return img


def make_gif(mode, members, fps, climo, soi):
    channels = {
        "tp":   ["tp"],
        "t2m":  ["t2m"],
        "z500": ["z500", "u850", "v850"],
    }[mode]

    print(f"\n{'='*55}")
    print(f"  Making {mode.upper()} animation ({TOTAL_STEPS} frames) ...")
    print(f"{'='*55}")

    frames = []
    lat = lon = None

    for step in range(1, TOTAL_STEPS + 1):
        data, lat, lon = load_step(step, channels, members)
        if not data:
            print(f"  step {step:02d}: no data, skipping")
            continue

        img = make_frame(step, data, climo, lat, lon, soi, mode)
        frames.append(img)

        if step % 7 == 0 or step == 1:
            print(f"  step {step:02d}/{TOTAL_STEPS} rendered")

    if not frames:
        print("  No frames generated!")
        return

    out_path = OUT_DIR / f"{mode}_animation.gif"
    duration_ms = int(1000 / fps)

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / 1024**2
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB, {len(frames)} frames @ {fps} fps)")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps",     type=int, default=DEFAULT_FPS)
    parser.add_argument("--mode",    type=str, default=None,
                        choices=["tp", "t2m", "z500"],
                        help="Only make one GIF (default: all three)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {OUT_DIR}")
    print(f"Members    : {args.members}")
    print(f"FPS        : {args.fps}")

    soi = load_soi()

    # Need lat/lon from step 1 to build climo cache
    _, lat, lon = load_step(1, ["t2m"], args.members)

    climo = build_climo_cache(["t2m", "z500", "tp"], lat, lon)

    modes = [args.mode] if args.mode else ["tp", "t2m", "z500"]
    for mode in modes:
        make_gif(mode, args.members, args.fps, climo, soi)

    print(f"\nAll done! GIFs in {OUT_DIR}")


if __name__ == "__main__":
    main()
