#!/usr/bin/env python3
"""
plot_erpas_style.py
===================
Reproduce ERPAS-style forecast plots from FuXi-S2S output for visual comparison.

Produces (matching ERPAS layout exactly):
  prec_wind850_<DATE>.gif   — Rainfall (mm/day) shaded + 850hPa winds, animated daily
  vort850_mslp_<DATE>.gif   — 850hPa vorticity shaded + MSLP contours, animated daily
  divg200_wind_z500_<DATE>.gif — 200hPa divergence + winds + 500hPa Z contours, animated
  rf_weekly_<DATE>.png      — 2×3: actual rainfall weeks 1-3 | anomaly weeks 1-3
  tmax_anom_weekly_<DATE>.png — 2×2: Tmax anomaly weeks 1-4 (India)

Uses ensemble mean of all available members.
Anomalies vs WeatherBench2 ERA5 1990-2019 climatology.

Usage
-----
  python plot_erpas_style.py --date 20260617
  python plot_erpas_style.py --date 20260617 --mode prec   # single plot
  python plot_erpas_style.py --date 20260617 --steps 28    # 4 weeks only
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
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from PIL import Image

warnings.filterwarnings("ignore")

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw")
OUT_DIR = Path("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/"
               "fuxi_s2s/erpas_style")

PROJ = ccrs.PlateCarree()
FPS  = 4   # ERPAS animates daily, so fast playback is fine

# ── ERPAS COLORMAPS ───────────────────────────────────────────────────────────
# Precipitation: exactly ERPAS green scheme
PREC_BOUNDS = [2, 4, 8, 12, 18, 25, 35]
PREC_COLORS = ["#d9f0d3","#a8ddb5","#4eb3d3","#2b8cbe","#08589e",
               "#084081","#02205e"]
PREC_CMAP   = mcolors.ListedColormap(PREC_COLORS)
PREC_NORM   = BoundaryNorm(PREC_BOUNDS, PREC_CMAP.N)

# Vorticity: black→blue→white→orange→red (ERPAS style)
VORT_BOUNDS = [-12,-9,-6,-3,0,3,6,9,12]
VORT_CMAP   = LinearSegmentedColormap.from_list("vort", [
    "#1a0030","#4a0080","#0000cc","#6699ff","#ccddff",
    "#ffffff",
    "#ffddaa","#ff8800","#cc2200","#660000",
], N=256)

# Divergence: same diverging scheme
DIV_CMAP = LinearSegmentedColormap.from_list("div", [
    "#1a0030","#4a0080","#0000cc","#6699ff","#ccddff",
    "#ffffff",
    "#ffddaa","#ff8800","#cc2200","#660000",
], N=256)

# Tmax anomaly: blue→white→red, ±10°C
TANOM_BOUNDS = [-10,-9,-7,-5,-3,-1,0,1,3,5,7,9,10]
TANOM_CMAP   = LinearSegmentedColormap.from_list("tanom", [
    "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0",
    "#ffffff",
    "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f",
], N=256)

# Rainfall anomaly (weekly): blue→white→red/brown
RANOM_BOUNDS = [-20,-15,-10,-5,0,5,10,15,20]
RANOM_CMAP   = LinearSegmentedColormap.from_list("ranom", [
    "#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3",
    "#ffffff",
    "#c7eae5","#80cdc1","#35978f","#01665e","#003c30",
], N=256)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_step(raw_dir, date_str, step, channels):
    """Load ensemble mean for one lead step across all available members."""
    accum = {ch: [] for ch in channels}
    lat = lon = None
    mem_dir = Path(raw_dir) / date_str / "member"
    if not mem_dir.exists():
        return {}, None, None
    for mem_path in sorted(mem_dir.iterdir()):
        f = mem_path / f"{step:02d}.nc"
        if not f.exists():
            continue
        da = xr.open_dataarray(str(f))
        lat = da.lat.values
        lon = da.lon.values
        for ch in channels:
            if ch in da.channel.values:
                accum[ch].append(da.sel(channel=ch).squeeze(drop=True).values)
    out = {ch: np.array(v).mean(axis=0) for ch, v in accum.items() if v}
    return out, lat, lon


def crop(arr, lat, lon, lat0, lat1, lon0, lon1):
    lm = (lat >= lat0) & (lat <= lat1)
    om = (lon >= lon0) & (lon <= lon1)
    return arr[np.ix_(lm, om)], lat[lm], lon[om]


# ── WB2 CLIMATOLOGY ───────────────────────────────────────────────────────────
WB2_VARS = {"t2m": "t2m", "z500": "z500", "tp": "tp24",
            "u850": "u850", "v850": "v850"}

def build_climo(init_date, nsteps, lat, lon):
    print("  Loading WB2 1990-2019 climatology …")
    try:
        from earth2studio.data import WB2Climatology
        wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr",
                             verbose=False)
    except Exception as e:
        print(f"  WARNING: WB2 unavailable ({e})")
        return None

    cache = {}
    wvars = list(dict.fromkeys(WB2_VARS.values()))
    for step in range(1, nsteps + 1):
        valid = init_date + datetime.timedelta(days=step)
        t     = datetime.datetime(2001, valid.month, valid.day, 0)
        da    = wb2(t, wvars)
        wlat  = da.lat.values; wlon = da.lon.values
        sc = {}
        for mch, wv in WB2_VARS.items():
            arr = da.sel(variable=wv).squeeze().values.astype(np.float32)
            if wv == "tp24":  arr = arr * 1000.0
            if wv == "z500":  arr = arr / 9.80665
            xda = xr.DataArray(arr, dims=["lat","lon"],
                               coords={"lat": wlat, "lon": wlon})
            sc[mch] = xda.interp(lat=lat, lon=lon, method="linear").values
        cache[step] = sc
        if step % 14 == 0:
            print(f"    … day {step}")
    print("  Done.\n")
    return cache


# ── ERPAS-STYLE BASE MAP ──────────────────────────────────────────────────────
def erpas_map(ax, lon0, lon1, lat0, lat1, title=None, fs_title=10):
    """White background, black borders — matches ERPAS style."""
    ax.set_extent([lon0, lon1, lat0, lat1], crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),    color="#b0cfe0", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),     color="#f5f5e8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="black", linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="black", linewidth=0.5, linestyle="-", zorder=4)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#555555", linewidth=0.3, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                      alpha=0.6, linestyle="--", zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.MultipleLocator(10)
    gl.ylocator   = mticker.MultipleLocator(8)
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabel_style = {"size": 7, "color": "black"}
    if title:
        ax.set_title(title, fontsize=fs_title, color="blue",
                     fontweight="bold", pad=3)


def frame_header(fig, init_date, valid_date, subtitle):
    """ERPAS-style red/black title at figure top."""
    fig.text(0.5, 0.97,
             f"FuXi-S2S   Forecast Valid Time = "
             f"00Z{valid_date.strftime('%d%b%Y').upper()}",
             ha="center", va="top", fontsize=12, color="red", fontweight="bold")
    fig.text(0.5, 0.935,
             f"Initial Condition : {init_date.strftime('%Y%m%d')}",
             ha="center", va="top", fontsize=10, color="black", fontweight="bold")
    fig.text(0.5, 0.905, subtitle,
             ha="center", va="top", fontsize=9, color="blue", fontweight="bold")


# ── PLOT 1: PRECIPITATION + 850hPa WIND (animated GIF) ───────────────────────
def make_prec_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps):
    print(f"\n── Precipitation + 850hPa Wind GIF ({nsteps} frames) ──")
    frames = []

    for step in range(1, nsteps + 1):
        data, lat, lon = load_step(raw_dir, date_str, step,
                                   ["tp","u850","v850"])
        if not data:
            continue
        valid = init_date + datetime.timedelta(days=step)

        # Domain: 20E–180, 20S–50N  (ERPAS style)
        tp,   lat_i, lon_i = crop(data["tp"],   lat, lon, -20, 50, 20, 180)
        u850, _,     _     = crop(data["u850"], lat, lon, -20, 50, 20, 180)
        v850, _,     _     = crop(data["v850"], lat, lon, -20, 50, 20, 180)

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        erpas_map(ax, 20, 180, -20, 50)

        # Shaded precip
        cf = ax.contourf(lon_i, lat_i, tp,
                         levels=PREC_BOUNDS, colors=PREC_COLORS,
                         transform=PROJ, extend="max", zorder=1)

        # 850 hPa wind vectors — skip every 3 (ERPAS density)
        skip = 3
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u850[::skip,::skip], v850[::skip,::skip],
                  transform=PROJ, scale=350, width=0.002,
                  color="black", alpha=0.85, zorder=5,
                  headwidth=3, headlength=3)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=PREC_CMAP, norm=PREC_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.04, shrink=0.7, aspect=30,
                          ticks=PREC_BOUNDS)
        cb.set_label("mm/day", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        frame_header(fig, init_date, valid,
                     "Rainfall (shaded, mm/day) & 850hPa winds (vector, 20→)")
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frames.append(Image.frombytes("RGBA",
                       fig.canvas.get_width_height(), buf).convert("RGB"))
        plt.close(fig)

        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")

    out = out_dir / f"prec_wind850_{date_str}.gif"
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {out}  ({out.stat().st_size/1024**2:.1f} MB)")


# ── PLOT 2: 850hPa VORTICITY + MSLP (animated GIF) ───────────────────────────
def make_vort_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps):
    print(f"\n── 850hPa Vorticity + MSLP GIF ({nsteps} frames) ──")
    frames = []

    for step in range(1, nsteps + 1):
        data, lat, lon = load_step(raw_dir, date_str, step,
                                   ["u850","v850","msl"])
        if not data:
            continue
        valid = init_date + datetime.timedelta(days=step)

        # Domain: 50E–110E, EQ–42N (ERPAS vorticity domain)
        u850, lat_i, lon_i = crop(data["u850"], lat, lon, 0, 42, 50, 110)
        v850 = crop(data["v850"], lat, lon, 0, 42, 50, 110)[0]
        msl  = crop(data["msl"],  lat, lon, 0, 42, 50, 110)[0] / 100.0  # Pa→hPa

        # Compute relative vorticity: dv/dx - du/dy
        # Approximate on regular 1.5° grid
        dx   = np.deg2rad(1.5) * 6371000 * np.cos(np.deg2rad(lat_i[:,None]))
        dy   = np.deg2rad(1.5) * 6371000
        dvdx = np.gradient(v850, axis=1) / dx
        dudy = np.gradient(u850, axis=0) / dy
        vort = (dvdx - dudy) * 1e5   # units: ×10⁻⁵ s⁻¹

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        erpas_map(ax, 50, 110, 0, 42)

        # Vorticity shading
        lim = 12
        cf  = ax.contourf(lon_i, lat_i, vort,
                          levels=np.linspace(-lim, lim, 25),
                          cmap=VORT_CMAP, transform=PROJ,
                          extend="both", zorder=1)

        # MSLP contours in blue
        cs = ax.contour(lon_i, lat_i, msl,
                        levels=np.arange(994, 1026, 2),
                        colors="blue", linewidths=0.6,
                        transform=PROJ, zorder=3)
        ax.clabel(cs, fmt="%d", fontsize=6, colors="blue", inline=True)

        cb = fig.colorbar(cf, ax=ax, orientation="horizontal",
                          pad=0.04, shrink=0.7, aspect=30,
                          ticks=np.arange(-12, 13, 3))
        cb.set_label("×10⁻⁵ s⁻¹", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        frame_header(fig, init_date, valid,
                     "850hPa Vorticity (shaded) &  mslp (contours, hPa)")
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frames.append(Image.frombytes("RGBA",
                       fig.canvas.get_width_height(), buf).convert("RGB"))
        plt.close(fig)

        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")

    out = out_dir / f"vort850_mslp_{date_str}.gif"
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {out}  ({out.stat().st_size/1024**2:.1f} MB)")


# ── PLOT 3: 200hPa DIVERGENCE + WINDS + 500hPa Z (animated GIF) ──────────────
def make_divg_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps):
    print(f"\n── 200hPa Divergence + Wind + Z500 GIF ({nsteps} frames) ──")
    frames = []

    for step in range(1, nsteps + 1):
        data, lat, lon = load_step(raw_dir, date_str, step,
                                   ["u200","v200","z500"])
        if not data:
            # FuXi has u200/v200 as u_component at 200hPa
            # channel names are u200, v200 — check availability
            continue
        valid = init_date + datetime.timedelta(days=step)

        # Global domain like ERPAS: 20E–180, 10S–90N
        u200, lat_i, lon_i = crop(data["u200"], lat, lon, -10, 90, 20, 180)
        v200 = crop(data["v200"], lat, lon, -10, 90, 20, 180)[0]
        z500 = crop(data["z500"], lat, lon, -10, 90, 20, 180)[0] / 9.80665

        # Divergence: du/dx + dv/dy
        dx   = np.deg2rad(1.5) * 6371000 * np.cos(np.deg2rad(lat_i[:,None]))
        dy   = np.deg2rad(1.5) * 6371000
        dudx = np.gradient(u200, axis=1) / dx
        dvdy = np.gradient(v200, axis=0) / dy
        divg = (dudx + dvdy) * 1e5   # ×10⁻⁵ s⁻¹

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        erpas_map(ax, 20, 180, -10, 90)

        lim = 4
        cf  = ax.contourf(lon_i, lat_i, divg,
                          levels=np.linspace(-lim, lim, 21),
                          cmap=DIV_CMAP, transform=PROJ,
                          extend="both", zorder=1)

        # Z500 contours
        cs = ax.contour(lon_i, lat_i, z500,
                        levels=np.arange(5300, 5960, 30),
                        colors="blue", linewidths=0.5,
                        transform=PROJ, zorder=3)
        ax.clabel(cs, fmt="%d", fontsize=5, colors="blue", inline=True)

        # 200hPa wind vectors
        skip = 3
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u200[::skip,::skip], v200[::skip,::skip],
                  transform=PROJ, scale=700, width=0.002,
                  color="black", alpha=0.7, zorder=5,
                  headwidth=2, headlength=2)

        cb = fig.colorbar(cf, ax=ax, orientation="horizontal",
                          pad=0.04, shrink=0.7, aspect=30,
                          ticks=np.arange(-4, 5, 1))
        cb.set_label("×10⁻⁵ s⁻¹", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        frame_header(fig, init_date, valid,
                     "200 hPa Divergence (shaded), Winds (vector, 50→) & 500mb GH (contours, m)")
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frames.append(Image.frombytes("RGBA",
                       fig.canvas.get_width_height(), buf).convert("RGB"))
        plt.close(fig)

        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")

    out = out_dir / f"divg200_wind_z500_{date_str}.gif"
    frames[0].save(str(out), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {out}  ({out.stat().st_size/1024**2:.1f} MB)")


# ── PLOT 4: WEEKLY ACTUAL + ANOMALY RAINFALL (static PNG) ────────────────────
def make_rf_weekly_png(raw_dir, date_str, init_date, out_dir, climo):
    print("\n── Weekly Rainfall PNG (actual + anomaly) ──")

    WEEKS = {1: range(1,8), 2: range(8,15), 3: range(15,22)}
    # India domain
    lat0,lat1,lon0,lon1 = 6, 38, 66, 100

    fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    fig.suptitle(
        f"MPME  Rainfall (mm/day)    IC={init_date.strftime('%Y%m%d')}\n"
        f"FuXi-S2S  vs  ERA5 1990–2019 climatology",
        fontsize=12, fontweight="bold", color="black", y=0.97)

    PREC_ACT_BOUNDS = [2, 4, 8, 12, 18, 25, 35]
    PREC_ACT_COLORS = ["#d9f0d3","#a8ddb5","#4eb3d3","#2b8cbe",
                       "#08589e","#084081","#02205e"]

    for col, (wk, steps) in enumerate(WEEKS.items()):
        # ── weekly mean ───────────────────────────────────────────────────────
        tp_days  = []
        tp_c_days = []
        lat_ref  = lon_ref = None

        for step in steps:
            d, lat, lon = load_step(raw_dir, date_str, step, ["tp"])
            if not d:
                continue
            lat_ref = lat; lon_ref = lon
            tp_i = crop(d["tp"], lat, lon, lat0, lat1, lon0, lon1)[0]
            tp_days.append(tp_i)
            if climo and step in climo and "tp" in climo[step]:
                tp_c_i = crop(climo[step]["tp"], lat, lon,
                              lat0, lat1, lon0, lon1)[0]
                tp_c_days.append(tp_c_i)

        if not tp_days:
            continue

        tp_mean  = np.mean(tp_days, axis=0)
        _, lat_i, lon_i = crop(tp_days[0], lat_ref, lon_ref,
                               lat0, lat1, lon0, lon1)
        # the crop already done above, just need lat_i/lon_i
        lm = (lat_ref >= lat0) & (lat_ref <= lat1)
        om = (lon_ref >= lon0) & (lon_ref <= lon1)
        lat_i = lat_ref[lm]; lon_i = lon_ref[om]

        # Date range label
        d0 = (init_date + datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date + datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        wk_label = f"Week{wk}: {d0}-{d1}"

        # ── Top row: actual ───────────────────────────────────────────────────
        ax = axes[0, col]
        erpas_map(ax, lon0, lon1, lat0, lat1,
                  title=wk_label, fs_title=9)
        cf = ax.contourf(lon_i, lat_i, tp_mean,
                         levels=PREC_ACT_BOUNDS, colors=PREC_ACT_COLORS,
                         transform=PROJ, extend="max", zorder=1)
        if col == 0:
            ax.text(-0.12, 0.5, "Actual\n(mm/day)", transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, fontweight="bold",
                    rotation=90, color="black")

        # ── Bottom row: anomaly ───────────────────────────────────────────────
        ax = axes[1, col]
        erpas_map(ax, lon0, lon1, lat0, lat1,
                  title=wk_label, fs_title=9)
        if tp_c_days:
            anom = tp_mean - np.mean(tp_c_days, axis=0)
            lim  = 20
            cf2  = ax.contourf(lon_i, lat_i, anom,
                               levels=np.linspace(-lim, lim, 17),
                               cmap=RANOM_CMAP, transform=PROJ,
                               extend="both", zorder=1)
            if col == 2:
                cb = plt.colorbar(cf2, ax=axes[1,:], orientation="horizontal",
                                  pad=0.06, shrink=0.6, aspect=30,
                                  ticks=np.arange(-20, 21, 5))
                cb.set_label("mm/day anomaly", fontsize=8)
                cb.ax.tick_params(labelsize=7)
        if col == 0:
            ax.text(-0.12, 0.5, "Anomaly\n(mm/day)", transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, fontweight="bold",
                    rotation=90, color="black")

    # Colorbar for actual rainfall (top row)
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(PREC_ACT_COLORS),
                               norm=BoundaryNorm(PREC_ACT_BOUNDS,
                               len(PREC_ACT_COLORS)))
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=axes[0,:], orientation="horizontal",
                       pad=0.06, shrink=0.6, aspect=30,
                       ticks=PREC_ACT_BOUNDS)
    cb2.set_label("mm/day", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.10, right=0.97, top=0.90,
                        bottom=0.12, hspace=0.22, wspace=0.08)
    out = out_dir / f"rf_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── PLOT 5: WEEKLY TMAX ANOMALY 4-PANEL (static PNG) ─────────────────────────
def make_tmax_anom_png(raw_dir, date_str, init_date, out_dir, climo):
    print("\n── Weekly Tmax Anomaly PNG (4 panels) ──")

    WEEKS4 = {1: range(1,8), 2: range(8,15), 3: range(15,22), 4: range(22,29)}
    lat0,lat1,lon0,lon1 = 7, 37, 67, 98

    fig, axes = plt.subplots(2, 2, figsize=(10, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    fig.suptitle(
        f"Maximum Temperature Anomaly (°C)  IC={init_date.strftime('%Y%m%d')}\n"
        "FuXi-S2S  vs  ERA5 1990–2019 climatology",
        fontsize=11, fontweight="bold", color="black", y=0.97)

    positions = [(0,0),(0,1),(1,0),(1,1)]

    for (r,c), (wk, steps) in zip(positions, WEEKS4.items()):
        t2m_days = []
        tc_days  = []
        lat_ref = lon_ref = None

        for step in steps:
            d, lat, lon = load_step(raw_dir, date_str, step, ["t2m"])
            if not d:
                continue
            lat_ref = lat; lon_ref = lon
            t2m_i = crop(d["t2m"], lat, lon, lat0, lat1, lon0, lon1)[0] - 273.15
            t2m_days.append(t2m_i)
            if climo and step in climo and "t2m" in climo[step]:
                tc_i = crop(climo[step]["t2m"], lat, lon,
                            lat0, lat1, lon0, lon1)[0] - 273.15
                tc_days.append(tc_i)

        if not t2m_days:
            continue

        # Tmax proxy: use daily T2m mean as Tmax (FuXi doesn't output Tmax)
        tmax_mean = np.mean(t2m_days, axis=0)
        lm = (lat_ref >= lat0) & (lat_ref <= lat1)
        om = (lon_ref >= lon0) & (lon_ref <= lon1)
        lat_i = lat_ref[lm]; lon_i = lon_ref[om]

        d0 = (init_date + datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date + datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        ax = axes[r, c]
        erpas_map(ax, lon0, lon1, lat0, lat1,
                  title=f"(Week{wk}: {d0}-{d1})", fs_title=8)

        if tc_days:
            anom = tmax_mean - np.mean(tc_days, axis=0)
            cf   = ax.contourf(lon_i, lat_i, anom,
                               levels=np.linspace(-10, 10, 21),
                               cmap=TANOM_CMAP, transform=PROJ,
                               extend="both", zorder=1)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=TANOM_CMAP,
                               norm=mcolors.Normalize(-10, 10))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal",
                      pad=0.05, shrink=0.65, aspect=30,
                      ticks=np.arange(-10, 11, 1))
    cb.set_label("°C  (T2m anomaly — proxy for Tmax anomaly)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.92,
                        bottom=0.10, hspace=0.18, wspace=0.08)
    out = out_dir / f"tmax_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
MODES = ["prec","vort","divg","rf_weekly","tmax_anom"]

def main():
    p = argparse.ArgumentParser(
        description="ERPAS-style plots from FuXi-S2S output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date",    required=True, help="Init date YYYYMMDD")
    p.add_argument("--raw_dir", default=str(RAW_DIR))
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: erpas_style/<date>/)")
    p.add_argument("--mode",    default=None, choices=MODES,
                   help="Single mode; omit = all")
    p.add_argument("--steps",   type=int, default=42, help="Lead days to animate")
    p.add_argument("--fps",     type=int, default=FPS)
    args = p.parse_args()

    init_date = datetime.date(int(args.date[:4]),
                              int(args.date[4:6]),
                              int(args.date[6:8]))

    out_dir = Path(args.out_dir) if args.out_dir \
              else OUT_DIR / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nERPAS-style plots  —  FuXi-S2S")
    print(f"  IC    : {init_date.strftime('%d %b %Y')} ({args.date})")
    print(f"  Raw   : {args.raw_dir}/{args.date}/member/")
    print(f"  Out   : {out_dir}")
    print(f"  Steps : {args.steps}   FPS: {args.fps}\n")

    # Check channels available
    d1, lat, lon = load_step(args.raw_dir, args.date, 1,
                             ["tp","u850","v850","msl","u200","v200","z500","t2m"])
    if lat is None:
        print(f"ERROR: no data at {args.raw_dir}/{args.date}/member/")
        sys.exit(1)
    print(f"  Available channels at step 1: {list(d1.keys())}\n")

    # Check if u200/v200 exist (FuXi output uses u200, v200 channel names)
    has_200 = "u200" in d1 and "v200" in d1

    # Build WB2 climo for static plots
    climo = None
    modes = [args.mode] if args.mode else MODES
    if any(m in modes for m in ["rf_weekly","tmax_anom"]):
        climo = build_climo(init_date, args.steps, lat, lon)

    for mode in modes:
        if mode == "prec":
            make_prec_gif(args.raw_dir, args.date, init_date,
                          out_dir, args.steps, args.fps)
        elif mode == "vort":
            make_vort_gif(args.raw_dir, args.date, init_date,
                          out_dir, args.steps, args.fps)
        elif mode == "divg":
            if not has_200:
                print("  SKIP divg — u200/v200 channels not found in model output")
            else:
                make_divg_gif(args.raw_dir, args.date, init_date,
                              out_dir, args.steps, args.fps)
        elif mode == "rf_weekly":
            make_rf_weekly_png(args.raw_dir, args.date, init_date,
                               out_dir, climo)
        elif mode == "tmax_anom":
            make_tmax_anom_png(args.raw_dir, args.date, init_date,
                               out_dir, climo)

    print(f"\nAll done!  →  {out_dir}")


if __name__ == "__main__":
    main()
