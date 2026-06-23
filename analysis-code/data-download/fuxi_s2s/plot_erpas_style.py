#!/usr/bin/env python3
"""
plot_erpas_style.py
===================
Reproduce ALL ERPAS forecast plots from FuXi-S2S output for visual comparison.
Uses SOI state shapefile for India panels. Ensemble mean of all members.

Plots produced (matching ERPAS one-to-one):
  GIFs (daily animated):
    prec_wind850_<DATE>.gif      Rainfall + 850hPa winds  (20E-180, 20S-50N)
    vort850_mslp_<DATE>.gif      850hPa vorticity + MSLP  (50E-110, 0-42N)
    divg200_wind_z500_<DATE>.gif 200hPa divergence + wind + Z500 (global)
    igpp_<DATE>.gif              Cyclogenesis probability  (40E-125, 0-40N)

  PNGs (weekly static):
    rf_weekly_<DATE>.png         2×3: India rainfall actual | anomaly wk1-3
    tmax_actual_weekly_<DATE>.png 2×2: India T2m actual weeks 1-4
    tmax_anom_weekly_<DATE>.png  2×2: India T2m anomaly weeks 1-4
    hw_weekly_<DATE>.png         4×2: Heat-stress probability weeks 1-4

Usage
-----
  python plot_erpas_style.py --date 20260617          # all plots
  python plot_erpas_style.py --date 20260617 --mode prec
  python plot_erpas_style.py --date 20260617 --steps 28 --fps 4
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
import cartopy.io.shapereader as shpreader
import numpy as np
import xarray as xr
from PIL import Image

warnings.filterwarnings("ignore")

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw")
OUT_DIR = Path("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/"
               "fuxi_s2s/erpas_style")
SOI_SHP = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"

SOI_CRS = ccrs.LambertConformal(
    central_longitude=80.0, central_latitude=24.0,
    standard_parallels=(12.472944, 35.172806),
    false_easting=4000000.0, false_northing=4000000.0,
)
PROJ = ccrs.PlateCarree()
FPS  = 4

# ── COLORMAPS (matching ERPAS visually) ───────────────────────────────────────
# Precipitation daily (GIF): tuned to FuXi-S2S output (~0–3 mm/day over India)
PREC_BOUNDS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
PREC_COLS   = ["#edf8e9","#bae4b3","#74c476","#31a354",
               "#2171b5","#08519c","#08306b"]
PREC_CMAP   = mcolors.ListedColormap(PREC_COLS)
PREC_NORM   = BoundaryNorm(PREC_BOUNDS, PREC_CMAP.N)

# Precipitation weekly total (PNG): daily×7 → mm/week, matches ERPAS scale
PREC_BOUNDS_WK = [2, 4, 8, 12, 18, 25, 35]
PREC_COLS_WK   = ["#d9f0d3","#a8ddb5","#4eb3d3","#2b8cbe",
                  "#08589e","#084081","#02205e"]
PREC_CMAP_WK   = mcolors.ListedColormap(PREC_COLS_WK)
PREC_NORM_WK   = BoundaryNorm(PREC_BOUNDS_WK, PREC_CMAP_WK.N)

# Vorticity / Divergence: dark-purple→blue→white→orange→dark-red
VDIV_CMAP = LinearSegmentedColormap.from_list("vdiv", [
    "#1a0030","#4a0080","#0000cc","#6699ff","#ccddff",
    "#ffffff",
    "#ffddaa","#ff8800","#cc2200","#660000",
], N=256)

# Tmax actual: white→yellow→orange→red→dark-red (ERPAS heat scale)
TMAX_BOUNDS = [0,10,20,25,28,30,32,34,36,38,40,42,44,46]
TMAX_COLS   = ["#ffffff","#ffffcc","#ffeda0","#fed976","#feb24c",
               "#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026",
               "#54001a","#2d0010","#10000a"]
TMAX_CMAP   = mcolors.ListedColormap(TMAX_COLS)
TMAX_NORM   = BoundaryNorm(TMAX_BOUNDS, TMAX_CMAP.N)

# Tmax anomaly: blue→white→red  ±10°C
TANOM_CMAP = LinearSegmentedColormap.from_list("tanom", [
    "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0",
    "#ffffff",
    "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f",
], N=256)

# Rainfall anomaly: brown→white→green
RANOM_CMAP = LinearSegmentedColormap.from_list("ranom", [
    "#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3",
    "#ffffff",
    "#c7eae5","#80cdc1","#35978f","#01665e","#003c30",
], N=256)

# Heat-stress probability: white→yellow→orange→red→magenta (ERPAS HW scale)
HW_BOUNDS = [20,30,50,70,90,100]
HW_COLS   = ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026","#7a0177"]
HW_CMAP   = mcolors.ListedColormap(HW_COLS)
HW_NORM   = BoundaryNorm(HW_BOUNDS, HW_CMAP.N)

# Cyclogenesis probability: cyan→green→yellow→red→magenta
IGPP_BOUNDS = [25,30,40,50,60,70,80,90,100]
IGPP_COLS   = ["#a6f2f2","#00cc44","#88dd00","#ffee00",
               "#ff9900","#ff3300","#cc0066","#880088"]
IGPP_CMAP   = mcolors.ListedColormap(IGPP_COLS)
IGPP_NORM   = BoundaryNorm(IGPP_BOUNDS, IGPP_CMAP.N)


# ── SHAPEFILE ─────────────────────────────────────────────────────────────────
def load_soi(shp=SOI_SHP):
    try:
        geoms = [r.geometry.simplify(5000)
                 for r in shpreader.Reader(shp).records()]
        print(f"  SOI: {len(geoms)} polygons")
        return geoms
    except Exception as e:
        print(f"  SOI failed ({e}) — using cartopy states")
        return None


def add_soi(ax, soi_geoms):
    """Add SOI state borders to an axes (India domain only)."""
    if soi_geoms:
        ax.add_geometries(soi_geoms, crs=SOI_CRS,
                          facecolor="none", edgecolor="#333333",
                          linewidth=0.5, zorder=4)
    else:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor="#555555", linewidth=0.3, zorder=4)


# ── BASE MAPS ─────────────────────────────────────────────────────────────────
def base_map(ax, lon0, lon1, lat0, lat1, with_states=False, soi=None):
    """ERPAS-style white/light-blue map."""
    ax.set_extent([lon0, lon1, lat0, lat1], crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), color="#b8d4e8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),  color="#f5f5e8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="black", linewidth=0.7, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="black", linewidth=0.5, zorder=5)
    if with_states:
        add_soi(ax, soi)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                      alpha=0.5, linestyle="--", zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.MultipleLocator(10)
    gl.ylocator   = mticker.MultipleLocator(8)
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabel_style = {"size": 7, "color": "black"}


def india_map(ax, soi, lon0=67, lon1=98, lat0=7, lat1=37):
    """Tight India panel with SOI state borders."""
    base_map(ax, lon0, lon1, lat0, lat1, with_states=True, soi=soi)
    # Tighter gridlines for India panels
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="grey",
                      alpha=0.4, linestyle="--", zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.MultipleLocator(7)
    gl.ylocator   = mticker.MultipleLocator(6)
    gl.xlabel_style = {"size": 6, "color": "black"}
    gl.ylabel_style = {"size": 6, "color": "black"}


def frame_header(fig, init_date, valid_date, subtitle):
    """ERPAS-style red/black/blue three-line header."""
    fig.text(0.5, 0.975,
             f"FuXi-S2S   Forecast Valid Time = "
             f"00Z{valid_date.strftime('%d%b%Y').upper()}",
             ha="center", va="top", fontsize=12,
             color="red", fontweight="bold")
    fig.text(0.5, 0.945,
             f"Initial Condition : {init_date.strftime('%Y%m%d')}",
             ha="center", va="top", fontsize=10,
             color="black", fontweight="bold")
    fig.text(0.5, 0.918, subtitle,
             ha="center", va="top", fontsize=9,
             color="blue", fontweight="bold")


def png_header(fig, init_date, title, subtitle=""):
    fig.text(0.5, 0.975, title,
             ha="center", va="top", fontsize=12,
             color="red", fontweight="bold")
    fig.text(0.5, 0.945,
             f"FuXi-S2S  ·  IC={init_date.strftime('%Y%m%d')}  ·  "
             "ERA5 1990–2019 climatology",
             ha="center", va="top", fontsize=9,
             color="black")
    if subtitle:
        fig.text(0.5, 0.920, subtitle,
                 ha="center", va="top", fontsize=8, color="blue")


# ── DATA I/O ──────────────────────────────────────────────────────────────────
# Set by --members CLI arg; None = use all members
_MEMBER_FILTER: set = None   # type: ignore


def _member_dirs(raw_dir, date_str):
    """Yield member Path objects, filtered by _MEMBER_FILTER."""
    mem_dir = Path(raw_dir) / date_str / "member"
    if not mem_dir.exists():
        return
    for mem in sorted(mem_dir.iterdir()):
        if _MEMBER_FILTER is None or int(mem.name) in _MEMBER_FILTER:
            yield mem


def load_step(raw_dir, date_str, step, channels):
    """Ensemble mean across selected member subdirectories."""
    accum = {ch: [] for ch in channels}
    lat = lon = None
    mem_dir = Path(raw_dir) / date_str / "member"
    if not mem_dir.exists():
        return {}, None, None
    for mem in _member_dirs(raw_dir, date_str):
        f = mem / f"{step:02d}.nc"
        if not f.exists():
            continue
        da  = xr.open_dataarray(str(f))
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


def weekly_mean(raw_dir, date_str, steps, channels):
    """Average a list of steps into a weekly mean. Returns (dict, lat, lon)."""
    accum = {ch: [] for ch in channels}
    lat = lon = None
    for step in steps:
        d, la, lo = load_step(raw_dir, date_str, step, channels)
        if not d:
            continue
        lat, lon = la, lo
        for ch in channels:
            if ch in d:
                accum[ch].append(d[ch])
    out = {ch: np.mean(v, axis=0) for ch, v in accum.items() if v}
    return out, lat, lon


def render(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return img


def save_gif(frames, path, fps):
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {path}  ({path.stat().st_size/1024**2:.1f} MB)")


# ── WB2 CLIMATOLOGY ───────────────────────────────────────────────────────────
WB2_MAP = {"t2m":"t2m", "z500":"z500", "tp":"tp24", "u850":"u850", "v850":"v850"}

def build_climo(init_date, nsteps, lat, lon):
    print("  Loading WB2 1990–2019 climatology …")
    try:
        from earth2studio.data import WB2Climatology
        wb2 = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr",
                             verbose=False)
    except Exception as e:
        print(f"  WARNING: WB2 unavailable ({e})")
        return None
    wvars = list(dict.fromkeys(WB2_MAP.values()))
    cache = {}
    for step in range(1, nsteps + 1):
        valid = init_date + datetime.timedelta(days=step)
        t     = datetime.datetime(2001, valid.month, valid.day, 0)
        da    = wb2(t, wvars)
        wlat, wlon = da.lat.values, da.lon.values
        sc = {}
        for mch, wv in WB2_MAP.items():
            arr = da.sel(variable=wv).squeeze().values.astype(np.float32)
            if wv == "tp24": arr *= 1000.0
            if wv == "z500": arr /= 9.80665
            xda = xr.DataArray(arr, dims=["lat","lon"],
                               coords={"lat": wlat, "lon": wlon})
            sc[mch] = xda.interp(lat=lat, lon=lon, method="linear").values
        cache[step] = sc
        if step % 14 == 0:
            print(f"    … day {step}")
    print("  Done.\n")
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. PRECIPITATION + 850hPa WIND (GIF) ─────────────────────────────────────
def make_prec_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[1/8] Precipitation + 850hPa Wind GIF …")
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["tp","u850","v850"])
        if not d: continue
        valid = init_date + datetime.timedelta(days=step)
        tp,   lat_i, lon_i = crop(d["tp"],   lat, lon, -20, 50, 20, 180)
        u850 = crop(d["u850"], lat, lon, -20, 50, 20, 180)[0]
        v850 = crop(d["v850"], lat, lon, -20, 50, 20, 180)[0]

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 20, 180, -20, 50)
        add_soi(ax, soi)
        ax.contourf(lon_i, lat_i, tp, levels=PREC_BOUNDS, colors=PREC_COLS,
                    transform=PROJ, extend="max", zorder=1)
        skip=3
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u850[::skip,::skip], v850[::skip,::skip],
                  transform=PROJ, scale=350, width=0.002, color="black",
                  alpha=0.85, zorder=5, headwidth=3, headlength=3)
        sm = plt.cm.ScalarMappable(cmap=PREC_CMAP, norm=PREC_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28, ticks=PREC_BOUNDS)
        cb.set_label("mm/day", fontsize=8); cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Rainfall (shaded, mm/day) & 850hPa winds (vector, 20→)")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"prec_wind850_{date_str}.gif", fps)


# ── 2. 850hPa VORTICITY + MSLP (GIF) ─────────────────────────────────────────
def make_vort_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[2/8] 850hPa Vorticity + MSLP GIF …")
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["u850","v850","msl"])
        if not d: continue
        valid = init_date + datetime.timedelta(days=step)
        u850, lat_i, lon_i = crop(d["u850"], lat, lon, 0, 42, 50, 110)
        v850 = crop(d["v850"], lat, lon, 0, 42, 50, 110)[0]
        msl  = crop(d["msl"],  lat, lon, 0, 42, 50, 110)[0] / 100.0

        dx   = np.deg2rad(1.5)*6371000*np.cos(np.deg2rad(lat_i[:,None]))
        dy   = np.deg2rad(1.5)*6371000
        vort = (np.gradient(v850,axis=1)/dx - np.gradient(u850,axis=0)/dy)*1e5

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 50, 110, 0, 42, with_states=True, soi=soi)
        ax.contourf(lon_i, lat_i, vort, levels=np.linspace(-12,12,25),
                    cmap=VDIV_CMAP, transform=PROJ, extend="both", zorder=1)
        cs = ax.contour(lon_i, lat_i, msl, levels=np.arange(994,1026,2),
                        colors="blue", linewidths=0.6, transform=PROJ, zorder=3)
        ax.clabel(cs, fmt="%d", fontsize=6, colors="blue", inline=True)
        sm = plt.cm.ScalarMappable(cmap=VDIV_CMAP,
                                   norm=mcolors.Normalize(-12,12))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28,
                          ticks=np.arange(-12,13,3))
        cb.set_label("×10⁻⁵ s⁻¹", fontsize=8); cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "850hPa Vorticity (shaded)  &  mslp (contours, hPa)")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"vort850_mslp_{date_str}.gif", fps)


# ── 3. 200hPa DIVERGENCE + WIND + Z500 (GIF) ─────────────────────────────────
def make_divg_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[3/8] 200hPa Divergence + Wind + Z500 GIF …")
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step,
                                ["u200","v200","z500"])
        if not d or "u200" not in d: continue
        valid = init_date + datetime.timedelta(days=step)
        u200, lat_i, lon_i = crop(d["u200"], lat, lon, -10, 90, 20, 180)
        v200 = crop(d["v200"], lat, lon, -10, 90, 20, 180)[0]
        z500 = crop(d["z500"], lat, lon, -10, 90, 20, 180)[0] / 9.80665

        dx   = np.deg2rad(1.5)*6371000*np.cos(np.deg2rad(lat_i[:,None]))
        dy   = np.deg2rad(1.5)*6371000
        divg = (np.gradient(u200,axis=1)/dx + np.gradient(v200,axis=0)/dy)*1e5

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 20, 180, -10, 90)
        add_soi(ax, soi)
        ax.contourf(lon_i, lat_i, divg, levels=np.linspace(-4,4,21),
                    cmap=VDIV_CMAP, transform=PROJ, extend="both", zorder=1)
        cs = ax.contour(lon_i, lat_i, z500, levels=np.arange(5300,5960,30),
                        colors="blue", linewidths=0.5, transform=PROJ, zorder=3)
        ax.clabel(cs, fmt="%d", fontsize=5, colors="blue", inline=True)
        skip=3
        ax.quiver(lon_i[::skip], lat_i[::skip],
                  u200[::skip,::skip], v200[::skip,::skip],
                  transform=PROJ, scale=700, width=0.002, color="black",
                  alpha=0.65, zorder=5, headwidth=2, headlength=2)
        sm = plt.cm.ScalarMappable(cmap=VDIV_CMAP, norm=mcolors.Normalize(-4,4))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28,
                          ticks=np.arange(-4,5,1))
        cb.set_label("×10⁻⁵ s⁻¹", fontsize=8); cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "200hPa Divergence (shaded), Winds (vector, 50→) & 500mb GH (contours, m)")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"divg200_wind_z500_{date_str}.gif", fps)


# ── 4. CYCLOGENESIS PROBABILITY (GIF) ─────────────────────────────────────────
def make_igpp_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    """
    Proxy for ERPAS IGPP: ensemble spread of 850hPa vorticity as a
    cyclogenesis-activity indicator. Fraction of members with vort > threshold.
    """
    print(f"\n[4/8] Cyclogenesis Probability (vorticity spread) GIF …")
    frames = []
    members = list(_member_dirs(raw_dir, date_str))
    if len(members) < 2:
        print("  Need ≥2 members for probability — skipping")
        return

    for step in range(1, nsteps+1):
        # Load each member separately for spread
        vort_members = []
        lat_ref = lon_ref = None
        for mem in members:
            f = mem / f"{step:02d}.nc"
            if not f.exists(): continue
            da = xr.open_dataarray(str(f))
            u = da.sel(channel="u850").squeeze(drop=True).values
            v = da.sel(channel="v850").squeeze(drop=True).values
            lat = da.lat.values; lon = da.lon.values
            lat_ref = lat; lon_ref = lon
            dx = np.deg2rad(1.5)*6371000*np.cos(np.deg2rad(lat[:,None]))
            dy = np.deg2rad(1.5)*6371000
            vort = (np.gradient(v,axis=1)/dx - np.gradient(u,axis=0)/dy)*1e5
            vort_members.append(vort)

        if not vort_members: continue
        valid = init_date + datetime.timedelta(days=step)

        # Probability of cyclonic anomaly (vort > 3×10⁻⁵ s⁻¹)
        vort_stack = np.array(vort_members)
        prob = (vort_stack > 3).mean(axis=0) * 100.0  # %

        # Domain: 40E-125E, 0-40N
        prob_i, lat_i, lon_i = crop(prob, lat_ref, lon_ref, 0, 40, 40, 125)

        fig = plt.figure(figsize=(10, 6), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 40, 125, 0, 40)
        add_soi(ax, soi)
        # Only plot where probability > 25%
        prob_masked = np.ma.masked_less(prob_i, 25)
        cf = ax.contourf(lon_i, lat_i, prob_masked,
                         levels=IGPP_BOUNDS, colors=IGPP_COLS,
                         transform=PROJ, extend="max", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=IGPP_CMAP, norm=IGPP_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28, ticks=IGPP_BOUNDS)
        cb.set_label("Probability (%)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Cyclogenesis & Vorticity Activity Probability from FuXi Ensemble")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"igpp_{date_str}.gif", fps)


# ── 5. WEEKLY ACTUAL + ANOMALY RAINFALL (2×3 PNG) ────────────────────────────
def make_rf_weekly(raw_dir, date_str, init_date, out_dir, climo, soi):
    print("\n[5/8] Weekly Rainfall PNG (actual | anomaly) …")
    # 4 weeks, daily avg ×7 → weekly total mm/week (matches ERPAS scale)
    WEEKS = {1: range(1,8), 2: range(8,15), 3: range(15,22), 4: range(22,29)}
    la0,la1,lo0,lo1 = 6, 38, 66, 100

    fig, axes = plt.subplots(2, 4, figsize=(18, 9),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    png_header(fig, init_date, "Rainfall Forecast  (mm/week)",
               "Top: Actual  ·  Bottom: Anomaly vs ERA5 1990–2019  ·  FuXi-S2S ensemble mean")

    for col, (wk, steps) in enumerate(WEEKS.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["tp"])
        if not d: continue
        # ×7: convert daily mean → weekly total (mm/week)
        tp_i = crop(d["tp"], lat, lon, la0, la1, lo0, lo1)[0] * 7
        lm = (lat>=la0)&(lat<=la1); om=(lon>=lo0)&(lon<=lo1)
        lat_i=lat[lm]; lon_i=lon[om]
        d0 = (init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        label = f"Week{wk}\n{d0}–{d1}"

        # Top: actual (mm/week)
        ax = axes[0, col]
        india_map(ax, soi, lo0, lo1, la0, la1)
        ax.contourf(lon_i, lat_i, tp_i, levels=PREC_BOUNDS_WK, colors=PREC_COLS_WK,
                    transform=PROJ, extend="max", zorder=1)
        ax.set_title(label, fontsize=9, color="blue", fontweight="bold", pad=3)
        if col == 0:
            ax.text(-0.16, 0.5, "Actual\n(mm/week)", transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, fontweight="bold",
                    rotation=90, color="black")

        # Bottom: anomaly (mm/week)
        ax = axes[1, col]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if climo:
            c_days = [climo[s]["tp"] for s in steps if s in climo and "tp" in climo[s]]
            if c_days:
                tp_c = crop(np.mean(c_days, axis=0), lat, lon, la0, la1, lo0, lo1)[0] * 7
                anom = tp_i - tp_c
                ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-30, 30, 17),
                            cmap=RANOM_CMAP, transform=PROJ,
                            extend="both", zorder=1)
        ax.set_title(label, fontsize=9, color="blue", fontweight="bold", pad=3)
        if col == 0:
            ax.text(-0.16, 0.5, "Anomaly\n(mm/week)", transform=ax.transAxes,
                    ha="right", va="center", fontsize=9, fontweight="bold",
                    rotation=90, color="black")

    # Colorbars
    sm_act = plt.cm.ScalarMappable(cmap=PREC_CMAP_WK, norm=PREC_NORM_WK)
    sm_act.set_array([])
    fig.colorbar(sm_act, ax=axes[0,:], orientation="horizontal",
                 pad=0.10, shrink=0.5, aspect=32, ticks=PREC_BOUNDS_WK,
                 label="mm/week").ax.tick_params(labelsize=7)

    sm_an = plt.cm.ScalarMappable(cmap=RANOM_CMAP, norm=mcolors.Normalize(-30, 30))
    sm_an.set_array([])
    fig.colorbar(sm_an, ax=axes[1,:], orientation="horizontal",
                 pad=0.10, shrink=0.5, aspect=32,
                 ticks=np.arange(-30, 31, 10),
                 label="mm/week anomaly").ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.89,
                        bottom=0.16, hspace=0.22, wspace=0.06)
    out = out_dir/f"rf_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 6. WEEKLY ACTUAL TMAX (2×2 PNG) ──────────────────────────────────────────
def make_tmax_actual(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[6/8] Weekly T2m Actual PNG (weeks 1–4) …")
    WEEKS4 = {1:range(1,8), 2:range(8,15), 3:range(15,22), 4:range(22,29)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig, axes = plt.subplots(2, 2, figsize=(10, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    png_header(fig, init_date, "Maximum Temperature Actual (°C)",
               "T2m used as proxy for daily Tmax  ·  FuXi-S2S ensemble mean")

    for (r,c), (wk, steps) in zip([(0,0),(0,1),(1,0),(1,1)], WEEKS4.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        if not d: continue
        t2m_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)[0] - 273.15
        lm=(lat>=la0)&(lat<=la1); om=(lon>=lo0)&(lon<=lo1)
        lat_i=lat[lm]; lon_i=lon[om]
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        ax = axes[r,c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        ax.contourf(lon_i, lat_i, t2m_i,
                    levels=TMAX_BOUNDS, colors=TMAX_COLS,
                    transform=PROJ, extend="both", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}–{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=3)

    sm = plt.cm.ScalarMappable(cmap=TMAX_CMAP, norm=TMAX_NORM)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal",
                 pad=0.09, shrink=0.6, aspect=28, ticks=TMAX_BOUNDS,
                 label="°C").ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.91,
                        bottom=0.14, hspace=0.18, wspace=0.08)
    out = out_dir/f"tmax_actual_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 7. WEEKLY TMAX ANOMALY (2×2 PNG) ─────────────────────────────────────────
def make_tmax_anom(raw_dir, date_str, init_date, out_dir, climo, soi):
    print("\n[7/8] Weekly T2m Anomaly PNG (weeks 1–4) …")
    WEEKS4 = {1:range(1,8), 2:range(8,15), 3:range(15,22), 4:range(22,29)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig, axes = plt.subplots(2, 2, figsize=(10, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    png_header(fig, init_date, "Maximum Temperature Anomaly (°C)",
               "T2m anomaly vs ERA5 1990–2019 climatology")

    for (r,c), (wk, steps) in zip([(0,0),(0,1),(1,0),(1,1)], WEEKS4.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        if not d: continue
        t2m_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)[0] - 273.15
        lm=(lat>=la0)&(lat<=la1); om=(lon>=lo0)&(lon<=lo1)
        lat_i=lat[lm]; lon_i=lon[om]
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        ax = axes[r,c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if climo:
            c_days=[climo[s]["t2m"] for s in steps if s in climo and "t2m" in climo[s]]
            if c_days:
                t2m_c = crop(np.mean(c_days,axis=0)-273.15, lat, lon,
                             la0, la1, lo0, lo1)[0]
                anom  = t2m_i - t2m_c
                ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-10,10,21),
                            cmap=TANOM_CMAP, transform=PROJ,
                            extend="both", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}–{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=3)

    sm = plt.cm.ScalarMappable(cmap=TANOM_CMAP, norm=mcolors.Normalize(-10,10))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal",
                 pad=0.09, shrink=0.6, aspect=28,
                 ticks=np.arange(-10,11,1),
                 label="°C  (T2m anomaly)").ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.91,
                        bottom=0.14, hspace=0.18, wspace=0.08)
    out = out_dir/f"tmax_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 8. HEAT-STRESS PROBABILITY (4×2 PNG) ─────────────────────────────────────
def make_hw_weekly(raw_dir, date_str, init_date, out_dir, soi):
    """
    Proxy for ERPAS heat-wave probability:
    fraction of members with T2m > 35°C (HW) or > 40°C (SHW) per grid cell.
    """
    print("\n[8/8] Heat-Stress Probability PNG (weeks 1–4) …")
    WEEKS4 = {1:range(1,8), 2:range(8,15), 3:range(15,22), 4:range(22,29)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    members = list(_member_dirs(raw_dir, date_str))

    fig, axes = plt.subplots(4, 2, figsize=(10, 18),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    png_header(fig, init_date,
               f"Prediction from IC={init_date.strftime('%Y%m%d')}",
               "Heat Stress (T2m>35°C) & Severe Heat Stress (T2m>40°C) — % ensemble")

    for row, (wk, steps) in enumerate(WEEKS4.items()):
        d0s = (init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1s = (init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        wlabel = f"W{wk} Lead\n({d0s}–{d1s})"

        # Collect daily t2m from each member → weekly mean per member
        hw_prob  = []   # fraction > 35°C
        shw_prob = []   # fraction > 40°C
        lat_ref = lon_ref = None

        for mem in members:
            mem_days = []
            for step in steps:
                f = mem / f"{step:02d}.nc"
                if not f.exists(): continue
                da = xr.open_dataarray(str(f))
                t  = da.sel(channel="t2m").squeeze(drop=True).values - 273.15
                lat_ref = da.lat.values; lon_ref = da.lon.values
                mem_days.append(t)
            if not mem_days: continue
            t_wmean = np.mean(mem_days, axis=0)
            hw_prob.append(t_wmean > 35)
            shw_prob.append(t_wmean > 40)

        for col, (prob_list, thresh_label) in enumerate(
                [(hw_prob,"HW (>35°C)"), (shw_prob,"SHW (>40°C)")]):
            ax = axes[row, col]
            india_map(ax, soi, lo0, lo1, la0, la1)
            ax.set_title(f"{thresh_label}", fontsize=8,
                         color="red" if col==0 else "darkred",
                         fontweight="bold", pad=2)

            # Y-axis week label on left panel
            if col == 0:
                ax.text(-0.18, 0.5, wlabel, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8,
                        fontweight="bold", color="red", rotation=0)

            if prob_list and lat_ref is not None:
                prob_arr = np.mean(prob_list, axis=0)*100.0
                prob_i   = crop(prob_arr, lat_ref, lon_ref,
                                la0, la1, lo0, lo1)[0]
                lm=(lat_ref>=la0)&(lat_ref<=la1)
                om=(lon_ref>=lo0)&(lon_ref<=lo1)
                lat_i=lat_ref[lm]; lon_i=lon_ref[om]
                prob_m = np.ma.masked_less(prob_i, 20)
                ax.contourf(lon_i, lat_i, prob_m,
                            levels=HW_BOUNDS, colors=HW_COLS,
                            transform=PROJ, extend="max", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=HW_CMAP, norm=HW_NORM)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal",
                 pad=0.09, shrink=0.5, aspect=28, ticks=HW_BOUNDS,
                 label="% ensemble members").ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.92,
                        bottom=0.10, hspace=0.20, wspace=0.10)
    out = out_dir/f"hw_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
MODES = ["prec","vort","divg","igpp","rf_weekly",
         "tmax_actual","tmax_anom","hw"]

def main():
    p = argparse.ArgumentParser(
        description="ERPAS-style plots from FuXi-S2S (all 8 plot types)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date",    required=True, help="Init date YYYYMMDD")
    p.add_argument("--raw_dir", default=str(RAW_DIR))
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: erpas_style/<date>/)")
    p.add_argument("--soi_shp", default=SOI_SHP,
                   help="SOI STATE_BOUNDARY.shp path")
    p.add_argument("--mode",    default=None, choices=MODES,
                   help="Single mode; omit = all 8")
    p.add_argument("--steps",   type=int, default=42)
    p.add_argument("--fps",     type=int, default=FPS)
    p.add_argument("--members", type=int, nargs="+", default=None,
                   help="Member indices to use (e.g. 0 1 2); default = all")
    args = p.parse_args()

    # Wire up member filter before any data loading
    global _MEMBER_FILTER
    _MEMBER_FILTER = set(args.members) if args.members else None

    init_date = datetime.date(int(args.date[:4]),
                              int(args.date[4:6]),
                              int(args.date[6:8]))
    out_dir = Path(args.out_dir) if args.out_dir \
              else OUT_DIR / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_label = f"members {sorted(_MEMBER_FILTER)}" if _MEMBER_FILTER else "all members"
    print(f"\nERPAS-style plots  —  FuXi-S2S  ({args.date})  [{mem_label}]")
    print(f"  Raw : {args.raw_dir}/{args.date}/member/")
    print(f"  Out : {out_dir}\n")

    soi = load_soi(args.soi_shp)

    # Reference grid
    d1, lat, lon = load_step(args.raw_dir, args.date, 1,
                             ["tp","u850","v850","msl","u200","v200",
                              "z500","t2m"])
    if lat is None:
        print(f"ERROR: no data at {args.raw_dir}/{args.date}/member/")
        sys.exit(1)
    print(f"  Channels available: {list(d1.keys())}")
    has_200 = "u200" in d1 and "v200" in d1

    # Climo needed for anomaly plots
    modes = [args.mode] if args.mode else MODES
    need_climo = any(m in modes for m in ["rf_weekly","tmax_anom"])
    climo = build_climo(init_date, args.steps, lat, lon) if need_climo else None

    for mode in modes:
        if   mode == "prec":       make_prec_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "vort":       make_vort_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "divg":
            if not has_200: print("  SKIP divg — u200/v200 not found")
            else: make_divg_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "igpp":       make_igpp_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "rf_weekly":  make_rf_weekly(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "tmax_actual":make_tmax_actual(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "tmax_anom":  make_tmax_anom(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "hw":         make_hw_weekly(args.raw_dir, args.date, init_date, out_dir, soi)

    print(f"\nAll done!  →  {out_dir}")


if __name__ == "__main__":
    main()
