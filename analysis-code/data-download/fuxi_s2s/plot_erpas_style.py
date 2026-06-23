#!/usr/bin/env python3
"""
plot_erpas_style.py
===================
ALL ERPAS forecast plot types reproduced from FuXi-S2S output.

Plots produced (ERPAS layout / colorbars):
  GIFs (daily animated):
    prec_wind850_<D>.gif         Rainfall + 850hPa winds    (20E-180, 20S-50N)
    vort850_mslp_<D>.gif         850hPa vorticity + MSLP    (50E-110, 0-42N)
    divg200_wind_z500_<D>.gif    200hPa divg + wind + Z500  (20E-180, 10S-90N)
    rh700_wind_<D>.gif           700hPa RH + winds          (20E-180, 20S-40N)
    igpp_<D>.gif                 Cyclogenesis probability   (40E-125, 0-40N)
    temp_actual_<D>.gif          T2m actual (India, min+max)(67E-98, 7-37N)
    temp_anom_<D>.gif            T2m anomaly (India)        (67E-98, 7-37N)
    hw_daily_<D>.gif             HW/SHW probability (India) (67E-98, 7-37N)

  PNGs (weekly static):
    rf_weekly_<D>.png            1×4  India actual rainfall  mm/day
    rf_anom_weekly_<D>.png       2×2  India rainfall anomaly mm/day
    igpp_weekly_<D>.png          2×2  Cyclogenesis prob      (40E-125,0-40N)
    tmax_actual_weekly_<D>.png   2×2  India T2m actual       °C
    tmax_anom_weekly_<D>.png     2×2  India T2m anomaly      °C
    temp_weekly_<D>.png          2×4  India tmin+tmax actual °C
    temp_anom_weekly_<D>.png     2×4  India tmin+tmax anom   °C
    hw_weekly_<D>.png            4×2  Heat-stress prob       %

Usage
-----
  python plot_erpas_style.py --date 20260617
  python plot_erpas_style.py --date 20260617 --mode prec
  python plot_erpas_style.py --date 20260617 --members 0 1 2
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

# ── COLORMAPS ─────────────────────────────────────────────────────────────────

# Rainfall daily GIF: tuned to FuXi-S2S output (~0-3 mm/day)
PREC_BOUNDS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
PREC_COLS   = ["#edf8e9","#bae4b3","#74c476","#31a354",
               "#2171b5","#08519c","#08306b"]
PREC_CMAP   = mcolors.ListedColormap(PREC_COLS)
PREC_NORM   = BoundaryNorm(PREC_BOUNDS, PREC_CMAP.N)

# Rainfall weekly actual: FuXi-adapted scale (tp ~0.1-3 mm/day)
PREC_BOUNDS_WK = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
PREC_COLS_WK   = ["#edf8e9","#bae4b3","#74c476","#31a354",
                  "#2171b5","#08519c","#08306b"]
PREC_CMAP_WK   = mcolors.ListedColormap(PREC_COLS_WK)
PREC_NORM_WK   = BoundaryNorm(PREC_BOUNDS_WK, PREC_CMAP_WK.N)

# Rainfall anomaly weekly: -20,-15,-10,-5,-2, 2, 5, 10, 15, 20 mm/day
RANOM_BOUNDS = [-20,-15,-10,-5,-2, 2, 5, 10, 15, 20]
RANOM_COLS   = ["#543005","#8c510a","#bf812d","#dfc27d",
                "#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e"]
RANOM_CMAP   = mcolors.ListedColormap(RANOM_COLS)
RANOM_NORM   = BoundaryNorm(RANOM_BOUNDS, RANOM_CMAP.N)

# 700hPa RH: yellow=dry, blue=moist
RH_BOUNDS = [15, 30, 40, 45, 50, 55, 60, 70, 85, 95]
RH_COLS   = ["#c8601b","#e8a048","#f5d56a","#fffcd8","#d0eef5",
             "#97cce4","#4d9dc2","#2161a6","#083080"]
RH_CMAP   = mcolors.ListedColormap(RH_COLS)
RH_NORM   = BoundaryNorm(RH_BOUNDS, RH_CMAP.N)

# Vorticity / Divergence: dark-purple → blue → white → orange → dark-red
VDIV_CMAP = LinearSegmentedColormap.from_list("vdiv", [
    "#1a0030","#4a0080","#0000cc","#6699ff","#ccddff",
    "#ffffff",
    "#ffddaa","#ff8800","#cc2200","#660000",
], N=256)

# Tmax actual (ERPAS bounds °C): 0,10,20,25,28,30,32,34,36,38,40,42,44,46
TMAX_BOUNDS = [0,10,20,25,28,30,32,34,36,38,40,42,44,46]
TMAX_COLS   = ["#ffffff","#ffffcc","#ffeda0","#fed976","#feb24c",
               "#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026",
               "#54001a","#2d0010","#10000a"]
TMAX_CMAP   = mcolors.ListedColormap(TMAX_COLS)
TMAX_NORM   = BoundaryNorm(TMAX_BOUNDS, TMAX_CMAP.N)

# Tmin actual (ERPAS bounds °C): 0,4,8,12,16,20,24,26,28,30,32,34,36
TMIN_BOUNDS = [0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 36]
TMIN_COLS   = ["#ffffff","#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c",
               "#fc4e2a","#e31a1c","#bd0026","#800026","#54001a","#10000a"]
TMIN_CMAP   = mcolors.ListedColormap(TMIN_COLS)
TMIN_NORM   = BoundaryNorm(TMIN_BOUNDS, TMIN_CMAP.N)

# Temperature anomaly: blue → white → red  ±10°C
TANOM_CMAP = LinearSegmentedColormap.from_list("tanom", [
    "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0",
    "#ffffff",
    "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f",
], N=256)
TANOM_LEVELS = [-10,-9,-7,-5,-3,-1, 0, 1, 3, 5, 7, 9, 10]

# Heat-stress probability: white → yellow → orange → red → magenta
HW_BOUNDS = [20, 30, 50, 70, 90, 100]
HW_COLS   = ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026","#7a0177"]
HW_CMAP   = mcolors.ListedColormap(HW_COLS)
HW_NORM   = BoundaryNorm(HW_BOUNDS, HW_CMAP.N)

# Cyclogenesis probability: cyan → green → yellow → red → magenta
IGPP_BOUNDS = [25, 30, 40, 50, 60, 70, 80, 90, 100]
IGPP_COLS   = ["#a6f2f2","#00cc44","#88dd00","#ffee00",
               "#ff9900","#ff3300","#cc0066","#880088"]
IGPP_CMAP   = mcolors.ListedColormap(IGPP_COLS)
IGPP_NORM   = BoundaryNorm(IGPP_BOUNDS, IGPP_CMAP.N)

# OLR: low OLR = active convection (blue), high OLR = suppressed (orange/red)
OLR_BOUNDS = [150, 180, 200, 210, 220, 230, 240, 260, 280, 300]
OLR_COLS   = ["#000066","#0000cc","#4444ff","#88aaff","#ccddff",
              "#ffffff","#ffeecc","#ffaa44","#ff4400","#aa0000"]
OLR_CMAP   = mcolors.ListedColormap(OLR_COLS)
OLR_NORM   = BoundaryNorm(OLR_BOUNDS, OLR_CMAP.N)

# TCWV: dry=light yellow/green, moist=dark blue
TCWV_BOUNDS = [20, 30, 40, 45, 50, 55, 60, 65, 70]
TCWV_COLS   = ["#ffffb2","#c7e9c0","#78c679","#31a354","#a6bddb",
               "#74a9cf","#2b8cbe","#0570b0","#023858"]
TCWV_CMAP   = mcolors.ListedColormap(TCWV_COLS)
TCWV_NORM   = BoundaryNorm(TCWV_BOUNDS, TCWV_CMAP.N)

# Z500 anomaly: blue → white → red (ridge/trough)
Z500_ANOM_CMAP = LinearSegmentedColormap.from_list("z500anom", [
    "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0",
    "#ffffff",
    "#fddbc7","#f4a582","#d6604d","#b2182b","#67001f",
], N=256)

# Wind shear (200-850): white=low TC-favorable, green→orange→red=unfavorable
SHEAR_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 40]
SHEAR_COLS   = ["#ffffff","#c7e9c0","#74c476","#ffffb2",
                "#fecc5c","#fd8d3c","#e31a1c","#99000d"]
SHEAR_CMAP   = mcolors.ListedColormap(SHEAR_COLS)
SHEAR_NORM   = BoundaryNorm(SHEAR_BOUNDS, SHEAR_CMAP.N)

# SST: cool blue → warm red (Indian Ocean range)
SST_BOUNDS = [22, 24, 26, 27, 28, 29, 30, 31, 32]
SST_COLS   = ["#0000aa","#0055ff","#00aaff","#00ddff",
              "#aaffaa","#ffff00","#ffaa00","#ff4400","#cc0000"]
SST_CMAP   = mcolors.ListedColormap(SST_COLS)
SST_NORM   = BoundaryNorm(SST_BOUNDS, SST_CMAP.N)


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
    if soi_geoms:
        ax.add_geometries(soi_geoms, crs=SOI_CRS,
                          facecolor="none", edgecolor="#333333",
                          linewidth=0.5, zorder=4)
    else:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor="#555555", linewidth=0.3, zorder=4)


# ── BASE MAPS ─────────────────────────────────────────────────────────────────
def base_map(ax, lon0, lon1, lat0, lat1, with_states=False, soi=None):
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
    base_map(ax, lon0, lon1, lat0, lat1, with_states=True, soi=soi)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="grey",
                      alpha=0.4, linestyle="--", zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlocator   = mticker.MultipleLocator(7)
    gl.ylocator   = mticker.MultipleLocator(6)
    gl.xlabel_style = {"size": 6, "color": "black"}
    gl.ylabel_style = {"size": 6, "color": "black"}


def frame_header(fig, init_date, valid_date, subtitle):
    """ERPAS-style red/black/blue three-line GIF header."""
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
    """Place 2-3 line header for PNG plots. Call AFTER subplots_adjust."""
    fig.text(0.5, 0.980, title,
             ha="center", va="top", fontsize=12,
             color="red", fontweight="bold")
    fig.text(0.5, 0.955,
             f"FuXi-S2S  ·  IC={init_date.strftime('%Y%m%d')}  ·  "
             "ERA5 1990-2019 climatology",
             ha="center", va="top", fontsize=9, color="black")
    if subtitle:
        fig.text(0.5, 0.933, subtitle,
                 ha="center", va="top", fontsize=8, color="blue")


# ── DATA I/O ──────────────────────────────────────────────────────────────────
_MEMBER_FILTER: set = None   # type: ignore


def _member_dirs(raw_dir, date_str):
    mem_dir = Path(raw_dir) / date_str / "member"
    if not mem_dir.exists():
        return
    for mem in sorted(mem_dir.iterdir()):
        if _MEMBER_FILTER is None or int(mem.name) in _MEMBER_FILTER:
            yield mem


def load_step(raw_dir, date_str, step, channels):
    """Ensemble mean across selected members for given step."""
    accum = {ch: [] for ch in channels}
    lat = lon = None
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


def q_to_rh(q700, t700, p_hPa=700.0):
    """Compute RH (%) from specific humidity and temperature at p_hPa."""
    T_C = t700 - 273.15
    es = 6.1078 * np.exp(17.2694 * T_C / (T_C + 237.29))
    e  = q700 * p_hPa / (0.622 + 0.378 * q700)
    return np.clip(e / es * 100.0, 0, 100)


def render(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return img


def save_gif(frames, path, fps):
    if not frames:
        print(f"  WARNING: no frames — {path} not written")
        return
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    print(f"  → {path}  ({path.stat().st_size/1024**2:.1f} MB)")


# ── WB2 CLIMATOLOGY ───────────────────────────────────────────────────────────
WB2_MAP = {"t2m":"t2m", "z500":"z500", "tp":"tp24", "u850":"u850", "v850":"v850"}

def build_climo(init_date, nsteps, lat, lon):
    print("  Loading WB2 1990-2019 climatology …")
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
    print(f"\n[1] Precipitation + 850hPa Wind GIF …")
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
        skip = 3
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
    print(f"\n[2] 850hPa Vorticity + MSLP GIF …")
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
        sm = plt.cm.ScalarMappable(cmap=VDIV_CMAP, norm=mcolors.Normalize(-12,12))
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
    print(f"\n[3] 200hPa Divergence + Wind + Z500 GIF …")
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
        skip = 3
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


# ── 4. 700hPa RH + WIND (GIF) ────────────────────────────────────────────────
def make_rh_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[4] 700hPa RH + Wind GIF …")
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step,
                                ["q700","t700","u700","v700"])
        if not d or "q700" not in d or "t700" not in d: continue
        valid = init_date + datetime.timedelta(days=step)

        q700, lat_i, lon_i = crop(d["q700"], lat, lon, -20, 40, 20, 180)
        t700 = crop(d["t700"], lat, lon, -20, 40, 20, 180)[0]
        rh   = q_to_rh(q700, t700, 700.0)

        fig = plt.figure(figsize=(10, 7.5), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 20, 180, -20, 40)
        ax.contourf(lon_i, lat_i, rh, levels=RH_BOUNDS, colors=RH_COLS,
                    transform=PROJ, extend="both", zorder=1)

        if "u700" in d and "v700" in d:
            u700 = crop(d["u700"], lat, lon, -20, 40, 20, 180)[0]
            v700 = crop(d["v700"], lat, lon, -20, 40, 20, 180)[0]
            skip = 3
            ax.quiver(lon_i[::skip], lat_i[::skip],
                      u700[::skip,::skip], v700[::skip,::skip],
                      transform=PROJ, scale=350, width=0.002, color="black",
                      alpha=0.85, zorder=5, headwidth=3, headlength=3)

        sm = plt.cm.ScalarMappable(cmap=RH_CMAP, norm=RH_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28, ticks=RH_BOUNDS)
        cb.set_label("RH (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "700hPa Relative humidity (%)  &  700hPa winds (vector, 20→)")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"rh700_wind_{date_str}.gif", fps)


# ── 5. CYCLOGENESIS PROBABILITY daily (GIF) ───────────────────────────────────
def make_igpp_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[5] Cyclogenesis Probability GIF …")
    frames = []
    members = list(_member_dirs(raw_dir, date_str))
    if len(members) < 2:
        print("  Need ≥2 members — skipping")
        return

    for step in range(1, nsteps+1):
        vort_members = []
        lat_ref = lon_ref = None
        for mem in members:
            f = mem / f"{step:02d}.nc"
            if not f.exists(): continue
            da  = xr.open_dataarray(str(f))
            u   = da.sel(channel="u850").squeeze(drop=True).values
            v   = da.sel(channel="v850").squeeze(drop=True).values
            lat = da.lat.values; lon = da.lon.values
            lat_ref = lat; lon_ref = lon
            dx   = np.deg2rad(1.5)*6371000*np.cos(np.deg2rad(lat[:,None]))
            dy   = np.deg2rad(1.5)*6371000
            vort = (np.gradient(v,axis=1)/dx - np.gradient(u,axis=0)/dy)*1e5
            vort_members.append(vort)

        if not vort_members: continue
        valid = init_date + datetime.timedelta(days=step)
        vort_stack = np.array(vort_members)
        prob = (vort_stack > 3).mean(axis=0) * 100.0
        prob_i, lat_i, lon_i = crop(prob, lat_ref, lon_ref, 0, 40, 40, 125)

        fig = plt.figure(figsize=(10, 6), facecolor="white")
        ax  = fig.add_subplot(1,1,1, projection=PROJ)
        base_map(ax, 40, 125, 0, 40)
        add_soi(ax, soi)
        prob_masked = np.ma.masked_less(prob_i, 25)
        ax.contourf(lon_i, lat_i, prob_masked,
                    levels=IGPP_BOUNDS, colors=IGPP_COLS,
                    transform=PROJ, extend="max", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=IGPP_CMAP, norm=IGPP_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.65, aspect=28, ticks=IGPP_BOUNDS[:-1])
        cb.set_label("Probability (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Cyclogenesis & Evolution probability from FuXi-IGPP")
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.05, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"igpp_{date_str}.gif", fps)


# ── 6. TEMPERATURE ACTUAL daily (GIF, India, tmin+tmax) ──────────────────────
def make_temp_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[6] Temperature Actual daily GIF (India) …")
    la0,la1,lo0,lo1 = 7, 37, 67, 98
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["t2m"])
        if not d: continue
        valid = init_date + datetime.timedelta(days=step)
        t2m_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)[0] - 273.15
        lat_i = lat[(lat>=la0)&(lat<=la1)]
        lon_i = lon[(lon>=lo0)&(lon<=lo1)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw=dict(projection=PROJ),
                                 facecolor="white")
        plt.subplots_adjust(top=0.85, bottom=0.22, left=0.04, right=0.97,
                            wspace=0.08)

        for col, (cmap, norm, bounds, label) in enumerate([
            (TMIN_CMAP, TMIN_NORM, TMIN_BOUNDS, "Minimum Temp. Actual in °C"),
            (TMAX_CMAP, TMAX_NORM, TMAX_BOUNDS, "Maximum Temp. Actual in °C"),
        ]):
            ax = axes[col]
            india_map(ax, soi, lo0, lo1, la0, la1)
            ax.contourf(lon_i, lat_i, t2m_i, levels=bounds, colors=list(cmap.colors),
                        transform=PROJ, extend="both", zorder=1)
            ax.set_title(label, fontsize=10, color="blue", fontweight="bold", pad=3)

            cax = fig.add_axes([0.07+col*0.50, 0.08, 0.42, 0.03])
            sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds)
            cb.ax.tick_params(labelsize=6)

        fig.text(0.5, 0.970,
                 f"FuXi-S2S  Forecast Valid Time = 00Z{valid.strftime('%d%b%Y').upper()}",
                 ha="center", va="top", fontsize=11, color="red", fontweight="bold")
        fig.text(0.5, 0.935,
                 f"Initial Condition : {init_date.strftime('%Y%m%d')}",
                 ha="center", va="top", fontsize=9, color="black", fontweight="bold")
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"temp_actual_{date_str}.gif", fps)


# ── 7. TEMPERATURE ANOMALY daily (GIF, India) ─────────────────────────────────
def make_temp_anom_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps,
                       climo, soi):
    print(f"\n[7] Temperature Anomaly daily GIF (India) …")
    if climo is None:
        print("  No climo available — skipping")
        return
    la0,la1,lo0,lo1 = 7, 37, 67, 98
    frames = []
    for step in range(1, nsteps+1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["t2m"])
        if not d or step not in climo: continue
        valid = init_date + datetime.timedelta(days=step)
        t2m_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)[0] - 273.15
        t2m_c = crop(climo[step]["t2m"] - 273.15, lat, lon, la0, la1, lo0, lo1)[0]
        anom  = t2m_i - t2m_c
        lat_i = lat[(lat>=la0)&(lat<=la1)]
        lon_i = lon[(lon>=lo0)&(lon<=lo1)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw=dict(projection=PROJ),
                                 facecolor="white")
        plt.subplots_adjust(top=0.85, bottom=0.22, left=0.04, right=0.97,
                            wspace=0.08)

        for col, label in enumerate(["Minimum Temp. Anomaly in °C",
                                     "Maximum Temp. Anomaly in °C"]):
            ax = axes[col]
            india_map(ax, soi, lo0, lo1, la0, la1)
            ax.contourf(lon_i, lat_i, anom, levels=TANOM_LEVELS,
                        cmap=TANOM_CMAP, transform=PROJ, extend="both", zorder=1)
            ax.set_title(label, fontsize=10, color="blue", fontweight="bold", pad=3)

        cax = fig.add_axes([0.20, 0.07, 0.60, 0.03])
        sm  = plt.cm.ScalarMappable(cmap=TANOM_CMAP, norm=mcolors.Normalize(-10,10))
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TANOM_LEVELS)
        cb.set_label("°C", fontsize=8); cb.ax.tick_params(labelsize=6)

        fig.text(0.5, 0.970,
                 f"FuXi-S2S  Forecasted Temp Anomaly  Valid Time=00Z{valid.strftime('%d%b%Y').upper()}",
                 ha="center", va="top", fontsize=10, color="red", fontweight="bold")
        fig.text(0.5, 0.935,
                 f"Initial Condition : {init_date.strftime('%Y%m%d')}",
                 ha="center", va="top", fontsize=9, color="black", fontweight="bold")
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"temp_anom_{date_str}.gif", fps)


# ── 8. HEAT-STRESS daily (GIF, India, HW + SHW) ───────────────────────────────
def make_hw_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    print(f"\n[8] Heat-Stress Probability daily GIF …")
    members = list(_member_dirs(raw_dir, date_str))
    la0,la1,lo0,lo1 = 7, 37, 67, 98
    frames = []
    for step in range(1, nsteps+1):
        hw_prob = []; shw_prob = []; lat_ref = lon_ref = None
        for mem in members:
            f = mem / f"{step:02d}.nc"
            if not f.exists(): continue
            da = xr.open_dataarray(str(f))
            t  = da.sel(channel="t2m").squeeze(drop=True).values - 273.15
            lat_ref = da.lat.values; lon_ref = da.lon.values
            hw_prob.append(t > 30); shw_prob.append(t > 34)

        if not hw_prob: continue
        valid = init_date + datetime.timedelta(days=step)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw=dict(projection=PROJ),
                                 facecolor="white")
        plt.subplots_adjust(top=0.85, bottom=0.22, left=0.04, right=0.97,
                            wspace=0.08)

        for col, (prob_list, label) in enumerate([
            (hw_prob,  "Heat Wave (T2m > 35°C)"),
            (shw_prob, "Severe Heat Wave (T2m > 40°C)"),
        ]):
            ax = axes[col]
            india_map(ax, soi, lo0, lo1, la0, la1)
            prob_arr = np.mean(prob_list, axis=0)*100.0
            prob_i   = crop(prob_arr, lat_ref, lon_ref, la0, la1, lo0, lo1)[0]
            prob_m   = np.ma.masked_less(prob_i, 20)
            lat_i = lat_ref[(lat_ref>=la0)&(lat_ref<=la1)]
            lon_i = lon_ref[(lon_ref>=lo0)&(lon_ref<=lo1)]
            ax.contourf(lon_i, lat_i, prob_m, levels=HW_BOUNDS, colors=HW_COLS,
                        transform=PROJ, extend="max", zorder=1)
            ax.set_title(label, fontsize=10,
                         color="red" if col==0 else "darkred",
                         fontweight="bold", pad=3)

        cax = fig.add_axes([0.20, 0.08, 0.60, 0.03])
        sm  = plt.cm.ScalarMappable(cmap=HW_CMAP, norm=HW_NORM)
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                           ticks=HW_BOUNDS[:-1])
        cb.set_label("Probability (%)", fontsize=8); cb.ax.tick_params(labelsize=7)

        fig.text(0.5, 0.970,
                 f"Forecast Valid Time = 00Z{valid.strftime('%d%b%Y').upper()}  "
                 f"IC: {init_date.strftime('%Y%m%d')}",
                 ha="center", va="top", fontsize=10, color="blue", fontweight="bold")
        fig.text(0.5, 0.940,
                 "Probability of Occurrence for:  Heat Wave  |  Severe Heat Wave",
                 ha="center", va="top", fontsize=9, color="black")
        frames.append(render(fig))
        if step % 7 == 0 or step == 1: print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir/f"hw_daily_{date_str}.gif", fps)


# ── 9. WEEKLY ACTUAL RAINFALL (2×3 PNG, 6 weeks) ─────────────────────────────
def make_rf_weekly(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[9] Weekly Actual Rainfall PNG (weeks 1-6) …")
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    la0,la1,lo0,lo1 = 6, 38, 66, 100
    POS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.04, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.06)
    png_header(fig, init_date, "FuXi-S2S Actual Rainfall (mm/day)",
               "Weekly mean  ·  India  ·  Weeks 1-6")

    for (r,c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["tp"])
        d0 = (init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r, c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if d:
            tp_i, lat_i, lon_i = crop(d["tp"], lat, lon, la0, la1, lo0, lo1)
            ax.contourf(lon_i, lat_i, tp_i, levels=PREC_BOUNDS_WK, colors=PREC_COLS_WK,
                        transform=PROJ, extend="max", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=PREC_CMAP_WK, norm=PREC_NORM_WK)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=PREC_BOUNDS_WK)
    cb.set_label("mm/day", fontsize=9); cb.ax.tick_params(labelsize=8)
    out = out_dir/f"rf_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 9b. 6-WEEK MONSOON RAINFALL EVOLUTION (2×3 PNG) ──────────────────────────
def make_rf_weekly6(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[9b] 6-Week Monsoon Rainfall PNG (2×3) …")
    WEEKS6 = {
        1: range(1, 8), 2: range(8, 15),
        3: range(15,22), 4: range(22,29),
        5: range(29,36), 6: range(36,43),
    }
    # Wider domain to track monsoon advance: 60E-100E, 5N-40N
    la0,la1,lo0,lo1 = 5, 40, 60, 102

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.04, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.06)
    png_header(fig, init_date,
               "FuXi-S2S Monsoon Rainfall — 6-Week Evolution (mm/day)",
               "Weekly mean  ·  India + neighbourhood")

    positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    for (r,c), (wk, steps) in zip(positions, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["tp"])
        d0 = (init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        ax = axes[r, c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if d:
            tp_i = crop(d["tp"], lat, lon, la0, la1, lo0, lo1)[0]
            lat_i = lat[(lat>=la0)&(lat<=la1)]
            lon_i = lon[(lon>=lo0)&(lon<=lo1)]
            ax.contourf(lon_i, lat_i, tp_i,
                        levels=PREC_BOUNDS_WK, colors=PREC_COLS_WK,
                        transform=PROJ, extend="max", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=10,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=PREC_CMAP_WK, norm=PREC_NORM_WK)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=PREC_BOUNDS_WK)
    cb.set_label("mm/day", fontsize=9); cb.ax.tick_params(labelsize=8)

    out = out_dir/f"rf_weekly6_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 10. WEEKLY RAINFALL ANOMALY (2×3 PNG, 6 weeks) ───────────────────────────
def make_rf_anom_weekly(raw_dir, date_str, init_date, out_dir, climo, soi):
    print("\n[10] Weekly Rainfall Anomaly PNG (weeks 1-6) …")
    if climo is None:
        print("  No climo — skipping")
        return
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    POS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0,la1,lo0,lo1 = 6, 38, 66, 100

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "FuXi-S2S Rainfall Anomaly (mm/day)",
               "vs ERA5 1990-2019 climatology  ·  India  ·  Weeks 1-6")

    for (r,c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["tp"])
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r,c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if d:
            tp_i, lat_i, lon_i = crop(d["tp"], lat, lon, la0, la1, lo0, lo1)
            c_days = [climo[s]["tp"] for s in steps if s in climo and "tp" in climo[s]]
            if c_days:
                tp_c = crop(np.mean(c_days, axis=0), lat, lon, la0, la1, lo0, lo1)[0]
                ax.contourf(lon_i, lat_i, tp_i - tp_c,
                            levels=RANOM_BOUNDS, colors=RANOM_COLS,
                            transform=PROJ, extend="both", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=RANOM_CMAP, norm=RANOM_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=RANOM_BOUNDS)
    cb.ax.tick_params(labelsize=7)
    out = out_dir/f"rf_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 11. IGPP WEEKLY (2×3 PNG, 6 weeks) ───────────────────────────────────────
def make_igpp_weekly(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[11] IGPP Weekly PNG (weeks 1-6) …")
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    POS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    members = list(_member_dirs(raw_dir, date_str))
    if len(members) < 2:
        print("  Need ≥2 members — skipping")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.10)
    png_header(fig, init_date,
               "Cyclogenesis & Evolution Probability (%) from FuXi-IGPP",
               "Ensemble fraction with 850hPa vorticity > 3×10⁻⁵ s⁻¹  ·  Weeks 1-6")

    for (r,c), (wk, steps) in zip(POS, WEEKS6.items()):
        d0s=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1s=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        vort_days = []
        lat_ref = lon_ref = None
        for mem in members:
            for step in steps:
                f = mem / f"{step:02d}.nc"
                if not f.exists(): continue
                da  = xr.open_dataarray(str(f))
                u   = da.sel(channel="u850").squeeze(drop=True).values
                v   = da.sel(channel="v850").squeeze(drop=True).values
                lat = da.lat.values; lon = da.lon.values
                lat_ref = lat; lon_ref = lon
                dx   = np.deg2rad(1.5)*6371000*np.cos(np.deg2rad(lat[:,None]))
                dy   = np.deg2rad(1.5)*6371000
                vort = (np.gradient(v,axis=1)/dx - np.gradient(u,axis=0)/dy)*1e5
                vort_days.append(vort)

        ax = axes[r,c]
        base_map(ax, 40, 125, 0, 40)
        ax.set_title(f"(W{wk}: {d0s}-{d1s})", fontsize=8,
                     color="blue", fontweight="bold", pad=4)
        if vort_days and lat_ref is not None:
            prob = (np.array(vort_days) > 3).mean(axis=0)*100.0
            prob_i, lat_i, lon_i = crop(prob, lat_ref, lon_ref, 0, 40, 40, 125)
            prob_m = np.ma.masked_less(prob_i, 25)
            ax.contourf(lon_i, lat_i, prob_m, levels=IGPP_BOUNDS, colors=IGPP_COLS,
                        transform=PROJ, extend="max", zorder=1)

    cax = fig.add_axes([0.20, 0.05, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=IGPP_CMAP, norm=IGPP_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                       ticks=IGPP_BOUNDS[:-1])
    cb.set_label("(%)", fontsize=9); cb.ax.tick_params(labelsize=8)
    out = out_dir/f"igpp_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 12. WEEKLY T2m ACTUAL (2×3 PNG, 6 weeks) ─────────────────────────────────
def make_tmax_actual(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[12] Weekly T2m Actual PNG (weeks 1-6) …")
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    POS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "Maximum Temperature Actual (°C)",
               "T2m proxy for Tmax  ·  FuXi-S2S ensemble mean  ·  Weeks 1-6")

    for (r,c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r,c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if d:
            t2m_i, lat_i, lon_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)
            ax.contourf(lon_i, lat_i, t2m_i - 273.15,
                        levels=TMAX_BOUNDS, colors=TMAX_COLS,
                        transform=PROJ, extend="both", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=TMAX_CMAP, norm=TMAX_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TMAX_BOUNDS)
    cb.set_label("°C", fontsize=9); cb.ax.tick_params(labelsize=7)
    out = out_dir/f"tmax_actual_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 13. WEEKLY T2m ANOMALY (2×3 PNG, 6 weeks) ────────────────────────────────
def make_tmax_anom(raw_dir, date_str, init_date, out_dir, climo, soi):
    print("\n[13] Weekly T2m Anomaly PNG (weeks 1-6) …")
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    POS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.88,
                        bottom=0.12, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "Maximum Temperature Anomaly (°C)",
               "T2m anomaly vs ERA5 1990-2019 climatology  ·  Weeks 1-6")

    for (r,c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r,c]
        india_map(ax, soi, lo0, lo1, la0, la1)
        if d and climo:
            t2m_i, lat_i, lon_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)
            c_days = [climo[s]["t2m"] for s in steps if s in climo and "t2m" in climo[s]]
            if c_days:
                t2m_c = crop(np.mean(c_days,axis=0), lat, lon, la0, la1, lo0, lo1)[0]
                ax.contourf(lon_i, lat_i, (t2m_i - t2m_c),
                            levels=TANOM_LEVELS, cmap=TANOM_CMAP,
                            transform=PROJ, extend="both", zorder=1)
        ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=TANOM_CMAP, norm=mcolors.Normalize(-10,10))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TANOM_LEVELS)
    cb.set_label("°C  (T2m anomaly)", fontsize=9); cb.ax.tick_params(labelsize=8)
    out = out_dir/f"tmax_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 14. COMBINED WEEKLY TMIN + TMAX ACTUAL (2×6, common colorbar) ────────────
def make_temp_weekly(raw_dir, date_str, init_date, out_dir, soi):
    """Row 0 = T2m with Tmin colorscale, Row 1 = T2m with Tmax colorscale.
    Single shared colorbar at bottom (TMAX scale covers the full range)."""
    print("\n[14] Combined Weekly Tmin+Tmax Actual PNG (weeks 1-6) …")
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig = plt.figure(figsize=(26, 11), facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88,
                        bottom=0.12, hspace=0.20, wspace=0.06)

    for col, (wk, steps) in enumerate(WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        for row, (cmap, norm, bounds, side) in enumerate([
            (TMIN_CMAP, TMIN_NORM, TMIN_BOUNDS, "Min Temp\nActual (°C)"),
            (TMAX_CMAP, TMAX_NORM, TMAX_BOUNDS, "Max Temp\nActual (°C)"),
        ]):
            ax = fig.add_subplot(2, 6, row*6 + col + 1, projection=PROJ)
            india_map(ax, soi, lo0, lo1, la0, la1)
            if d:
                t2m_i, lat_i, lon_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)
                ax.contourf(lon_i, lat_i, t2m_i - 273.15, levels=bounds,
                            colors=list(cmap.colors),
                            transform=PROJ, extend="both", zorder=1)
            ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=7,
                         color="blue", fontweight="bold", pad=3)
            if col == 0:
                ax.text(-0.20, 0.5, side, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8,
                        fontweight="bold", color="red")

    # Single colorbar spanning full width — use TMAX scale (wider range)
    cax = fig.add_axes([0.10, 0.04, 0.80, 0.022])
    sm  = plt.cm.ScalarMappable(cmap=TMAX_CMAP, norm=TMAX_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TMAX_BOUNDS)
    cb.set_label("T2m (°C)", fontsize=9); cb.ax.tick_params(labelsize=7)

    png_header(fig, init_date,
               "Minimum Temperature Actual (°C)  |  Maximum Temperature Actual (°C)",
               "T2m as proxy  ·  FuXi-S2S ensemble mean  ·  Weeks 1-6")
    out = out_dir/f"temp_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 15. COMBINED WEEKLY TMIN + TMAX ANOMALY (2×6, common colorbar) ───────────
def make_temp_anom_weekly(raw_dir, date_str, init_date, out_dir, climo, soi):
    print("\n[15] Combined Weekly Tmin+Tmax Anomaly PNG (weeks 1-6) …")
    if climo is None:
        print("  No climo — skipping")
        return
    WEEKS6 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98

    fig = plt.figure(figsize=(26, 11), facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88,
                        bottom=0.12, hspace=0.20, wspace=0.06)

    for col, (wk, steps) in enumerate(WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["t2m"])
        d0=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")

        for row, side in enumerate(["Min Temp\nAnomaly (°C)", "Max Temp\nAnomaly (°C)"]):
            ax = fig.add_subplot(2, 6, row*6 + col + 1, projection=PROJ)
            india_map(ax, soi, lo0, lo1, la0, la1)
            if d and climo:
                t2m_i, lat_i, lon_i = crop(d["t2m"], lat, lon, la0, la1, lo0, lo1)
                c_days=[climo[s]["t2m"] for s in steps if s in climo and "t2m" in climo[s]]
                if c_days:
                    t2m_c = crop(np.mean(c_days,axis=0), lat, lon, la0, la1, lo0, lo1)[0]
                    ax.contourf(lon_i, lat_i, t2m_i - t2m_c,
                                levels=TANOM_LEVELS, cmap=TANOM_CMAP,
                                transform=PROJ, extend="both", zorder=1)
            ax.set_title(f"(Week{wk}: {d0}-{d1})", fontsize=7,
                         color="blue", fontweight="bold", pad=3)
            if col == 0:
                ax.text(-0.20, 0.5, side, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8,
                        fontweight="bold", color="red")

    cax = fig.add_axes([0.12, 0.055, 0.76, 0.022])
    sm  = plt.cm.ScalarMappable(cmap=TANOM_CMAP, norm=mcolors.Normalize(-10,10))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TANOM_LEVELS)
    cb.set_label("Temperature Anomaly (°C)", fontsize=9); cb.ax.tick_params(labelsize=7)

    png_header(fig, init_date,
               "Minimum Temp Anomaly (°C)    |    Maximum Temp Anomaly (°C)",
               f"T2m anomaly vs ERA5 1990-2019 climatology  ·  IC={init_date.strftime('%Y%m%d')}")
    out = out_dir/f"temp_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 16. HEAT-STRESS PROBABILITY WEEKLY (6×2 PNG) ─────────────────────────────
def make_hw_weekly(raw_dir, date_str, init_date, out_dir, soi):
    print("\n[16] Heat-Stress Probability Weekly PNG (weeks 1-6) …")
    WEEKS4 = {1:range(1,8), 2:range(8,15), 3:range(15,22),
              4:range(22,29), 5:range(29,36), 6:range(36,43)}
    la0,la1,lo0,lo1 = 7, 37, 67, 98
    members = list(_member_dirs(raw_dir, date_str))

    fig, axes = plt.subplots(6, 2, figsize=(10, 26),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.15, right=0.97, top=0.92,
                        bottom=0.08, hspace=0.22, wspace=0.10)
    png_header(fig, init_date,
               f"Prediction from IC={init_date.strftime('%Y%m%d')}",
               "HW (T2m>30°C) & SHW (T2m>34°C) — % ensemble  [t2m proxy for Tmax]")

    for row, (wk, steps) in enumerate(WEEKS4.items()):
        d0s=(init_date+datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1s=(init_date+datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        wlabel = f"W{wk} Lead\n({d0s}-{d1s})"

        hw_prob = []; shw_prob = []; lat_ref = lon_ref = None
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
            hw_prob.append(t_wmean > 30)
            shw_prob.append(t_wmean > 34)

        for col, (prob_list, thresh_label) in enumerate(
                [(hw_prob,"HW (t2m>30°C)"), (shw_prob,"SHW (t2m>34°C)")]):
            ax = axes[row, col]
            india_map(ax, soi, lo0, lo1, la0, la1)
            ax.set_title(thresh_label, fontsize=8,
                         color="red" if col==0 else "darkred",
                         fontweight="bold", pad=2)
            if col == 0:
                ax.text(-0.25, 0.5, wlabel, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8,
                        fontweight="bold", color="red", rotation=0)
            if prob_list and lat_ref is not None:
                prob_arr = np.mean(prob_list, axis=0)*100.0
                prob_i   = crop(prob_arr, lat_ref, lon_ref, la0, la1, lo0, lo1)[0]
                lat_i=lat_ref[(lat_ref>=la0)&(lat_ref<=la1)]
                lon_i=lon_ref[(lon_ref>=lo0)&(lon_ref<=lo1)]
                prob_m = np.ma.masked_less(prob_i, 20)
                ax.contourf(lon_i, lat_i, prob_m, levels=HW_BOUNDS, colors=HW_COLS,
                            transform=PROJ, extend="max", zorder=1)

    cax = fig.add_axes([0.20, 0.03, 0.60, 0.020])
    sm  = plt.cm.ScalarMappable(cmap=HW_CMAP, norm=HW_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                       ticks=HW_BOUNDS[:-1])
    cb.set_label("% ensemble members", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    out = out_dir/f"hw_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 17. OLR (ttr proxy) daily GIF ─────────────────────────────────────────────
def make_olr_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    """OLR = -ttr. Low OLR = deep convection (blue), high = suppressed (red)."""
    print(f"\n[17] OLR (ttr) daily GIF …")
    frames    = []
    olr_scale = None
    for step in range(1, nsteps + 1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["ttr"])
        if not d or "ttr" not in d:
            continue
        valid = init_date + datetime.timedelta(days=step)
        ttr   = d["ttr"]
        if olr_scale is None:
            # ERA5 daily ttr accumulation is ~1e7 J/m²; W/m² values are ~100-300
            olr_scale = 86400.0 if np.abs(ttr).mean() > 1e4 else 1.0
        olr = -ttr / olr_scale          # W/m², positive
        olr_i, lat_i, lon_i = crop(olr, lat, lon, -20, 40, 40, 180)

        fig = plt.figure(figsize=(11, 6), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        base_map(ax, 40, 180, -20, 40)
        ax.contourf(lon_i, lat_i, olr_i, levels=OLR_BOUNDS, colors=OLR_COLS,
                    transform=PROJ, extend="both", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=OLR_CMAP, norm=OLR_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.65, aspect=30, ticks=OLR_BOUNDS)
        cb.set_label("OLR (W/m²)  —  Low = Active Convection", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Outgoing Longwave Radiation  ·  MJO/ISO Tracker  ·  (40-180°E)")
        plt.subplots_adjust(top=0.87, bottom=0.17, left=0.04, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir / f"olr_{date_str}.gif", fps)


# ── 18. TCWV daily GIF ────────────────────────────────────────────────────────
def make_tcwv_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    """Total column water vapour — monsoon moisture surge tracking."""
    print(f"\n[18] TCWV daily GIF …")
    frames = []
    for step in range(1, nsteps + 1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["tcwv"])
        if not d or "tcwv" not in d:
            continue
        valid   = init_date + datetime.timedelta(days=step)
        tcwv_i, lat_i, lon_i = crop(d["tcwv"], lat, lon, -20, 40, 40, 180)

        fig = plt.figure(figsize=(11, 6), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        base_map(ax, 40, 180, -20, 40)
        ax.contourf(lon_i, lat_i, tcwv_i, levels=TCWV_BOUNDS, colors=TCWV_COLS,
                    transform=PROJ, extend="both", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=TCWV_CMAP, norm=TCWV_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.65, aspect=30, ticks=TCWV_BOUNDS)
        cb.set_label("Total Column Water Vapour (kg/m²)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Total Column Water Vapour  ·  Moisture Transport & Monsoon Surges")
        plt.subplots_adjust(top=0.87, bottom=0.17, left=0.04, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir / f"tcwv_{date_str}.gif", fps)


# ── 19. Hovmoller diagram (rainfall + OLR vs lon×time) ────────────────────────
def make_hovmoller(raw_dir, date_str, init_date, out_dir, nsteps, soi):
    """Time-longitude (Hovmoller) diagram averaged 5-25°N — MJO propagation."""
    print("\n[19] Hovmoller diagram …")
    tp_rows, olr_rows = [], []
    lons      = None
    olr_scale = None

    for step in range(1, nsteps + 1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["tp", "ttr"])
        if not d:
            continue
        lons = lon
        lm = (lat >= 5) & (lat <= 25)
        if "tp" in d:
            tp_rows.append(d["tp"][lm, :].mean(axis=0))
        if "ttr" in d:
            ttr = d["ttr"]
            if olr_scale is None:
                olr_scale = 86400.0 if np.abs(ttr).mean() > 1e4 else 1.0
            olr_rows.append((-ttr / olr_scale)[lm, :].mean(axis=0))

    if not tp_rows:
        print("  No data — skipping")
        return

    om      = (lons >= 50) & (lons <= 170)
    lon_hov = lons[om]
    tp_arr  = np.array(tp_rows)[:, om]
    days    = np.arange(1, len(tp_rows) + 1)
    wk_tks  = [7, 14, 21, 28, 35, 42]

    has_olr = bool(olr_rows)
    ncols   = 2 if has_olr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 12), facecolor="white")
    if ncols == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.10, wspace=0.12)

    # Rainfall panel
    ax  = axes[0]
    cf  = ax.pcolormesh(lon_hov, days, tp_arr,
                        cmap=PREC_CMAP, norm=PREC_NORM, shading="auto")
    for wd in wk_tks:
        ax.axhline(wd, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Lead Time (days)", fontsize=10, fontweight="bold")
    ax.set_title("Rainfall (mm/day)  [5-25°N mean]",
                 fontsize=11, color="blue", fontweight="bold", pad=6)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(60, 171, 20))
    ax.set_yticks(wk_tks)
    ax.set_yticklabels([f"Day {d}  (Wk {d//7})" for d in wk_tks], fontsize=8)
    ax.grid(color="grey", alpha=0.25, linewidth=0.5)
    cax0 = fig.add_axes([0.08, 0.04, 0.38 if has_olr else 0.85, 0.025])
    fig.colorbar(cf, cax=cax0, orientation="horizontal",
                 ticks=PREC_BOUNDS).ax.tick_params(labelsize=7)

    # OLR panel
    if has_olr:
        olr_arr = np.array(olr_rows)[:, om]
        ax2 = axes[1]
        cf2 = ax2.pcolormesh(lon_hov, days, olr_arr,
                             cmap=OLR_CMAP, norm=OLR_NORM, shading="auto")
        for wd in wk_tks:
            ax2.axhline(wd, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
        ax2.set_title("OLR (W/m²)  [5-25°N]  —  Low = Deep Convection",
                      fontsize=11, color="blue", fontweight="bold", pad=6)
        ax2.invert_yaxis()
        ax2.set_xticks(np.arange(60, 171, 20))
        ax2.set_yticks(wk_tks)
        ax2.set_yticklabels([f"Day {d}  (Wk {d//7})" for d in wk_tks], fontsize=8)
        ax2.grid(color="grey", alpha=0.25, linewidth=0.5)
        cax1 = fig.add_axes([0.55, 0.04, 0.40, 0.025])
        fig.colorbar(cf2, cax=cax1, orientation="horizontal",
                     ticks=OLR_BOUNDS).ax.tick_params(labelsize=7)

    png_header(fig, init_date,
               "Hovmoller Diagram  —  FuXi-S2S 42-day Forecast",
               "Indo-Pacific (50-170°E)  ·  Averaged 5-25°N  ·  Day 1 at top")
    out = out_dir / f"hovmoller_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 20. Z500 anomaly weekly 2×3 ───────────────────────────────────────────────
def make_z500_anom_weekly(raw_dir, date_str, init_date, out_dir, climo, soi):
    """Z500 height anomaly vs ERA5 WB2 climatology, Asia domain, weeks 1-6."""
    print("\n[20] Z500 Anomaly Weekly PNG …")
    if not climo:
        print("  No climatology available — skipping z500_anom")
        return
    WEEKS6 = {1: range(1,8), 2: range(8,15), 3: range(15,22),
              4: range(22,29), 5: range(29,36), 6: range(36,43)}
    POS     = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0, la1, lo0, lo1 = 0, 60, 20, 150

    fig, axes = plt.subplots(2, 3, figsize=(19, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.90,
                        bottom=0.10, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "500 hPa Geopotential Height Anomaly (m)",
               "vs ERA5 1990-2019 climatology  ·  Weeks 1-6  ·  Ridge(+) / Trough(−)")

    for (r, c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["z500"])
        d0 = (init_date + datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date + datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r, c]
        base_map(ax, lo0, lo1, la0, la1)
        if d and "z500" in d:
            z5_i, lat_i, lon_i = crop(d["z500"], lat, lon, la0, la1, lo0, lo1)
            z5_m  = z5_i / 9.80665           # geopotential → height (m)
            c_vals = [climo[s]["z500"]
                      for s in steps if s in climo and "z500" in climo[s]]
            if c_vals:
                z5_c = crop(np.mean(c_vals, axis=0), lat, lon,
                            la0, la1, lo0, lo1)[0]
                anom = z5_m - z5_c
                ax.contourf(lon_i, lat_i, anom,
                            levels=np.linspace(-100, 100, 21),
                            cmap=Z500_ANOM_CMAP, transform=PROJ,
                            extend="both", zorder=1)
                cs = ax.contour(lon_i, lat_i, z5_m,
                               levels=np.arange(5400, 5950, 60),
                               colors="black", linewidths=0.5,
                               transform=PROJ, zorder=3)
                ax.clabel(cs, fmt="%d", fontsize=5, inline=True)
        ax.set_title(f"Week {wk}  ({d0}–{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=Z500_ANOM_CMAP,
                                norm=mcolors.Normalize(-100, 100))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                       ticks=np.arange(-100, 101, 20))
    cb.set_label("Z500 Anomaly (m)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    out = out_dir / f"z500_anom_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 21. 850hPa wind speed + quiver weekly 2×3 ─────────────────────────────────
def make_wind850_weekly(raw_dir, date_str, init_date, out_dir, soi):
    """850hPa wind speed (shaded) + quiver arrows — LLJ evolution, weeks 1-6."""
    print("\n[21] 850hPa Wind (Speed + Direction) Weekly PNG …")
    WEEKS6  = {1: range(1,8), 2: range(8,15), 3: range(15,22),
               4: range(22,29), 5: range(29,36), 6: range(36,43)}
    POS     = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0, la1, lo0, lo1 = 0, 40, 40, 110
    SPD_LEVS = [0, 2, 4, 6, 8, 10, 12, 16, 20]
    SPD_NORM = BoundaryNorm(SPD_LEVS, 256)

    fig, axes = plt.subplots(2, 3, figsize=(19, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.90,
                        bottom=0.10, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "850 hPa Wind Speed & Direction (m/s)",
               "Weekly mean  ·  LLJ = Low Level Jet  ·  Weeks 1-6")

    for (r, c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps, ["u850", "v850"])
        d0 = (init_date + datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date + datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r, c]
        base_map(ax, lo0, lo1, la0, la1)
        if d and "u850" in d and "v850" in d:
            u, lat_i, lon_i = crop(d["u850"], lat, lon, la0, la1, lo0, lo1)
            v   = crop(d["v850"], lat, lon, la0, la1, lo0, lo1)[0]
            spd = np.sqrt(u**2 + v**2)
            ax.contourf(lon_i, lat_i, spd, levels=SPD_LEVS,
                        cmap="YlOrRd", norm=SPD_NORM,
                        transform=PROJ, extend="max", zorder=1)
            n   = max(1, len(lon_i) // 16)
            ax.quiver(lon_i[::n], lat_i[::n], u[::n, ::n], v[::n, ::n],
                      transform=PROJ, scale=5, scale_units="xy",
                      width=0.003, headwidth=4, headlength=5,
                      color="black", zorder=4, alpha=0.85)
        ax.set_title(f"Week {wk}  ({d0}–{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap="YlOrRd", norm=SPD_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=SPD_LEVS)
    cb.set_label("Wind Speed (m/s)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    out = out_dir / f"wind850_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ── 22. SST daily GIF ─────────────────────────────────────────────────────────
def make_sst_gif(raw_dir, date_str, init_date, out_dir, nsteps, fps, soi):
    """SST evolution — Indian Ocean, Bay of Bengal, Arabian Sea."""
    print(f"\n[22] SST daily GIF …")
    frames = []
    for step in range(1, nsteps + 1):
        d, lat, lon = load_step(raw_dir, date_str, step, ["sst"])
        if not d or "sst" not in d:
            continue
        valid = init_date + datetime.timedelta(days=step)
        sst_c = d["sst"] - 273.15          # K → °C
        sst_m = np.ma.masked_where(np.isnan(sst_c) | (sst_c < 15.0), sst_c)
        sst_i, lat_i, lon_i = crop(sst_m.data, lat, lon, -30, 35, 30, 120)
        mask_i = crop(sst_m.mask.astype(float) if not np.ndim(sst_m.mask) == 0
                      else np.zeros_like(sst_c), lat, lon, -30, 35, 30, 120)[0]
        sst_plot = np.ma.array(sst_i, mask=mask_i > 0.5)

        fig = plt.figure(figsize=(10, 7), facecolor="white")
        ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
        base_map(ax, 30, 120, -30, 35)
        ax.contourf(lon_i, lat_i, sst_plot, levels=SST_BOUNDS, colors=SST_COLS,
                    transform=PROJ, extend="both", zorder=1)
        sm = plt.cm.ScalarMappable(cmap=SST_CMAP, norm=SST_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.65, aspect=30, ticks=SST_BOUNDS)
        cb.set_label("SST (°C)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        frame_header(fig, init_date, valid,
                     "Sea Surface Temperature  ·  Indian Ocean / Bay of Bengal / Arabian Sea")
        plt.subplots_adjust(top=0.87, bottom=0.17, left=0.04, right=0.97)
        frames.append(render(fig))
        if step % 7 == 0 or step == 1:
            print(f"  frame {step:02d}/{nsteps}")
    save_gif(frames, out_dir / f"sst_{date_str}.gif", fps)


# ── 23. Wind shear (200-850 hPa) weekly 2×3 ──────────────────────────────────
def make_shear_weekly(raw_dir, date_str, init_date, out_dir, soi):
    """Vertical wind shear |V200−V850| — TC genesis environment, weeks 1-6."""
    print("\n[23] Wind Shear (200-850 hPa) Weekly PNG …")
    WEEKS6 = {1: range(1,8), 2: range(8,15), 3: range(15,22),
              4: range(22,29), 5: range(29,36), 6: range(36,43)}
    POS    = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    la0, la1, lo0, lo1 = -5, 35, 40, 120

    fig, axes = plt.subplots(2, 3, figsize=(19, 12),
                             subplot_kw=dict(projection=PROJ),
                             facecolor="white")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.90,
                        bottom=0.10, hspace=0.18, wspace=0.10)
    png_header(fig, init_date, "200-850 hPa Vertical Wind Shear (m/s)",
               "Low shear (<10 m/s) = TC-favorable  ·  dashed = 10 m/s  ·  Weeks 1-6")

    for (r, c), (wk, steps) in zip(POS, WEEKS6.items()):
        d, lat, lon = weekly_mean(raw_dir, date_str, steps,
                                  ["u200", "v200", "u850", "v850"])
        d0 = (init_date + datetime.timedelta(days=list(steps)[0])).strftime("%-d%b")
        d1 = (init_date + datetime.timedelta(days=list(steps)[-1])).strftime("%-d%b")
        ax = axes[r, c]
        base_map(ax, lo0, lo1, la0, la1)
        if d and all(k in d for k in ["u200", "v200", "u850", "v850"]):
            du, lat_i, lon_i = crop(d["u200"] - d["u850"],
                                    lat, lon, la0, la1, lo0, lo1)
            dv    = crop(d["v200"] - d["v850"], lat, lon, la0, la1, lo0, lo1)[0]
            shear = np.sqrt(du**2 + dv**2)
            ax.contourf(lon_i, lat_i, shear, levels=SHEAR_BOUNDS, colors=SHEAR_COLS,
                        transform=PROJ, extend="max", zorder=1)
            ax.contour(lon_i, lat_i, shear, levels=[10.0],
                       colors=["#333333"], linewidths=1.2,
                       linestyles=["--"], transform=PROJ, zorder=3)
        ax.set_title(f"Week {wk}  ({d0}–{d1})", fontsize=9,
                     color="blue", fontweight="bold", pad=4)

    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=SHEAR_CMAP, norm=SHEAR_NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=SHEAR_BOUNDS)
    cb.set_label("Wind Shear (m/s)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    out = out_dir / f"shear_weekly_{date_str}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
MODES = [
    "prec","vort","divg","rh","igpp",
    "temp_gif","temp_anom","hw_gif",
    "rf_weekly","rf_w6","rf_anom","igpp_weekly",
    "tmax_actual","tmax_anom","temp_weekly","temp_anom_weekly","hw",
    # Extra diagnostics
    "olr","tcwv","hovmoller","z500_anom","wind850","sst","shear",
]

def main():
    p = argparse.ArgumentParser(
        description="ERPAS-style plots from FuXi-S2S (all 16 plot types)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date",    required=True, help="Init date YYYYMMDD")
    p.add_argument("--raw_dir", default=str(RAW_DIR))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--soi_shp", default=SOI_SHP)
    p.add_argument("--mode",    default=None, choices=MODES)
    p.add_argument("--steps",   type=int, default=42)
    p.add_argument("--fps",     type=int, default=FPS)
    p.add_argument("--members", type=int, nargs="+", default=None,
                   help="Member indices (e.g. 0 1 2); default=all")
    args = p.parse_args()

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

    d1, lat, lon = load_step(args.raw_dir, args.date, 1,
                             ["tp","u850","v850","msl","u200","v200",
                              "z500","t2m","q700","t700","u700","v700"])
    if lat is None:
        print(f"ERROR: no data at {args.raw_dir}/{args.date}/member/")
        sys.exit(1)
    print(f"  Channels available: {list(d1.keys())}")
    has_200  = "u200" in d1
    has_rh   = "q700" in d1 and "t700" in d1

    modes = [args.mode] if args.mode else MODES
    need_climo = any(m in modes for m in
                     ["rf_anom","tmax_anom","temp_anom","temp_anom_weekly","z500_anom"])
    climo = build_climo(init_date, args.steps, lat, lon) if need_climo else None

    for mode in modes:
        if   mode == "prec":
            make_prec_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "vort":
            make_vort_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "divg":
            if not has_200: print("  SKIP divg — u200/v200 not found")
            else: make_divg_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "rh":
            if not has_rh: print("  SKIP rh — q700/t700 not found")
            else: make_rh_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "igpp":
            make_igpp_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "temp_gif":
            make_temp_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "temp_anom":
            make_temp_anom_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, climo, soi)
        elif mode == "hw_gif":
            make_hw_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "rf_weekly":
            make_rf_weekly(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "rf_w6":
            make_rf_weekly6(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "rf_anom":
            make_rf_anom_weekly(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "igpp_weekly":
            make_igpp_weekly(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "tmax_actual":
            make_tmax_actual(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "tmax_anom":
            make_tmax_anom(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "temp_weekly":
            make_temp_weekly(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "temp_anom_weekly":
            make_temp_anom_weekly(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "hw":
            make_hw_weekly(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "olr":
            make_olr_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "tcwv":
            make_tcwv_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "hovmoller":
            make_hovmoller(args.raw_dir, args.date, init_date, out_dir, args.steps, soi)
        elif mode == "z500_anom":
            make_z500_anom_weekly(args.raw_dir, args.date, init_date, out_dir, climo, soi)
        elif mode == "wind850":
            make_wind850_weekly(args.raw_dir, args.date, init_date, out_dir, soi)
        elif mode == "sst":
            make_sst_gif(args.raw_dir, args.date, init_date, out_dir, args.steps, args.fps, soi)
        elif mode == "shear":
            make_shear_weekly(args.raw_dir, args.date, init_date, out_dir, soi)

    print(f"\nAll done!  →  {out_dir}")


if __name__ == "__main__":
    main()
