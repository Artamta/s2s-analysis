#!/usr/bin/env python3
"""
plot_test_forecast.py
=====================
Visualise FuXi-S2S test forecast (2026-06-18 init).

Two figures per week:
  1. India  — TP (mm/day) | T2m (°C) | Z500 + 850hPa wind
  2. Global — TP (mm/day) | T2m (°C)

Uses Survey of India STATE_BOUNDARY.shp for India state borders.

Usage
-----
  python plot_test_forecast.py          # all 6 weeks
  python plot_test_forecast.py --week 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

RAW_DIR   = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw/20260618")
PLOT_DIR  = Path("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/test_plots")
INIT_DATE = "2026-06-18"

# Survey of India official state boundaries
SOI_SHP = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"

INDIA = dict(lon_min=65, lon_max=100, lat_min=5, lat_max=38)

WEEKS = {1: range(1, 8), 2: range(8, 15), 3: range(15, 22),
         4: range(22, 29), 5: range(29, 36), 6: range(36, 43)}

# Weekly-mean tp at 1.5° is small due to spatial/temporal averaging
TP_LEVELS = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]   # mm/day


def load_channel(raw_dir, channel, steps):
    members = sorted(raw_dir.glob("member/*"))
    all_mem = []
    for mem_dir in members:
        days = []
        for step in steps:
            f = mem_dir / f"{step:02d}.nc"
            if not f.exists():
                continue
            da = xr.open_dataarray(str(f))
            days.append(da.sel(channel=channel).squeeze(drop=True))
        if days:
            all_mem.append(xr.concat(days, dim="step").assign_coords(step=list(steps)))
    if not all_mem:
        return None
    return xr.concat(all_mem, dim="member").assign_coords(member=np.arange(len(all_mem)))


def emean(da):
    return da.mean(["member", "step"])


def india_slice(da):
    return da.sel(lat=slice(INDIA['lat_max'], INDIA['lat_min']),
                  lon=slice(INDIA['lon_min'], INDIA['lon_max']))


def add_india_map(ax, soi_geoms=None, soi_crs=None):
    """Set up India domain with SOI shapefile state borders."""
    proj = ccrs.PlateCarree()
    ax.set_extent([INDIA['lon_min'], INDIA['lon_max'],
                   INDIA['lat_min'], INDIA['lat_max']], crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), color="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),  color="white",     zorder=0)

    if soi_geoms is not None:
        ax.add_geometries(
            soi_geoms, crs=soi_crs,
            facecolor="none", edgecolor="black", linewidth=0.5, zorder=3
        )
    else:
        ax.add_feature(cfeature.BORDERS,   linewidth=0.6, linestyle="--")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="grey")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)


def add_global_map(ax):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), color="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),  color="white",     zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="grey", alpha=0.4)


SOI_CRS = ccrs.LambertConformal(
    central_longitude=80.0,
    central_latitude=24.0,
    standard_parallels=(12.472944, 35.172806),
    false_easting=4000000.0,
    false_northing=4000000.0,
)


def load_states_shp(path):
    """Load and simplify SOI state geometries; returns (geoms, src_crs) tuple.
    Simplifies to ~5 km tolerance in LCC metres so cartopy reproject is fast."""
    try:
        reader = shpreader.Reader(path)
        geoms  = [rec.geometry.simplify(5000) for rec in reader.records()]
        print(f"  Loaded SOI shapefile: {len(geoms)} state polygons (simplified)")
        return geoms, SOI_CRS
    except Exception as e:
        print(f"  WARNING: could not load shapefile ({e}) — using cartopy borders")
        return None, None


def plot_week(week, raw_dir, plot_dir, soi_geoms, soi_crs):
    steps = list(WEEKS[week])
    print(f"\n── Week {week}  (day {steps[0]}–{steps[-1]}) ──")

    tp   = load_channel(raw_dir, "tp",   steps)
    t2m  = load_channel(raw_dir, "t2m",  steps)
    z500 = load_channel(raw_dir, "z500", steps)
    u850 = load_channel(raw_dir, "u850", steps)
    v850 = load_channel(raw_dir, "v850", steps)

    if tp is None:
        print(f"  No data for week {week}, skipping.")
        return

    proj = ccrs.PlateCarree()

    tp_mean   = emean(tp)                   # mm/day
    t2m_C     = emean(t2m) - 273.15        # °C
    z500_gpm  = emean(z500) / 9.80665      # gpm
    u850_mean = emean(u850)
    v850_mean = emean(v850)

    lat = tp_mean.lat.values
    lon = tp_mean.lon.values

    tp_i   = india_slice(tp_mean)
    t2m_i  = india_slice(t2m_C)
    z_i    = india_slice(z500_gpm)
    u_i    = india_slice(u850_mean)
    v_i    = india_slice(v850_mean)

    print(f"  tp India:  max={float(tp_i.max()):.3f} mm/day")
    print(f"  t2m India: {float(t2m_i.min()):.1f} – {float(t2m_i.max()):.1f} °C")

    # ══ FIGURE 1: India ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw=dict(projection=proj))
    fig.suptitle(
        f"FuXi-S2S  |  Init: {INIT_DATE}  |  Week {week} "
        f"(day {steps[0]}–{steps[-1]}, ensemble mean)",
        fontsize=12, fontweight="bold")

    # Panel 1 — TP
    ax = axes[0]
    add_india_map(ax, soi_geoms, soi_crs)
    cf = ax.contourf(tp_i.lon, tp_i.lat, tp_i.values,
                     levels=TP_LEVELS, cmap="YlGnBu",
                     transform=proj, extend="max", zorder=1)
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05,
                 label="TP (mm/day, weekly mean)")
    ax.set_title("Total Precipitation", fontsize=11)

    # Panel 2 — T2m (°C)
    ax = axes[1]
    add_india_map(ax, soi_geoms, soi_crs)
    t2m_levels = np.arange(np.floor(float(t2m_i.min())/2)*2,
                            np.ceil(float(t2m_i.max())/2)*2 + 1, 2)
    cf = ax.contourf(t2m_i.lon, t2m_i.lat, t2m_i.values,
                     levels=t2m_levels, cmap="RdYlBu_r",
                     transform=proj, extend="both", zorder=1)
    cs = ax.contour(t2m_i.lon, t2m_i.lat, t2m_i.values,
                    levels=t2m_levels, colors="k", linewidths=0.3,
                    transform=proj, zorder=2)
    ax.clabel(cs, fmt="%d", fontsize=7)
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, label="T2m (°C)")
    ax.set_title("2m Temperature", fontsize=11)

    # Panel 3 — Z500 + 850 hPa wind
    ax = axes[2]
    add_india_map(ax, soi_geoms, soi_crs)
    cf = ax.contourf(z_i.lon, z_i.lat, z_i.values, levels=15,
                     cmap="RdYlBu_r", transform=proj, zorder=1)
    cs = ax.contour(z_i.lon, z_i.lat, z_i.values, levels=10,
                    colors="k", linewidths=0.4, transform=proj, zorder=2)
    ax.clabel(cs, fmt="%d", fontsize=7)
    skip = 2
    ax.quiver(u_i.lon.values[::skip], u_i.lat.values[::skip],
              u_i.values[::skip, ::skip], v_i.values[::skip, ::skip],
              transform=proj, scale=200, width=0.003, color="k", zorder=5)
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, label="Z500 (gpm)")
    ax.set_title("Z500 + 850hPa Wind", fontsize=11)

    plt.tight_layout()
    out = plot_dir / f"india_week{week:02d}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ══ FIGURE 2: Global ══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(20, 5),
                             subplot_kw=dict(projection=ccrs.Robinson()))
    fig.suptitle(
        f"FuXi-S2S  |  Init: {INIT_DATE}  |  Week {week} — Global",
        fontsize=12, fontweight="bold")

    ax = axes[0]
    add_global_map(ax)
    cf = ax.contourf(lon, lat, tp_mean.values,
                     levels=TP_LEVELS, cmap="YlGnBu",
                     transform=proj, extend="max", zorder=1)
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04,
                 label="TP (mm/day, weekly mean)", shrink=0.85)
    ax.set_title("Global Precipitation", fontsize=11)

    ax = axes[1]
    add_global_map(ax)
    t2m_g_levels = np.arange(-30, 46, 5)
    cf = ax.contourf(lon, lat, t2m_C.values,
                     levels=t2m_g_levels, cmap="RdYlBu_r",
                     transform=proj, extend="both", zorder=1)
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04,
                 label="T2m (°C)", shrink=0.85)
    ax.set_title("Global 2m Temperature", fontsize=11)

    plt.tight_layout()
    out = plot_dir / f"global_week{week:02d}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found — run inference first")
        sys.exit(1)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading : {RAW_DIR}")
    print(f"Saving  : {PLOT_DIR}")

    soi_geoms, soi_crs = load_states_shp(SOI_SHP)
    weeks = [args.week] if args.week else list(WEEKS.keys())

    for w in weeks:
        plot_week(w, RAW_DIR, PLOT_DIR, soi_geoms, soi_crs)

    print(f"\nDone — plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
