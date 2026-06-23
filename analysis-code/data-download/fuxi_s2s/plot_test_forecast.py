#!/usr/bin/env python3
"""
plot_test_forecast.py
=====================
Visualise FuXi-S2S test forecast (2026-06-18 init).

Produces two figures per lead-time week:
  1. Global   — Z500 + wind (850 hPa), T2m anomaly
  2. India    — TP accumulation, T2m anomaly, Z500

Usage
-----
  python plot_test_forecast.py                  # all weeks
  python plot_test_forecast.py --week 1         # week 1 only (day 1-7)
  python plot_test_forecast.py --member 0       # specific member (default: ensemble mean)
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

# ── PATHS ─────────────────────────────────────────────────────────────────────
RAW_DIR  = Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw/20260618")
PLOT_DIR = Path("/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/test_plots")
INIT_DATE = "2026-06-18"

# India domain
INDIA = dict(lon_min=65, lon_max=100, lat_min=5, lat_max=38)

# Climatology reference: use the bundled sample input as a rough proxy
# (for anomalies we use the ensemble mean at each lead as the "anomaly from
#  persistence" — i.e. departure from init state — since we have no climo here)

WEEKS = {1: range(1, 8), 2: range(8, 15), 3: range(15, 22), 4: range(22, 29),
         5: range(29, 36), 6: range(36, 43)}


def load_members(raw_dir: Path, channel: str, steps: range) -> xr.DataArray:
    """Load a single channel across all members and steps → (member, step, lat, lon)."""
    members = sorted(raw_dir.glob("member/*"))
    all_mem = []
    for mem_dir in members:
        steps_list = []
        for step in steps:
            f = mem_dir / f"{step:02d}.nc"
            if not f.exists():
                continue
            da = xr.open_dataarray(str(f))
            steps_list.append(da.sel(channel=channel).squeeze())
        if steps_list:
            all_mem.append(xr.concat(steps_list, dim="step").assign_coords(step=list(steps)))
    if not all_mem:
        return None
    return xr.concat(all_mem, dim="member").assign_coords(member=np.arange(len(all_mem)))


def add_india_box(ax):
    ax.set_extent([INDIA['lon_min'], INDIA['lon_max'],
                   INDIA['lat_min'], INDIA['lat_max']], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="grey")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)


def add_global(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="grey", alpha=0.4)


def plot_week(week: int, raw_dir: Path, plot_dir: Path):
    steps = list(WEEKS[week])
    print(f"\n── Week {week}  (day {steps[0]}–{steps[-1]}) ──")

    # ── Load channels ─────────────────────────────────────────────────────────
    tp   = load_members(raw_dir, "tp",   steps)   # mm (already ×1000 in model)
    t2m  = load_members(raw_dir, "t2m",  steps)   # K
    z500 = load_members(raw_dir, "z500", steps)   # m²/s²
    u850 = load_members(raw_dir, "u850", steps)
    v850 = load_members(raw_dir, "v850", steps)

    if tp is None:
        print(f"  No data found for week {week}, skipping.")
        return

    # Ensemble mean, time-mean over the week
    tp_mean   = tp.mean(["member", "step"])        # (lat, lon)
    t2m_mean  = t2m.mean(["member", "step"]) - 273.15
    z500_mean = z500.mean(["member", "step"]) / 9.80665   # → gpm
    u850_mean = u850.mean(["member", "step"])
    v850_mean = v850.mean(["member", "step"])

    # T2m anomaly from week-1 (persistence proxy)
    if week > 1:
        t2m_w1   = load_members(raw_dir, "t2m", list(WEEKS[1]))
        t2m_anom = t2m_mean - (t2m_w1.mean(["member", "step"]) - 273.15)
        anom_label = "T2m anomaly from week-1 mean (°C)"
    else:
        t2m_anom  = t2m_mean - t2m_mean.mean()
        anom_label = "T2m departure from spatial mean (°C)"

    lat = tp_mean.lat.values
    lon = tp_mean.lon.values

    proj = ccrs.PlateCarree()

    # ══ FIGURE 1: India ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw=dict(projection=proj))
    fig.suptitle(f"FuXi-S2S  |  Init: {INIT_DATE}  |  Week {week} "
                 f"(day {steps[0]}–{steps[-1]}, ensemble mean)",
                 fontsize=13, fontweight="bold")

    # Panel 1: TP accumulation
    ax = axes[0]
    add_india_box(ax)
    tp_india = tp_mean.sel(lat=slice(38, 5), lon=slice(65, 100))
    cf = ax.contourf(tp_india.lon, tp_india.lat, tp_india.values,
                     levels=[0, 1, 2, 5, 10, 20, 30, 50, 75, 100],
                     cmap="YlGnBu", transform=proj, extend="max")
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, label="TP (mm/day mean)")
    ax.set_title("Total Precipitation", fontsize=11)

    # Panel 2: T2m anomaly
    ax = axes[1]
    add_india_box(ax)
    t2m_india = t2m_anom.sel(lat=slice(38, 5), lon=slice(65, 100))
    lim = max(2, float(np.abs(t2m_india.values).max()))
    cf = ax.contourf(t2m_india.lon, t2m_india.lat, t2m_india.values,
                     levels=np.linspace(-lim, lim, 21),
                     cmap="RdBu_r", transform=proj, extend="both")
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, label=anom_label)
    ax.set_title("T2m Anomaly", fontsize=11)

    # Panel 3: Z500 + 850 hPa wind
    ax = axes[2]
    add_india_box(ax)
    z_india = z500_mean.sel(lat=slice(38, 5), lon=slice(65, 100))
    u_india = u850_mean.sel(lat=slice(38, 5), lon=slice(65, 100))
    v_india = v850_mean.sel(lat=slice(38, 5), lon=slice(65, 100))
    cf = ax.contourf(z_india.lon, z_india.lat, z_india.values,
                     levels=15, cmap="RdYlBu_r", transform=proj)
    cs = ax.contour(z_india.lon, z_india.lat, z_india.values,
                    levels=10, colors="k", linewidths=0.5, transform=proj)
    ax.clabel(cs, fmt="%d", fontsize=7)
    # Wind vectors — subsample
    skip = 2
    ax.quiver(u_india.lon.values[::skip], u_india.lat.values[::skip],
              u_india.values[::skip, ::skip], v_india.values[::skip, ::skip],
              transform=proj, scale=150, width=0.004, color="k")
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
    fig.suptitle(f"FuXi-S2S  |  Init: {INIT_DATE}  |  Week {week} — Global",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    add_global(ax)
    cf = ax.contourf(lon, lat, tp_mean.values,
                     levels=[0, 1, 2, 5, 10, 20, 30, 50, 75, 100],
                     cmap="YlGnBu", transform=proj, extend="max")
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04,
                 label="TP (mm/day mean)", shrink=0.8)
    ax.set_title("Global Precipitation", fontsize=11)

    ax = axes[1]
    add_global(ax)
    lim = max(3, float(np.abs(t2m_anom.values).max()))
    cf = ax.contourf(lon, lat, t2m_anom.values,
                     levels=np.linspace(-lim, lim, 21),
                     cmap="RdBu_r", transform=proj, extend="both")
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04,
                 label=anom_label, shrink=0.8)
    ax.set_title("Global T2m Anomaly", fontsize=11)

    plt.tight_layout()
    out = plot_dir / f"global_week{week:02d}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None,
                        help="Week number 1-6 (default: all)")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: output dir not found: {RAW_DIR}")
        print("Run inference first: sbatch slurm/fuxi_test_inference.sbatch")
        sys.exit(1)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading from : {RAW_DIR}")
    print(f"Saving plots : {PLOT_DIR}")

    weeks = [args.week] if args.week else list(WEEKS.keys())
    for w in weeks:
        plot_week(w, RAW_DIR, PLOT_DIR)

    print(f"\nDone. Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
