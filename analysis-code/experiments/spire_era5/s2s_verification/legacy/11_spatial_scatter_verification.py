"""
11_spatial_scatter_verification.py

Model (Spire) vs Ground Truth (ERA5) verification figures:

  Fig 1: Spatial maps — Spire (model) and ERA5 (obs) side by side per week
          for T2m-mean and T2m-max anomalies
  Fig 2: Scatter plots — each point is a grid cell, Spire vs ERA5
          R², MAE, RMSE annotated on each panel; one panel per week

Variables: t2m_mean and t2m_max anomalies, W1–W6
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import string

# ── paths ──────────────────────────────────────────────────────────────────────
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

print("Loading weekly_anomalies_v2.nc …")
ds = xr.open_dataset("weekly_anomalies_v2.nc")
lats = ds["latitude"].values
lons = ds["longitude"].values
weeks = [1, 2, 3, 4, 5, 6]
wlabels = ["W1 (d1–7)", "W2 (d8–14)", "W3 (d15–21)",
           "W4 (d22–28)", "W5 (d29–35)", "W6 (d36–42)"]
it = pd.DatetimeIndex(ds["init_time"].values)
init_range = f"{it[0]:%Y-%m-%d} to {it[-1]:%Y-%m-%d}"

proj = ccrs.PlateCarree()
extent = [55, 105, 0, 50]
letters = list(string.ascii_lowercase)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute metrics between two flat arrays
# ══════════════════════════════════════════════════════════════════════════════
def metrics(model_flat, obs_flat):
    mask = np.isfinite(model_flat) & np.isfinite(obs_flat)
    m, o = model_flat[mask], obs_flat[mask]
    if len(m) < 10:
        return np.nan, np.nan, np.nan
    r, _ = pearsonr(m, o)
    r2 = r ** 2
    mae = np.mean(np.abs(m - o))
    rmse = np.sqrt(np.mean((m - o) ** 2))
    return r2, mae, rmse


# ══════════════════════════════════════════════════════════════════════════════
# Helper: single map panel
# ══════════════════════════════════════════════════════════════════════════════
def base_map(ax, data, cmap, norm, title="", letter=None,
             left_label=False, bottom_label=False):
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                       transform=proj, rasterized=True, shading="auto")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.7, edgecolor="black", zorder=4)
    ax.set_extent(extent, crs=proj)
    gl = ax.gridlines(crs=proj, linewidth=0.3, color="gray",
                      linestyle=":", alpha=0.6, zorder=3)
    gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_label
    gl.bottom_labels = bottom_label
    gl.xlabel_style = {"size": 7, "color": "0.3"}
    gl.ylabel_style = {"size": 7, "color": "0.3"}
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
    if letter:
        ax.text(0.025, 0.96, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))
    return im


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 – Spatial: Spire (model) vs ERA5 (obs), 2×6 per variable
# ══════════════════════════════════════════════════════════════════════════════
def fig_spatial_model_vs_obs(spire_var, era5_var, vmax, unit, cmap, fname,
                              suptitle, row1_label, row2_label, ticks=None):
    sp = ds[spire_var].mean("init_time").values   # (week, lat, lon)
    e5 = ds[era5_var].mean("init_time").values
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.8),
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.055, right=0.905, bottom=0.06, top=0.87,
                        hspace=0.08, wspace=0.05)

    for i, wk in enumerate(weeks):
        wi = wk - 1
        spw, e5w = sp[wi], e5[wi]

        r2, mae, rmse = metrics(spw.ravel(), e5w.ravel())
        corner_txt = f"R²={r2:.2f}\nMAE={mae:.2f}"

        base_map(axes[0, i], spw, cmap, norm, title=wlabels[i],
                 letter=f"({letters[i]})", left_label=(i == 0))
        im = base_map(axes[1, i], e5w, cmap, norm, title="",
                      letter=f"({letters[i+6]})",
                      left_label=(i == 0), bottom_label=True)
        axes[1, i].text(0.97, 0.04, corner_txt, transform=axes[1, i].transAxes,
                        fontsize=7.5, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))

    fig.text(0.022, 0.66, row1_label, ha="center", va="center",
             fontsize=9.5, fontweight="bold", rotation=90)
    fig.text(0.022, 0.27, row2_label, ha="center", va="center",
             fontsize=9.5, fontweight="bold", rotation=90)

    cax = fig.add_axes([0.918, 0.06, 0.013, 0.81])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both",
                      ticks=ticks)
    cb.set_label(unit, fontsize=9, labelpad=5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.955)
    fig.savefig(f"{OUT}/{fname}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")


print("Fig 1a: spatial model-vs-obs  T2m-mean …")
fig_spatial_model_vs_obs(
    "spire_t2m_mean_anom", "era5_t2m_mean_anom",
    vmax=6.0, unit="Mean T2m anomaly (K)", cmap=plt.cm.RdBu_r,
    fname="fig1a_spatial_mean_model_vs_obs.png",
    suptitle=(f"Spire (model) vs ERA5 (ground truth)  |  Mean-T2m Anomaly  |"
              f"  90-init mean  |  {init_range}"),
    row1_label="Spire forecast\n(model)",
    row2_label="ERA5 observed\n(ground truth)",
    ticks=[-6, -4, -2, 0, 2, 4, 6])

print("Fig 1b: spatial model-vs-obs  T2m-max …")
fig_spatial_model_vs_obs(
    "spire_t2m_max_anom", "era5_t2m_max_anom",
    vmax=6.0, unit="Max T2m anomaly (K)", cmap=plt.cm.RdBu_r,
    fname="fig1b_spatial_max_model_vs_obs.png",
    suptitle=(f"Spire (model) vs ERA5 (ground truth)  |  Max-T2m Anomaly  |"
              f"  90-init mean  |  {init_range}"),
    row1_label="Spire forecast\n(model)",
    row2_label="ERA5 observed\n(ground truth)",
    ticks=[-6, -4, -2, 0, 2, 4, 6])


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Scatter plots: Spire vs ERA5, one panel per week
#         Each point = one grid cell (init-time averaged)
#         Annotate: R², MAE, RMSE
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter_model_vs_obs(spire_var, era5_var, xlim, unit, fname,
                              suptitle, color):
    sp = ds[spire_var].mean("init_time").values   # (week, lat, lon)
    e5 = ds[era5_var].mean("init_time").values

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.90,
                        hspace=0.38, wspace=0.28)

    for i, wk in enumerate(weeks):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        wi = wk - 1
        spw = sp[wi].ravel()
        e5w = e5[wi].ravel()

        mask = np.isfinite(spw) & np.isfinite(e5w)
        spw_m, e5w_m = spw[mask], e5w[mask]

        r2, mae, rmse = metrics(spw_m, e5w_m)

        # scatter (downsample for speed if huge)
        if len(spw_m) > 5000:
            idx = np.random.choice(len(spw_m), 5000, replace=False)
            spw_plot, e5w_plot = spw_m[idx], e5w_m[idx]
        else:
            spw_plot, e5w_plot = spw_m, e5w_m

        ax.scatter(e5w_plot, spw_plot, s=10, alpha=0.45, color=color,
                   linewidths=0, rasterized=True)

        # 1:1 line
        lo, hi = xlim
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="1:1")

        # regression line
        if len(spw_m) >= 2:
            m_coef = np.polyfit(e5w_m, spw_m, 1)
            xfit = np.linspace(lo, hi, 100)
            ax.plot(xfit, np.polyval(m_coef, xfit), "-", color="firebrick",
                    lw=1.4, alpha=0.85, label=f"fit (slope={m_coef[0]:.2f})")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ERA5 (ground truth)  {unit}", fontsize=9)
        ax.set_ylabel(f"Spire (model)  {unit}", fontsize=9)
        ax.set_title(f"({letters[i]}) {wlabels[i]}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", "box")

        stats_txt = f"R²  = {r2:.3f}\nMAE = {mae:.3f} K\nRMSE= {rmse:.3f} K"
        ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow",
                          ec="0.6", alpha=0.95))
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.97)
    fig.savefig(f"{OUT}/{fname}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")


print("Fig 2a: scatter  T2m-mean …")
fig_scatter_model_vs_obs(
    "spire_t2m_mean_anom", "era5_t2m_mean_anom",
    xlim=(-8, 8), unit="(K)",
    fname="fig2a_scatter_mean_model_vs_obs.png",
    suptitle=(f"Spire vs ERA5  |  Mean-T2m Anomaly — scatter (grid-cell, 90-init mean)"
              f"  |  {init_range}"),
    color="#2166AC")

print("Fig 2b: scatter  T2m-max …")
fig_scatter_model_vs_obs(
    "spire_t2m_max_anom", "era5_t2m_max_anom",
    xlim=(-8, 8), unit="(K)",
    fname="fig2b_scatter_max_model_vs_obs.png",
    suptitle=(f"Spire vs ERA5  |  Max-T2m Anomaly — scatter (grid-cell, 90-init mean)"
              f"  |  {init_range}"),
    color="#D73027")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 – Scatter plots: per init-time, India-mean anomaly
#         X = ERA5 India-mean, Y = Spire India-mean, one point per init date
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter_india_mean(spire_var, era5_var, xlim, unit, fname, suptitle, color):
    sp = ds[spire_var].values   # (init, week, lat, lon)
    e5 = ds[era5_var].values

    sp_india = np.nanmean(sp, axis=(-1, -2))   # (init, week)
    e5_india = np.nanmean(e5, axis=(-1, -2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.90,
                        hspace=0.38, wspace=0.28)

    for i, wk in enumerate(weeks):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        wi = wk - 1
        spw = sp_india[:, wi]
        e5w = e5_india[:, wi]

        r2, mae, rmse = metrics(spw, e5w)

        ax.scatter(e5w, spw, s=30, alpha=0.7, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3)

        lo, hi = xlim
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="1:1")

        if len(spw) >= 2:
            m_coef = np.polyfit(e5w, spw, 1)
            xfit = np.linspace(lo, hi, 100)
            ax.plot(xfit, np.polyval(m_coef, xfit), "-", color="firebrick",
                    lw=1.4, alpha=0.85, label=f"fit (slope={m_coef[0]:.2f})")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ERA5 India-mean  {unit}", fontsize=9)
        ax.set_ylabel(f"Spire India-mean  {unit}", fontsize=9)
        ax.set_title(f"({letters[i]}) {wlabels[i]}  (n={len(spw)} inits)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", "box")

        stats_txt = f"R²  = {r2:.3f}\nMAE = {mae:.3f} K\nRMSE= {rmse:.3f} K"
        ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow",
                          ec="0.6", alpha=0.95))
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.97)
    fig.savefig(f"{OUT}/{fname}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")


print("Fig 3a: scatter India-mean  T2m-mean …")
fig_scatter_india_mean(
    "spire_t2m_mean_anom", "era5_t2m_mean_anom",
    xlim=(-6, 6), unit="(K)",
    fname="fig3a_scatter_india_mean_model_vs_obs.png",
    suptitle=(f"Spire vs ERA5  |  India-mean Mean-T2m Anomaly"
              f"  (each point = one init date)  |  {init_range}"),
    color="#2166AC")

print("Fig 3b: scatter India-mean  T2m-max …")
fig_scatter_india_mean(
    "spire_t2m_max_anom", "era5_t2m_max_anom",
    xlim=(-6, 6), unit="(K)",
    fname="fig3b_scatter_india_max_model_vs_obs.png",
    suptitle=(f"Spire vs ERA5  |  India-mean Max-T2m Anomaly"
              f"  (each point = one init date)  |  {init_range}"),
    color="#D73027")

print("\nAll figures saved to", OUT)
