"""
10_publication_figures_v2.py

Publication-quality figures from the CORRECTED, consistent-baseline anomalies
(weekly_anomalies_v2.nc). Produces both mean-vs-mean and max-vs-max variants.

Design fixes over the previous version:
  - Spire anomaly is now real (was identically zero due to a climo bug).
  - Spire and ERA5 share the SAME climatology in each variant → fair magnitudes.
  - No empty/blank panels: each figure's grid exactly matches its panel count.
  - Diverging colormaps centred on zero with symmetric, generous ranges.
  - Coastlines only (no disputed borders), India domain.

Figures:
  fig_A_t2m_mean_anomaly.png  — Spire vs ERA5, mean-T2m, 2×6 (W1–W6)
  fig_B_t2m_max_anomaly.png   — Spire vs ERA5, max-T2m,  2×6 (W1–W6)
  fig_C_bias_mean.png         — (Spire−ERA5) mean-T2m bias, 2×3
  fig_D_bias_max.png          — (Spire−ERA5) max-T2m bias,  2×3
  fig_E_skill_summary.png     — India-mean ACC / RMSE / anomaly-vs-week lines
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

print("Loading weekly_anomalies_v2.nc …")
ds = xr.open_dataset("weekly_anomalies_v2.nc")
lats = ds["latitude"].values
lons = ds["longitude"].values
weeks = [1, 2, 3, 4, 5, 6]
wlabels = ["W1 (d1–7)", "W2 (d8–14)", "W3 (d15–21)",
           "W4 (d22–28)", "W5 (d29–35)", "W6 (d36–42)"]
init_times = ds["init_time"].values
import pandas as pd
it = pd.DatetimeIndex(init_times)
init_range = f"{it[0]:%Y-%m-%d} to {it[-1]:%Y-%m-%d}"

proj = ccrs.PlateCarree()
extent = [55, 105, 0, 50]
letters = list(string.ascii_lowercase)

def imean(a):
    return float(np.nanmean(a))

def base_map(ax, data, cmap, norm, title="", letter=None, corner=None,
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
    if corner:
        ax.text(0.97, 0.04, corner, transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))
    return im

# ══════════════════════════════════════════════════════════════════════════════
# Generic 2×6 Spire-vs-ERA5 anomaly comparison
# ══════════════════════════════════════════════════════════════════════════════
def make_comparison_2x6(spire_var, era5_var, vmax, unit, cmap, fname, title,
                        row1_label, row2_label, ticks=None):
    sp = ds[spire_var].mean("init_time").values   # (week, lat, lon)
    e5 = ds[era5_var].mean("init_time").values
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.8),
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.055, right=0.905, bottom=0.06, top=0.87,
                        hspace=0.08, wspace=0.05)
    for i, wk in enumerate(weeks):
        wi = wk - 1
        im = base_map(axes[0, i], sp[wi], cmap, norm, title=wlabels[i],
                      letter=f"({letters[i]})",
                      corner=f"μ={imean(sp[wi]):+.1f}",
                      left_label=(i == 0))
        im = base_map(axes[1, i], e5[wi], cmap, norm, title="",
                      letter=f"({letters[i+6]})",
                      corner=f"μ={imean(e5[wi]):+.1f}",
                      left_label=(i == 0), bottom_label=True)

    fig.text(0.022, 0.66, row1_label, ha="center", va="center",
             fontsize=9.5, fontweight="bold", rotation=90)
    fig.text(0.022, 0.27, row2_label, ha="center", va="center",
             fontsize=9.5, fontweight="bold", rotation=90)

    cax = fig.add_axes([0.918, 0.06, 0.013, 0.81])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both",
                      ticks=ticks)
    cb.set_label(unit, fontsize=9, labelpad=5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.955)
    fig.savefig(f"{OUT}/{fname}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")

# ── Fig A: mean-T2m ────────────────────────────────────────────────────────────
print("Fig A: mean-T2m anomaly …")
make_comparison_2x6(
    "spire_t2m_mean_anom", "era5_t2m_mean_anom",
    vmax=6.0, unit="Mean T2m anomaly (K)", cmap=plt.cm.RdBu_r,
    fname="fig_A_t2m_mean_anomaly.png",
    title=f"Spire JFM 2026 S2S  |  2m MEAN Temperature Anomaly (90-init mean, {init_range})  |  both vs WB2 1990–2019",
    row1_label="Spire forecast\n(mean T2m)",
    row2_label="ERA5 observed\n(mean T2m)",
    ticks=[-6, -4, -2, 0, 2, 4, 6])

# ── Fig B: max-T2m ─────────────────────────────────────────────────────────────
print("Fig B: max-T2m anomaly …")
make_comparison_2x6(
    "spire_t2m_max_anom", "era5_t2m_max_anom",
    vmax=6.0, unit="Max T2m anomaly (K)", cmap=plt.cm.RdBu_r,
    fname="fig_B_t2m_max_anomaly.png",
    title=f"Spire JFM 2026 S2S  |  2m MAX Temperature Anomaly (90-init mean, {init_range})  |  both vs ERA5 1991–2020 Tmax climo",
    row1_label="Spire forecast\n(max T2m)",
    row2_label="ERA5 observed\n(max T2m)",
    ticks=[-6, -4, -2, 0, 2, 4, 6])

# ══════════════════════════════════════════════════════════════════════════════
# Bias maps (Spire − ERA5), 2×3
# ══════════════════════════════════════════════════════════════════════════════
def make_bias_2x3(spire_var, era5_var, vmax, fname, title, unit):
    bias = (ds[spire_var].mean("init_time").values -
            ds[era5_var].mean("init_time").values)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7),
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.06, top=0.88,
                        hspace=0.13, wspace=0.06)
    for i, wk in enumerate(weeks):
        r, c = divmod(i, 3)
        im = base_map(axes[r, c], bias[wk-1], plt.cm.RdBu_r, norm,
                      title=wlabels[i], letter=f"({letters[i]})",
                      corner=f"μ={imean(bias[wk-1]):+.1f}",
                      left_label=(c == 0), bottom_label=(r == 1))
    cax = fig.add_axes([0.915, 0.06, 0.014, 0.82])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both",
                      ticks=np.arange(-vmax, vmax+1, 2))
    cb.set_label(unit, fontsize=9, labelpad=5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.955)
    fig.savefig(f"{OUT}/{fname}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")

print("Fig C: mean-T2m bias …")
make_bias_2x3("spire_t2m_mean_anom", "era5_t2m_mean_anom", 6.0,
              "fig_C_bias_mean.png",
              "Spire − ERA5  |  MEAN-T2m forecast bias  |  weekly, 90-init mean",
              "Bias Spire−ERA5 (K)")
print("Fig D: max-T2m bias …")
make_bias_2x3("spire_t2m_max_anom", "era5_t2m_max_anom", 6.0,
              "fig_D_bias_max.png",
              "Spire − ERA5  |  MAX-T2m forecast bias  |  weekly, 90-init mean",
              "Bias Spire−ERA5 (K)")

# ══════════════════════════════════════════════════════════════════════════════
# Skill summary: ACC (per-gridpoint Pearson across 90 inits, then India-mean),
#                RMSE, and India-mean anomaly vs lead week — for both variants
# ══════════════════════════════════════════════════════════════════════════════
print("Fig E: skill summary …")

def acc_rmse_per_week(spire_var, era5_var):
    sp = ds[spire_var].values   # (init, week, lat, lon)
    e5 = ds[era5_var].values
    nW = sp.shape[1]
    acc = np.full(nW, np.nan)
    rmse = np.full(nW, np.nan)
    for w in range(nW):
        spw = sp[:, w].reshape(sp.shape[0], -1)   # (init, npix)
        e5w = e5[:, w].reshape(e5.shape[0], -1)
        # ACC: correlation across inits at each pixel, then mean over pixels
        rs = []
        for p in range(spw.shape[1]):
            a, b = spw[:, p], e5w[:, p]
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                rs.append(pearsonr(a, b)[0])
        acc[w] = np.nanmean(rs) if rs else np.nan
        rmse[w] = np.sqrt(np.nanmean((spw - e5w) ** 2))
    return acc, rmse

acc_mean, rmse_mean = acc_rmse_per_week("spire_t2m_mean_anom", "era5_t2m_mean_anom")
acc_max,  rmse_max  = acc_rmse_per_week("spire_t2m_max_anom",  "era5_t2m_max_anom")

sp_mean_series = [imean(ds["spire_t2m_mean_anom"].mean("init_time").values[w-1]) for w in weeks]
e5_mean_series = [imean(ds["era5_t2m_mean_anom"].mean("init_time").values[w-1])  for w in weeks]
sp_max_series  = [imean(ds["spire_t2m_max_anom"].mean("init_time").values[w-1])  for w in weeks]
e5_max_series  = [imean(ds["era5_t2m_max_anom"].mean("init_time").values[w-1])   for w in weeks]

x = np.arange(1, 7)
xt = ["W1\nd1-7", "W2\nd8-14", "W3\nd15-21", "W4\nd22-28", "W5\nd29-35", "W6\nd36-42"]

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(left=0.06, right=0.985, bottom=0.13, top=0.86, wspace=0.27)

# (a) ACC
ax[0].plot(x, acc_mean, "o-", color="#D73027", lw=2, ms=7, label="mean-T2m")
ax[0].plot(x, acc_max,  "s-", color="#2166AC", lw=2, ms=7, label="max-T2m")
ax[0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax[0].axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.8)
ax[0].fill_between([0.5, 6.5], 0.5, 1.0, color="green", alpha=0.05)
ax[0].text(6.35, 0.5, "skill\n(ACC=0.5)", fontsize=7, color="gray", va="center")
ax[0].set_xlim(0.5, 6.5); ax[0].set_ylim(-0.3, 1.0)
ax[0].set_xticks(x); ax[0].set_xticklabels(xt, fontsize=8)
ax[0].set_ylabel("ACC", fontsize=10)
ax[0].set_title("(a) Anomaly Correlation Coefficient", fontsize=10.5, fontweight="bold")
ax[0].legend(fontsize=9, loc="upper right"); ax[0].grid(True, alpha=0.3)

# (b) RMSE
ax[1].plot(x, rmse_mean, "o-", color="#D73027", lw=2, ms=7, label="mean-T2m")
ax[1].plot(x, rmse_max,  "s-", color="#2166AC", lw=2, ms=7, label="max-T2m")
ax[1].set_xticks(x); ax[1].set_xticklabels(xt, fontsize=8)
ax[1].set_ylabel("RMSE (K)", fontsize=10)
ax[1].set_title("(b) RMSE vs ERA5", fontsize=10.5, fontweight="bold")
ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)

# (c) India-mean anomaly: forecast vs observed
ax[2].plot(x, sp_mean_series, "o-",  color="#D73027", lw=2, ms=7, label="Spire mean")
ax[2].plot(x, e5_mean_series, "o--", color="#F4A582", lw=2, ms=7, label="ERA5 mean")
ax[2].plot(x, sp_max_series,  "s-",  color="#2166AC", lw=2, ms=7, label="Spire max")
ax[2].plot(x, e5_max_series,  "s--", color="#92C5DE", lw=2, ms=7, label="ERA5 max")
ax[2].axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax[2].set_xticks(x); ax[2].set_xticklabels(xt, fontsize=8)
ax[2].set_ylabel("India-mean anomaly (K)", fontsize=10)
ax[2].set_title("(c) India-mean T2m anomaly", fontsize=10.5, fontweight="bold")
ax[2].legend(fontsize=8.5, ncol=2); ax[2].grid(True, alpha=0.3)

fig.suptitle(f"Spire JFM 2026 S2S  |  India-domain Skill Summary  |  90 init dates ({init_range})",
             fontsize=12, fontweight="bold", y=0.965)
fig.savefig(f"{OUT}/fig_E_skill_summary.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  → fig_E_skill_summary.png")

print("\nAll v2 figures saved to", OUT)
