"""
12_master_publication_plots.py

Master publication-quality figure generator for Spire JFM 2026 S2S Verification.
All figures read from weekly_anomalies_v2.nc (the corrected, consistent-baseline data).

Figures produced (all saved to figures/pub/):
  ── Spatial anomaly maps ──
  P01_anomaly_mean.png   — 2×6, Spire vs ERA5 mean-T2m anomaly
  P02_anomaly_max.png    — 2×6, Spire vs ERA5 max-T2m anomaly

  ── Bias maps ──
  P03_bias_mean.png      — 2×3, Spire−ERA5 mean-T2m bias
  P04_bias_max.png       — 2×3, Spire−ERA5 max-T2m bias

  ── Scatter (gridcell, 90-init mean) with R², RMSE, MAE ──
  P05_scatter_mean.png   — 2×3, T2m-mean: Spire vs ERA5 per gridcell
  P06_scatter_max.png    — 2×3, T2m-max:  Spire vs ERA5 per gridcell

  ── Scatter (per init date, India-mean) ──
  P07_scatter_initmean_t2m.png  — 2×3, T2m-mean India-mean vs ERA5 (each dot = 1 init)
  P08_scatter_initmean_max.png  — 2×3, T2m-max  India-mean

  ── Line / skill plots ──
  P09_acc_skill_maps.png    — 3×6, per-gridpoint ACC maps for T2m / Precip / Z500
  P10_acc_vs_lead.png       — India-mean ACC vs lead for T2m-mean, T2m-max, Precip, Z500
  P11_rmse_vs_lead.png      — RMSE vs lead
  P12_bias_vs_lead.png      — India-mean bias vs lead
  P13_anomaly_vs_lead.png   — India-mean anomaly (Spire vs ERA5) vs lead
  P14_skill_dashboard.png   — 6-panel combined dashboard (ACC, RMSE, bias,
                               scatter summary, anomaly, Taylor diagram)
  P15_acc_rmse_heatmap.png  — heatmap: variable × week, ACC / RMSE side-by-side
"""

import os
import string
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATA_FILE = "weekly_anomalies_v2.nc"
OUT_DIR   = "figures/pub"
os.makedirs(OUT_DIR, exist_ok=True)

DPI   = 300
PROJ  = ccrs.PlateCarree()
EXTNT = [55, 105, 0, 50]

WEEKS   = [1, 2, 3, 4, 5, 6]
WLABELS = ["W1\nd1–7", "W2\nd8–14", "W3\nd15–21",
           "W4\nd22–28", "W5\nd29–35", "W6\nd36–42"]
WLABELS_SHORT = ["W1", "W2", "W3", "W4", "W5", "W6"]

# Colour palette
C_SPIRE = "#D73027"    # warm red
C_ERA5  = "#2166AC"    # cool blue
C_PRECIP = "#1a9850"   # green
C_Z500   = "#762a83"   # purple
C_TMAX   = "#F46D43"   # orange

# Typography
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     10.5,
    "axes.labelsize":     9.5,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#FAFAFA",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

letters = list(string.ascii_lowercase)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading weekly_anomalies_v2.nc …")
ds   = xr.open_dataset(DATA_FILE)
lats = ds["latitude"].values
lons = ds["longitude"].values
it   = pd.DatetimeIndex(ds["init_time"].values)
INIT_RANGE = f"{it[0]:%Y-%m-%d} to {it[-1]:%Y-%m-%d}"
N_INITS    = len(it)
print(f"  {N_INITS} init dates, {len(lats)}×{len(lons)} grid, {it[0].date()}–{it[-1].date()}")

# Convenience: init-mean fields (week, lat, lon)
SP_MEAN = ds["spire_t2m_mean_anom"].mean("init_time").values
E5_MEAN = ds["era5_t2m_mean_anom"].mean("init_time").values
SP_MAX  = ds["spire_t2m_max_anom"].mean("init_time").values
E5_MAX  = ds["era5_t2m_max_anom"].mean("init_time").values
SP_PREC = ds["spire_precip_anom"].mean("init_time").values
E5_PREC = ds["era5_precip_anom"].mean("init_time").values
SP_Z500 = ds["spire_z500_anom"].mean("init_time").values
E5_Z500 = ds["era5_z500_anom"].mean("init_time").values

# India-mean time series (init, week)
def india_mean(var):
    return np.nanmean(ds[var].values, axis=(-1, -2))

SP_MEAN_TS = india_mean("spire_t2m_mean_anom")
E5_MEAN_TS = india_mean("era5_t2m_mean_anom")
SP_MAX_TS  = india_mean("spire_t2m_max_anom")
E5_MAX_TS  = india_mean("era5_t2m_max_anom")
SP_PREC_TS = india_mean("spire_precip_anom")
E5_PREC_TS = india_mean("era5_precip_anom")
SP_Z500_TS = india_mean("spire_z500_anom")
E5_Z500_TS = india_mean("era5_z500_anom")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig, name):
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {name}")


def metrics(m, o):
    """R², RMSE, MAE, bias, Pearson-r from two flat arrays."""
    mask = np.isfinite(m) & np.isfinite(o)
    m, o = m[mask], o[mask]
    if len(m) < 5:
        return dict(r2=np.nan, r=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan, n=0)
    r, _ = pearsonr(m, o)
    return dict(r2=r**2, r=r, rmse=float(np.sqrt(np.mean((m-o)**2))),
                mae=float(np.mean(np.abs(m-o))),
                bias=float(np.mean(m-o)), n=int(mask.sum()))


def base_map(ax, data, cmap, norm, title="", letter=None, corner=None,
             left_label=False, bottom_label=False):
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                       transform=PROJ, rasterized=True, shading="auto")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.75, edgecolor="k", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.35, edgecolor="0.4", zorder=4, linestyle=":")
    ax.set_extent(EXTNT, crs=PROJ)
    gl = ax.gridlines(crs=PROJ, linewidth=0.25, color="gray",
                      linestyle=":", alpha=0.5, zorder=3)
    gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
    gl.top_labels = gl.right_labels = False
    gl.left_labels   = left_label
    gl.bottom_labels  = bottom_label
    gl.xlabel_style  = {"size": 7, "color": "0.35"}
    gl.ylabel_style  = {"size": 7, "color": "0.35"}
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
    if letter:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.88))
    if corner:
        ax.text(0.975, 0.04, corner, transform=ax.transAxes, fontsize=7.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.88))
    return im


def add_colorbar(fig, im, cax, label, ticks=None):
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both", ticks=ticks)
    cb.set_label(label, fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=8)
    return cb


def acc_rmse_series(spire_arr, era5_arr):
    """Per-week: pixel-mean ACC, India-mean RMSE across init dates."""
    nW = spire_arr.shape[1]
    acc  = np.full(nW, np.nan)
    rmse = np.full(nW, np.nan)
    bias_arr = np.full(nW, np.nan)
    for w in range(nW):
        sp = spire_arr[:, w].reshape(N_INITS, -1)
        e5 = era5_arr[:, w].reshape(N_INITS, -1)
        rs = [pearsonr(sp[:, p], e5[:, p])[0]
              for p in range(sp.shape[1])
              if np.std(sp[:, p]) > 1e-6 and np.std(e5[:, p]) > 1e-6]
        acc[w]  = np.nanmean(rs) if rs else np.nan
        rmse[w] = float(np.sqrt(np.nanmean((sp - e5)**2)))
        bias_arr[w] = float(np.nanmean(sp - e5))
    return acc, rmse, bias_arr


# Pre-compute skill curves for all 4 variables
print("Computing skill curves …")
ACC_MEAN, RMSE_MEAN, BIAS_MEAN = acc_rmse_series(ds["spire_t2m_mean_anom"].values, ds["era5_t2m_mean_anom"].values)
ACC_MAX,  RMSE_MAX,  BIAS_MAX  = acc_rmse_series(ds["spire_t2m_max_anom"].values,  ds["era5_t2m_max_anom"].values)
ACC_PREC, RMSE_PREC, BIAS_PREC = acc_rmse_series(ds["spire_precip_anom"].values,   ds["era5_precip_anom"].values)
ACC_Z500, RMSE_Z500, BIAS_Z500 = acc_rmse_series(ds["spire_z500_anom"].values,     ds["era5_z500_anom"].values)

x_wk = np.arange(1, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# P01 — Anomaly spatial comparison: mean-T2m
# ═══════════════════════════════════════════════════════════════════════════════
def plot_anomaly_2x6(sp, e5, vmax, unit, cmap, fname, title, row1, row2, ticks):
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 6, figsize=(20, 7.5),
                              subplot_kw={"projection": PROJ})
    fig.subplots_adjust(left=0.055, right=0.9, bottom=0.05, top=0.87,
                        hspace=0.06, wspace=0.04)
    for i, wk in enumerate(WEEKS):
        wi = wk - 1
        base_map(axes[0, i], sp[wi], cmap, norm,
                 title=WLABELS[i].replace("\n", " "),
                 letter=f"({letters[i]})",
                 corner=f"μ={np.nanmean(sp[wi]):+.2f}",
                 left_label=(i == 0))
        im = base_map(axes[1, i], e5[wi], cmap, norm,
                      letter=f"({letters[i+6]})",
                      corner=f"μ={np.nanmean(e5[wi]):+.2f}",
                      left_label=(i == 0), bottom_label=True)
    fig.text(0.022, 0.67, row1, ha="center", va="center",
             fontsize=10, fontweight="bold", rotation=90)
    fig.text(0.022, 0.27, row2, ha="center", va="center",
             fontsize=10, fontweight="bold", rotation=90)
    cax = fig.add_axes([0.912, 0.05, 0.012, 0.82])
    add_colorbar(fig, im, cax, unit, ticks=ticks)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.96)
    save(fig, fname)

print("\nP01 — Anomaly mean …")
plot_anomaly_2x6(SP_MEAN, E5_MEAN, 6, "Mean T2m anomaly (K)", plt.cm.RdBu_r,
    "P01_anomaly_mean.png",
    f"Spire JFM 2026 S2S  |  2m MEAN Temperature Anomaly (90-init mean, {INIT_RANGE})  |  both vs WB2 1990–2019",
    "Spire forecast\n(mean T2m)", "ERA5 observed\n(mean T2m)",
    [-6, -4, -2, 0, 2, 4, 6])

print("P02 — Anomaly max …")
plot_anomaly_2x6(SP_MAX, E5_MAX, 6, "Max T2m anomaly (K)", plt.cm.RdBu_r,
    "P02_anomaly_max.png",
    f"Spire JFM 2026 S2S  |  2m MAX Temperature Anomaly (90-init mean, {INIT_RANGE})  |  both vs ERA5 1991–2020 Tmax climo",
    "Spire forecast\n(max T2m)", "ERA5 observed\n(max T2m)",
    [-6, -4, -2, 0, 2, 4, 6])

# ═══════════════════════════════════════════════════════════════════════════════
# P03 / P04 — Bias maps
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bias_2x3(sp, e5, vmax, unit, fname, title):
    bias = sp - e5  # (week, lat, lon)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                              subplot_kw={"projection": PROJ})
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.05, top=0.87,
                        hspace=0.12, wspace=0.06)
    for i, wk in enumerate(WEEKS):
        r, c = divmod(i, 3)
        im = base_map(axes[r, c], bias[wk-1], plt.cm.RdBu_r, norm,
                      title=WLABELS[i].replace("\n", " "),
                      letter=f"({letters[i]})",
                      corner=f"μ={np.nanmean(bias[wk-1]):+.2f}",
                      left_label=(c == 0), bottom_label=(r == 1))
    cax = fig.add_axes([0.915, 0.05, 0.013, 0.82])
    add_colorbar(fig, im, cax, unit, ticks=np.arange(-vmax, vmax+1, 2))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.96)
    save(fig, fname)

print("P03 — Bias mean …")
plot_bias_2x3(SP_MEAN, E5_MEAN, 6, "Bias Spire−ERA5 (K)",
    "P03_bias_mean.png",
    "Spire − ERA5  |  MEAN-T2m forecast bias  |  weekly, 90-init mean")
print("P04 — Bias max …")
plot_bias_2x3(SP_MAX, E5_MAX, 6, "Bias Spire−ERA5 (K)",
    "P04_bias_max.png",
    "Spire − ERA5  |  MAX-T2m forecast bias  |  weekly, 90-init mean")

# ═══════════════════════════════════════════════════════════════════════════════
# P05 / P06 — Scatter (gridcell, 90-init mean)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_scatter_gridcell(sp, e5, xlim, unit, fname, title, color, cmap_density):
    """2×3 scatter per week. Each point = one gridcell (init-time averaged)."""
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.91,
                        hspace=0.40, wspace=0.30)
    lo, hi = xlim

    for i, wk in enumerate(WEEKS):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        wi = wk - 1
        spw = sp[wi].ravel()
        e5w = e5[wi].ravel()

        mask = np.isfinite(spw) & np.isfinite(e5w)
        spw, e5w = spw[mask], e5w[mask]
        m = metrics(spw, e5w)

        # Density-coloured scatter
        if len(spw) > 3000:
            idx = np.random.choice(len(spw), 3000, replace=False)
            spw_p, e5w_p = spw[idx], e5w[idx]
        else:
            spw_p, e5w_p = spw, e5w

        try:
            xy   = np.vstack([e5w_p, spw_p])
            kde  = gaussian_kde(xy)(xy)
            sort = kde.argsort()
            sc   = ax.scatter(e5w_p[sort], spw_p[sort], c=kde[sort], s=12,
                              cmap=cmap_density, alpha=0.75, linewidths=0,
                              rasterized=True)
        except Exception:
            ax.scatter(e5w_p, spw_p, s=10, alpha=0.5, color=color,
                       linewidths=0, rasterized=True)

        # 1:1 line
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, alpha=0.6, label="1:1")
        # OLS regression line
        if len(spw) >= 5:
            coef = np.polyfit(e5w, spw, 1)
            xfit = np.linspace(lo, hi, 100)
            ax.plot(xfit, np.polyval(coef, xfit), "-",
                    color=color, lw=1.8, alpha=0.9,
                    label=f"OLS (k={coef[0]:.2f})")

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ERA5 (ground truth)  {unit}", fontsize=9)
        ax.set_ylabel(f"Spire (model)  {unit}", fontsize=9)
        ax.set_title(f"({letters[i]}) {WLABELS[i].replace(chr(10), ' ')}",
                     fontsize=10.5, fontweight="bold")
        ax.set_aspect("equal", "box")

        # Stats box
        stats_txt = (f"r  = {m['r']:+.3f}\n"
                     f"R² = {m['r2']:.3f}\n"
                     f"RMSE = {m['rmse']:.3f} K\n"
                     f"MAE  = {m['mae']:.3f} K\n"
                     f"Bias = {m['bias']:+.3f} K\n"
                     f"n = {m['n']}")
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFFDE7",
                          ec="0.55", alpha=0.95))
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.97)
    save(fig, fname)


print("P05 — Scatter gridcell mean …")
plot_scatter_gridcell(
    SP_MEAN, E5_MEAN, (-8, 8), "(K)",
    "P05_scatter_mean.png",
    f"Spire vs ERA5  |  Mean-T2m Anomaly — grid-cell scatter (90-init mean)  |  {INIT_RANGE}",
    C_ERA5, "Blues")

print("P06 — Scatter gridcell max …")
plot_scatter_gridcell(
    SP_MAX, E5_MAX, (-8, 8), "(K)",
    "P06_scatter_max.png",
    f"Spire vs ERA5  |  Max-T2m Anomaly — grid-cell scatter (90-init mean)  |  {INIT_RANGE}",
    C_SPIRE, "Reds")

# ═══════════════════════════════════════════════════════════════════════════════
# P07 / P08 — Scatter per init date (India-mean)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_scatter_india_init(sp_ts, e5_ts, xlim, unit, fname, title, color):
    """2×3 scatter, each dot = one init date, X=ERA5 India-mean, Y=Spire India-mean."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.91,
                        hspace=0.40, wspace=0.30)
    lo, hi = xlim

    for i, wk in enumerate(WEEKS):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        wi = wk - 1
        spw = sp_ts[:, wi]
        e5w = e5_ts[:, wi]
        m = metrics(spw, e5w)

        # colour points by Julian date for temporal context
        jdays = it.dayofyear
        sc = ax.scatter(e5w, spw, c=jdays, cmap="twilight_shifted",
                        s=35, alpha=0.85, edgecolors="white",
                        linewidths=0.4, zorder=3, rasterized=True)

        # colourbar (hide — add once per subplot row? or just remove it)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="1:1")
        if len(spw) >= 5:
            coef = np.polyfit(e5w, spw, 1)
            xfit = np.linspace(lo, hi, 100)
            ax.plot(xfit, np.polyval(coef, xfit), "-",
                    color=color, lw=1.8, alpha=0.9,
                    label=f"OLS (k={coef[0]:.2f})")

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"ERA5 India-mean  {unit}", fontsize=9)
        ax.set_ylabel(f"Spire India-mean  {unit}", fontsize=9)
        ax.set_title(f"({letters[i]}) {WLABELS[i].replace(chr(10), ' ')}  (n={N_INITS} inits)",
                     fontsize=10.5, fontweight="bold")
        ax.set_aspect("equal", "box")

        stats_txt = (f"r  = {m['r']:+.3f}\n"
                     f"R² = {m['r2']:.3f}\n"
                     f"RMSE = {m['rmse']:.3f} K\n"
                     f"MAE  = {m['mae']:.3f} K\n"
                     f"Bias = {m['bias']:+.3f} K")
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFFDE7",
                          ec="0.55", alpha=0.95))
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.97)
    save(fig, fname)


print("P07 — Scatter init-mean T2m-mean …")
plot_scatter_india_init(
    SP_MEAN_TS, E5_MEAN_TS, (-6, 6), "(K)",
    "P07_scatter_initmean_t2m.png",
    f"Spire vs ERA5  |  India-mean Mean-T2m Anomaly (each point = 1 init date)  |  {INIT_RANGE}",
    C_ERA5)

print("P08 — Scatter init-mean T2m-max …")
plot_scatter_india_init(
    SP_MAX_TS, E5_MAX_TS, (-6, 6), "(K)",
    "P08_scatter_initmean_max.png",
    f"Spire vs ERA5  |  India-mean Max-T2m Anomaly (each point = 1 init date)  |  {INIT_RANGE}",
    C_SPIRE)

# ═══════════════════════════════════════════════════════════════════════════════
# P09 — ACC skill maps (3 variables × 6 weeks)
# ═══════════════════════════════════════════════════════════════════════════════
print("P09 — ACC skill maps …")

def compute_acc_maps(spire_var, era5_var):
    """Pixelwise Pearson r across 90 inits → (week, lat, lon) ACC map."""
    sp = ds[spire_var].values   # (init, week, lat, lon)
    e5 = ds[era5_var].values
    nW, nLat, nLon = sp.shape[1], sp.shape[2], sp.shape[3]
    acc = np.full((nW, nLat, nLon), np.nan, dtype=np.float32)
    for w in range(nW):
        spw = sp[:, w]  # (init, lat, lon)
        e5w = e5[:, w]
        for la in range(nLat):
            for lo in range(nLon):
                a, b = spw[:, la, lo], e5w[:, la, lo]
                if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                    acc[w, la, lo] = pearsonr(a, b)[0]
    return acc

print("  computing T2m ACC maps …")
ACC_MAP_MEAN = compute_acc_maps("spire_t2m_mean_anom", "era5_t2m_mean_anom")
print("  computing Precip ACC maps …")
ACC_MAP_PREC = compute_acc_maps("spire_precip_anom",   "era5_precip_anom")
print("  computing Z500 ACC maps …")
ACC_MAP_Z500 = compute_acc_maps("spire_z500_anom",     "era5_z500_anom")

acc_cmap = plt.cm.RdYlGn
acc_norm = BoundaryNorm(np.arange(-1, 1.05, 0.1), ncolors=256, clip=True)

fig, axes = plt.subplots(3, 6, figsize=(20, 10.5),
                          subplot_kw={"projection": PROJ})
fig.subplots_adjust(left=0.055, right=0.91, bottom=0.04, top=0.88,
                    hspace=0.08, wspace=0.04)

row_data   = [ACC_MAP_MEAN, ACC_MAP_PREC, ACC_MAP_Z500]
row_labels = ["T2m mean\nACC", "Precipitation\nACC", "Z500\nACC"]
row_units  = ["T2m-mean India r=", "Precip India r=", "Z500 India r="]
row_acc_ts = [ACC_MEAN, ACC_PREC, ACC_Z500]

for ri, (rdat, rlbl, rac) in enumerate(zip(row_data, row_labels, row_acc_ts)):
    for ci, wk in enumerate(WEEKS):
        wi = wk - 1
        india_r = rac[wi]
        im = base_map(axes[ri, ci], rdat[wi], acc_cmap, acc_norm,
                      title=(WLABELS[ci].replace("\n", " ") if ri == 0 else ""),
                      letter=f"({letters[ri*6+ci]})",
                      corner=f"India r={india_r:+.2f}" if np.isfinite(india_r) else "",
                      left_label=(ci == 0), bottom_label=(ri == 2))
    fig.text(0.022, [0.76, 0.505, 0.245][ri], rlbl,
             ha="center", va="center", fontsize=10, fontweight="bold", rotation=90)

cax = fig.add_axes([0.925, 0.04, 0.012, 0.84])
cb = fig.colorbar(im, cax=cax, orientation="vertical",
                  ticks=np.arange(-1, 1.1, 0.2))
cb.set_label("Anomaly Correlation Coefficient (ACC)", fontsize=9, labelpad=6)
cb.ax.tick_params(labelsize=8)

# Draw ACC=0 and ACC=0.5 contours as hatching reference lines in legend
fig.suptitle(f"Spire JFM 2026 S2S  |  ACC Skill Maps (pixel-wise, 90 inits)  |  stippling: |ACC| < 0.3",
             fontsize=12, fontweight="bold", y=0.945)

# Stippling: |ACC| < 0.3
for ri, rdat in enumerate(row_data):
    for ci, wk in enumerate(WEEKS):
        wi = wk - 1
        mask = np.abs(rdat[wi]) < 0.3
        yl, xl = np.where(mask)
        if len(yl) > 0:
            axes[ri, ci].plot(lons[xl], lats[yl], ".", color="0.3",
                              ms=0.8, alpha=0.4, transform=PROJ, zorder=5)

save(fig, "P09_acc_skill_maps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P10 — ACC vs lead (all 4 variables)
# ═══════════════════════════════════════════════════════════════════════════════
print("P10 — ACC vs lead …")

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.1, right=0.82, bottom=0.13, top=0.88)

lines = [
    (ACC_MEAN, C_SPIRE, "o-", "T2m-mean", 2.0),
    (ACC_MAX,  C_TMAX,  "s-", "T2m-max",  2.0),
    (ACC_PREC, C_PRECIP,"^-", "Precip",   2.0),
    (ACC_Z500, C_Z500,  "D-", "Z500",     2.0),
]
for acc, col, mk, lbl, lw in lines:
    ax.plot(x_wk, acc, mk, color=col, lw=lw, ms=8, label=lbl,
            markeredgecolor="white", markeredgewidth=0.5)

ax.axhline(0.0,  color="k",    lw=0.9, ls="--", alpha=0.5)
ax.axhline(0.3,  color="gray", lw=0.8, ls=":",  alpha=0.7)
ax.axhline(0.5,  color="gray", lw=1.0, ls="-.", alpha=0.7)
ax.fill_between([0.5, 6.5], 0.5, 1.0, color="#4caf50", alpha=0.07)
ax.fill_between([0.5, 6.5], -1, 0,    color="#f44336", alpha=0.04)

ax.text(6.35, 0.5,  "ACC=0.5\n(skill)", fontsize=7.5, va="center", color="gray")
ax.text(6.35, 0.3,  "ACC=0.3",          fontsize=7.5, va="center", color="gray")

ax.set_xlim(0.5, 6.5); ax.set_ylim(-0.45, 1.0)
ax.set_xticks(x_wk); ax.set_xticklabels(WLABELS, fontsize=9)
ax.set_ylabel("India-mean ACC (Pearson r)", fontsize=11)
ax.set_xlabel("Forecast lead week", fontsize=11)
ax.set_title(f"Spire JFM 2026 S2S  |  India-mean Anomaly Correlation Coefficient vs Lead\n"
             f"{INIT_RANGE}", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
          bbox_to_anchor=(1.24, 0.99))
ax.grid(True, alpha=0.35)
save(fig, "P10_acc_vs_lead.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P11 — RMSE vs lead
# ═══════════════════════════════════════════════════════════════════════════════
print("P11 — RMSE vs lead …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.86, wspace=0.28)

# left: T2m-mean and T2m-max
axes[0].plot(x_wk, RMSE_MEAN, "o-", color=C_SPIRE, lw=2, ms=8,
             label="T2m-mean", markeredgecolor="white", markeredgewidth=0.5)
axes[0].plot(x_wk, RMSE_MAX,  "s-", color=C_TMAX,  lw=2, ms=8,
             label="T2m-max",  markeredgecolor="white", markeredgewidth=0.5)
axes[0].set_ylabel("RMSE (K)", fontsize=11)
axes[0].set_title("(a) Temperature RMSE", fontsize=11, fontweight="bold")
axes[0].set_xticks(x_wk); axes[0].set_xticklabels(WLABELS, fontsize=9)
axes[0].legend(fontsize=10)

# right: Precip and Z500 (different axes)
ax_p = axes[1]
ax_z = ax_p.twinx()
ln1 = ax_p.plot(x_wk, RMSE_PREC, "^-", color=C_PRECIP, lw=2, ms=8,
                label="Precip (mm/day)", markeredgecolor="white", markeredgewidth=0.5)
ln2 = ax_z.plot(x_wk, RMSE_Z500, "D-", color=C_Z500,  lw=2, ms=8,
                label="Z500 (gpm)", markeredgecolor="white", markeredgewidth=0.5)
ax_p.set_ylabel("RMSE Precip (mm/day)", fontsize=10, color=C_PRECIP)
ax_z.set_ylabel("RMSE Z500 (gpm)",      fontsize=10, color=C_Z500)
ax_p.tick_params(axis="y", labelcolor=C_PRECIP)
ax_z.tick_params(axis="y", labelcolor=C_Z500)
ax_p.set_title("(b) Precip & Z500 RMSE", fontsize=11, fontweight="bold")
ax_p.set_xticks(x_wk); ax_p.set_xticklabels(WLABELS, fontsize=9)
lns = ln1 + ln2
ax_p.legend(lns, [l.get_label() for l in lns], fontsize=10, loc="upper left")

for ax in [axes[0], ax_p]:
    ax.set_xlabel("Forecast lead week", fontsize=10)
    ax.grid(True, alpha=0.35)

fig.suptitle(f"Spire JFM 2026 S2S  |  RMSE vs Forecast Lead  |  {INIT_RANGE}",
             fontsize=12, fontweight="bold", y=0.95)
save(fig, "P11_rmse_vs_lead.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P12 — Bias vs lead
# ═══════════════════════════════════════════════════════════════════════════════
print("P12 — Bias vs lead …")

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.1, right=0.82, bottom=0.13, top=0.88)

for arr, col, mk, lbl in [
    (BIAS_MEAN, C_SPIRE, "o-", "T2m-mean"),
    (BIAS_MAX,  C_TMAX,  "s-", "T2m-max"),
    (BIAS_PREC, C_PRECIP,"^-", "Precip"),
    (BIAS_Z500, C_Z500,  "D-", "Z500"),
]:
    ax.plot(x_wk, arr, mk, color=col, lw=2, ms=8, label=lbl,
            markeredgecolor="white", markeredgewidth=0.5)

ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.6)
ax.fill_between(x_wk, 0, BIAS_MEAN, alpha=0.12, color=C_SPIRE)
ax.set_xlim(0.5, 6.5)
ax.set_xticks(x_wk); ax.set_xticklabels(WLABELS, fontsize=9)
ax.set_ylabel("India-mean bias (Spire − ERA5)", fontsize=11)
ax.set_xlabel("Forecast lead week", fontsize=11)
ax.set_title(f"Spire JFM 2026 S2S  |  India-mean Forecast Bias vs Lead\n{INIT_RANGE}",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
          bbox_to_anchor=(1.24, 0.99))
ax.grid(True, alpha=0.35)
save(fig, "P12_bias_vs_lead.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P13 — India-mean anomaly vs lead (Spire vs ERA5)
# ═══════════════════════════════════════════════════════════════════════════════
print("P13 — Anomaly vs lead …")

def imean_wk(ts):
    return [float(np.nanmean(ts[:, w])) for w in range(6)]

sp_mean_wk = imean_wk(SP_MEAN_TS)
e5_mean_wk = imean_wk(E5_MEAN_TS)
sp_max_wk  = imean_wk(SP_MAX_TS)
e5_max_wk  = imean_wk(E5_MAX_TS)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.86, wspace=0.30)

for ax, sp, e5, unit, title_sfx in [
    (axes[0], sp_mean_wk, e5_mean_wk, "K", "(a) Mean T2m"),
    (axes[1], sp_max_wk,  e5_max_wk,  "K", "(b) Max T2m"),
]:
    ax.plot(x_wk, sp, "o-", color=C_SPIRE, lw=2, ms=8, label="Spire forecast",
            markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(x_wk, e5, "s--", color=C_ERA5, lw=2, ms=8, label="ERA5 observed",
            markeredgecolor="white", markeredgewidth=0.5)
    ax.fill_between(x_wk, sp, e5, alpha=0.12, color="gray", label="Spire–ERA5 gap")
    ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.5)
    ax.set_xticks(x_wk); ax.set_xticklabels(WLABELS, fontsize=9)
    ax.set_ylabel(f"India-mean anomaly ({unit})", fontsize=10)
    ax.set_xlabel("Forecast lead week", fontsize=10)
    ax.set_title(title_sfx, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5); ax.grid(True, alpha=0.35)

fig.suptitle(f"Spire JFM 2026 S2S  |  India-mean T2m Anomaly: Spire vs ERA5  |  {INIT_RANGE}",
             fontsize=12, fontweight="bold", y=0.96)
save(fig, "P13_anomaly_vs_lead.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P14 — 6-panel skill dashboard
# ═══════════════════════════════════════════════════════════════════════════════
print("P14 — Skill dashboard …")

fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, bottom=0.07, top=0.90)

ax_acc  = fig.add_subplot(gs[0, 0])
ax_rmse = fig.add_subplot(gs[0, 1])
ax_bias = fig.add_subplot(gs[0, 2])
ax_anom = fig.add_subplot(gs[1, 0])
ax_scat = fig.add_subplot(gs[1, 1])
ax_heat = fig.add_subplot(gs[1, 2])

# (a) ACC
for arr, col, mk, lbl in [
    (ACC_MEAN, C_SPIRE, "o", "T2m-mean"),
    (ACC_MAX,  C_TMAX,  "s", "T2m-max"),
    (ACC_PREC, C_PRECIP,"^", "Precip"),
    (ACC_Z500, C_Z500,  "D", "Z500"),
]:
    ax_acc.plot(x_wk, arr, mk+"-", color=col, lw=2, ms=7, label=lbl,
                markeredgecolor="white", markeredgewidth=0.4)
ax_acc.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
ax_acc.axhline(0.5, color="gray", lw=0.7, ls="-.", alpha=0.7)
ax_acc.set_xlim(0.5, 6.5); ax_acc.set_ylim(-0.45, 1.0)
ax_acc.set_xticks(x_wk); ax_acc.set_xticklabels(WLABELS_SHORT, fontsize=9)
ax_acc.set_ylabel("ACC (r)", fontsize=10); ax_acc.set_title("(a) ACC vs Lead", fontsize=11, fontweight="bold")
ax_acc.legend(fontsize=8, loc="upper right"); ax_acc.grid(True, alpha=0.3)

# (b) RMSE — only T2m
ax_rmse.plot(x_wk, RMSE_MEAN, "o-", color=C_SPIRE, lw=2, ms=7,
             label="T2m-mean (K)", markeredgecolor="white", markeredgewidth=0.4)
ax_rmse.plot(x_wk, RMSE_MAX,  "s-", color=C_TMAX,  lw=2, ms=7,
             label="T2m-max (K)",  markeredgecolor="white", markeredgewidth=0.4)
ax_rmse.set_xticks(x_wk); ax_rmse.set_xticklabels(WLABELS_SHORT, fontsize=9)
ax_rmse.set_ylabel("RMSE (K)", fontsize=10); ax_rmse.set_title("(b) RMSE vs Lead", fontsize=11, fontweight="bold")
ax_rmse.legend(fontsize=9); ax_rmse.grid(True, alpha=0.3)

# (c) Bias
for arr, col, mk, lbl in [
    (BIAS_MEAN, C_SPIRE, "o", "T2m-mean"),
    (BIAS_MAX,  C_TMAX,  "s", "T2m-max"),
]:
    ax_bias.plot(x_wk, arr, mk+"-", color=col, lw=2, ms=7, label=lbl,
                 markeredgecolor="white", markeredgewidth=0.4)
ax_bias.axhline(0, color="k", lw=0.9, ls="--", alpha=0.5)
ax_bias.fill_between(x_wk, 0, BIAS_MEAN, alpha=0.1, color=C_SPIRE)
ax_bias.set_xticks(x_wk); ax_bias.set_xticklabels(WLABELS_SHORT, fontsize=9)
ax_bias.set_ylabel("Bias Spire−ERA5 (K)", fontsize=10)
ax_bias.set_title("(c) Forecast Bias vs Lead", fontsize=11, fontweight="bold")
ax_bias.legend(fontsize=9); ax_bias.grid(True, alpha=0.3)

# (d) India-mean anomaly Spire vs ERA5
ax_anom.plot(x_wk, sp_mean_wk, "o-",  color=C_SPIRE,  lw=2, ms=7, label="Spire mean")
ax_anom.plot(x_wk, e5_mean_wk, "o--", color=C_ERA5,   lw=2, ms=7, label="ERA5 mean")
ax_anom.plot(x_wk, sp_max_wk,  "s-",  color=C_TMAX,   lw=2, ms=7, label="Spire max")
ax_anom.plot(x_wk, e5_max_wk,  "s--", color="#92C5DE",lw=2, ms=7, label="ERA5 max")
ax_anom.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax_anom.set_xticks(x_wk); ax_anom.set_xticklabels(WLABELS_SHORT, fontsize=9)
ax_anom.set_ylabel("India-mean anomaly (K)", fontsize=10)
ax_anom.set_title("(d) India-mean Anomaly vs Lead", fontsize=11, fontweight="bold")
ax_anom.legend(fontsize=8, ncol=2); ax_anom.grid(True, alpha=0.3)

# (e) Scatter summary — W1 mean T2m init-scatter
spw = SP_MEAN_TS[:, 0]
e5w = E5_MEAN_TS[:, 0]
m0  = metrics(spw, e5w)
ax_scat.scatter(e5w, spw, c=it.dayofyear, cmap="plasma",
                s=35, alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)
lo, hi = -6, 6
ax_scat.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5)
if len(spw) >= 5:
    coef = np.polyfit(e5w, spw, 1)
    xfit = np.linspace(lo, hi, 100)
    ax_scat.plot(xfit, np.polyval(coef, xfit), "-", color=C_SPIRE, lw=2, alpha=0.9)
ax_scat.set_xlim(lo, hi); ax_scat.set_ylim(lo, hi)
ax_scat.set_aspect("equal", "box")
ax_scat.set_xlabel("ERA5 India-mean (K)", fontsize=10)
ax_scat.set_ylabel("Spire India-mean (K)", fontsize=10)
ax_scat.set_title("(e) W1 Init-scatter: Mean T2m", fontsize=11, fontweight="bold")
ax_scat.text(0.04, 0.97,
             f"r={m0['r']:+.3f}  R²={m0['r2']:.3f}\nRMSE={m0['rmse']:.3f} K  bias={m0['bias']:+.3f} K",
             transform=ax_scat.transAxes, fontsize=8.5, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="0.55", alpha=0.95))
ax_scat.grid(True, alpha=0.3)

# (f) Heatmap: ACC per variable × week
hm_vars = ["T2m-mean", "T2m-max", "Precip", "Z500"]
hm_acc  = np.array([ACC_MEAN, ACC_MAX, ACC_PREC, ACC_Z500])  # (4, 6)
im = ax_heat.imshow(hm_acc, aspect="auto", cmap="RdYlGn",
                    vmin=-0.5, vmax=1.0, interpolation="nearest")
ax_heat.set_xticks(range(6)); ax_heat.set_xticklabels(WLABELS_SHORT, fontsize=9)
ax_heat.set_yticks(range(4)); ax_heat.set_yticklabels(hm_vars, fontsize=10)
ax_heat.set_title("(f) ACC Heatmap (variable × week)", fontsize=11, fontweight="bold")
for vi in range(4):
    for wi in range(6):
        val = hm_acc[vi, wi]
        txt = f"{val:+.2f}" if np.isfinite(val) else "nan"
        ax_heat.text(wi, vi, txt, ha="center", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if abs(val) > 0.55 else "black")
fig.colorbar(im, ax=ax_heat, label="ACC", shrink=0.85)

fig.suptitle(f"Spire JFM 2026 S2S  |  Verification Skill Dashboard  |  {INIT_RANGE}",
             fontsize=13.5, fontweight="bold", y=0.96)
save(fig, "P14_skill_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# P15 — ACC + RMSE heatmap side-by-side
# ═══════════════════════════════════════════════════════════════════════════════
print("P15 — ACC/RMSE heatmap …")

hm_rmse = np.array([RMSE_MEAN, RMSE_MAX, RMSE_PREC, RMSE_Z500])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.85, wspace=0.35)

im_a = axes[0].imshow(hm_acc,  aspect="auto", cmap="RdYlGn",
                       vmin=-0.5, vmax=1.0, interpolation="nearest")
im_b = axes[1].imshow(hm_rmse, aspect="auto", cmap="YlOrRd",
                       vmin=0, interpolation="nearest")

for ax, im, mat, title, fmt in [
    (axes[0], im_a, hm_acc,  "(a) ACC (Pearson r)",  "{:+.2f}"),
    (axes[1], im_b, hm_rmse, "(b) RMSE (mixed units)", "{:.2f}"),
]:
    ax.set_xticks(range(6)); ax.set_xticklabels(WLABELS_SHORT, fontsize=10)
    ax.set_yticks(range(4)); ax.set_yticklabels(hm_vars, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for vi in range(4):
        for wi in range(6):
            val = mat[vi, wi]
            txt = fmt.format(val) if np.isfinite(val) else "—"
            c = "white" if (im == im_a and abs(val) > 0.5) else "black"
            ax.text(wi, vi, txt, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=c)
    fig.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(f"Spire JFM 2026 S2S  |  ACC & RMSE Heatmap by Variable × Week  |  {INIT_RANGE}",
             fontsize=12.5, fontweight="bold", y=0.99)
save(fig, "P15_acc_rmse_heatmap.png")

print(f"\n✅  All {len(os.listdir(OUT_DIR))} figures saved to {OUT_DIR}/")
print("Figures:")
for f in sorted(os.listdir(OUT_DIR)):
    size_kb = os.path.getsize(f"{OUT_DIR}/{f}") // 1024
    print(f"  {f}  ({size_kb} KB)")
