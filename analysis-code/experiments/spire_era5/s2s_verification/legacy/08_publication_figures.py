"""
08_publication_figures.py

Complete set of publication-quality figures for Spire S2S hindcast paper.
All anomalies use the same WB2 1990-2019 climatology baseline (apples-to-apples).

Figures produced:
  Fig 1: Anomaly comparison — Spire forecast vs ERA5 observed (T2M, India, W1-W6)
  Fig 2: Forecast bias maps — (Spire - ERA5) mean error, T2M
  Fig 3: ACC skill maps — T2M, Precip, Z500
  Fig 4: RMSE skill maps — T2M, Precip, Z500
  Fig 5: India-mean summary — ACC / RMSE / Bias vs lead week (line plot)
  Fig 6: Anomaly comparison — Precip (W1-W6)
  Fig 7: Anomaly comparison — Z500 (W1-W6)
"""

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
ds  = xr.open_dataset("weekly_anomalies.nc")
sk  = xr.open_dataset("skill_metrics.nc")

lats = ds["latitude"].values
lons = ds["longitude"].values
weeks = [1, 2, 3, 4, 5, 6]
week_labels = ["W1\n(d1–7)", "W2\n(d8–14)", "W3\n(d15–21)",
               "W4\n(d22–28)", "W5\n(d29–35)", "W6\n(d36–42)"]

proj   = ccrs.PlateCarree()
extent = [55, 105, 0, 50]

# Precompute India-mean masks (all land, no NaN filter needed)
def india_mean(arr):
    return float(np.nanmean(arr))

# ── Helper: single-panel map ───────────────────────────────────────────────────
def make_map(ax, data, lons, lats, cmap, norm, title, letter=None, text=None):
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                       transform=proj, rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.7, edgecolor="k", zorder=4)
    ax.set_extent(extent, crs=proj)
    gl = ax.gridlines(crs=proj, linewidth=0.3, color="gray",
                      linestyle=":", alpha=0.6, zorder=3)
    gl.xlocator = mticker.FixedLocator([60, 70, 80, 90, 100])
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
    if letter:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    if text is not None:
        ax.text(0.97, 0.03, text, transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    return im

def add_colorbar(fig, im, ax_list, label, ticks=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # place colorbar to the right of the last axis in the row
    pos_last = ax_list[-1].get_position()
    pos_first = ax_list[0].get_position()
    cbar_ax = fig.add_axes([pos_last.x1 + 0.01,
                            pos_first.y0,
                            0.012,
                            pos_first.height])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical",
                      extend="both", ticks=ticks)
    cb.set_label(label, fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7)
    return cb

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — T2M Anomaly Comparison (Spire forecast vs ERA5 observed)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: T2M anomaly comparison …")

spire_t2m = ds["spire_t2m_anom"].mean("init_time").values   # (week, lat, lon)
era5_t2m  = ds["era5_t2m_anom"].mean("init_time").values

vmax_t2m = 6.0
cmap_rdbu = plt.cm.RdBu_r
norm_t2m  = TwoSlopeNorm(vmin=-vmax_t2m, vcenter=0, vmax=vmax_t2m)

fig1, axes1 = plt.subplots(2, 6, figsize=(18, 7),
                            subplot_kw={"projection": proj})
fig1.subplots_adjust(left=0.04, right=0.91, bottom=0.06, top=0.88,
                     hspace=0.10, wspace=0.06)

letters = list(string.ascii_lowercase)
for i, wk in enumerate(weeks):
    wk_idx = wk - 1
    sp = spire_t2m[wk_idx]
    e5 = era5_t2m[wk_idx]

    # Row 0: Spire
    im1 = make_map(axes1[0, i], sp, lons, lats, cmap_rdbu, norm_t2m,
                   week_labels[i].replace("\n", " "),
                   letter=f"({letters[i]})",
                   text=f"India={india_mean(sp):+.1f}K")
    axes1[0, i].gridlines(crs=proj, linewidth=0.3, color="gray",
                          linestyle=":", alpha=0.5)

    # Row 1: ERA5
    im1 = make_map(axes1[1, i], e5, lons, lats, cmap_rdbu, norm_t2m,
                   "",
                   letter=f"({letters[i+6]})",
                   text=f"India={india_mean(e5):+.1f}K")

# Row labels
fig1.text(0.015, 0.73, "Spire T2m forecast\n(vs WB2 1990–2019 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold",
          rotation=90)
fig1.text(0.015, 0.30, "ERA5 T2m observed\n(vs WB2 1990–2019 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold",
          rotation=90)

# Shared colorbar
cbar_ax1 = fig1.add_axes([0.925, 0.06, 0.013, 0.82])
cb1 = fig1.colorbar(im1, cax=cbar_ax1, orientation="vertical",
                    extend="both",
                    ticks=[-6, -4, -2, 0, 2, 4, 6])
cb1.set_label("Temperature anomaly (K)", fontsize=9, labelpad=5)
cb1.ax.tick_params(labelsize=8)

fig1.suptitle("Spire JFM 2026 S2S  |  2m Temperature Anomaly (90-init mean)  |  Both vs WB2 1990–2019 climatology",
              fontsize=11, fontweight="bold", y=0.96)
fig1.savefig(f"{OUTPUT_DIR}/fig1_t2m_anomaly_comparison.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("  → fig1_t2m_anomaly_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Forecast Bias Maps (Spire - ERA5)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: T2M bias maps …")

bias_t2m_raw = ds["spire_t2m_anom"].mean("init_time").values - ds["era5_t2m_anom"].mean("init_time").values

norm_bias = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)

fig2, axes2 = plt.subplots(2, 3, figsize=(13, 7),
                            subplot_kw={"projection": proj})
fig2.subplots_adjust(left=0.05, right=0.91, bottom=0.06, top=0.88,
                     hspace=0.15, wspace=0.06)

for i, wk in enumerate(weeks):
    row, col = divmod(i, 3)
    ax = axes2[row, col]
    b = bias_t2m_raw[wk - 1]
    im2 = make_map(ax, b, lons, lats, cmap_rdbu, norm_bias,
                   week_labels[i].replace("\n", " "),
                   letter=f"({letters[i]})",
                   text=f"India={india_mean(b):+.1f}K")

cbar_ax2 = fig2.add_axes([0.925, 0.06, 0.013, 0.82])
cb2 = fig2.colorbar(im2, cax=cbar_ax2, orientation="vertical",
                    extend="both", ticks=[-8, -6, -4, -2, 0, 2, 4, 6, 8])
cb2.set_label("Bias (Spire − ERA5) (K)", fontsize=9, labelpad=5)
cb2.ax.tick_params(labelsize=8)

fig2.suptitle("Spire JFM 2026 S2S  |  2m T2m Forecast Bias (Spire − ERA5)  |  Weekly mean over 90 inits",
              fontsize=11, fontweight="bold", y=0.96)
fig2.savefig(f"{OUTPUT_DIR}/fig2_t2m_bias_maps.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("  → fig2_t2m_bias_maps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ACC Skill Maps (T2M, Precip, Z500)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: ACC skill maps …")

acc_vars  = ["acc_t2m",    "acc_precip",   "acc_z500"]
var_labels = ["2m Temp",    "Precipitation", "Z500"]
units      = ["T2m ACC",    "Precip ACC",    "Z500 ACC"]

cmap_acc = mcolors.LinearSegmentedColormap.from_list(
    "acc", ["#7B3F00", "#C97A3A", "#F5D49A",
            "white",
            "#9EE0D0", "#3CA7A0", "#1B5C58"], N=256)
norm_acc = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig3, axes3 = plt.subplots(3, 6, figsize=(18, 10),
                            subplot_kw={"projection": proj})
fig3.subplots_adjust(left=0.05, right=0.91, bottom=0.04, top=0.91,
                     hspace=0.12, wspace=0.06)

for vi, (vname, vlabel) in enumerate(zip(acc_vars, var_labels)):
    da_acc = sk[vname].values   # (week, lat, lon)
    for i, wk in enumerate(weeks):
        ax = axes3[vi, i]
        d = da_acc[wk - 1]
        title = week_labels[i].replace("\n", " ") if vi == 0 else ""
        im3 = make_map(ax, d, lons, lats, cmap_acc, norm_acc,
                       title,
                       letter=f"({letters[vi*6+i]})",
                       text=f"μ={india_mean(d):+.2f}")
    # Row label
    fig3.text(0.025, axes3[vi, 0].get_position().y0 +
              axes3[vi, 0].get_position().height / 2,
              vlabel, ha="center", va="center",
              fontsize=10, fontweight="bold", rotation=90)

cbar_ax3 = fig3.add_axes([0.925, 0.04, 0.013, 0.87])
cb3 = fig3.colorbar(im3, cax=cbar_ax3, orientation="vertical",
                    extend="neither",
                    ticks=[-1, -0.6, -0.2, 0, 0.2, 0.6, 1])
cb3.set_label("ACC", fontsize=9, labelpad=5)
cb3.ax.tick_params(labelsize=8)

fig3.suptitle("Spire JFM 2026 S2S  |  Anomaly Correlation Coefficient (90 inits)  |  India domain",
              fontsize=11, fontweight="bold", y=0.97)
fig3.savefig(f"{OUTPUT_DIR}/fig3_acc_skill_maps.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print("  → fig3_acc_skill_maps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — RMSE Maps (T2M, Precip, Z500)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: RMSE skill maps …")

rmse_vars   = ["rmse_t2m", "rmse_precip", "rmse_z500"]
rmse_units  = ["RMSE (K)", "RMSE (mm/d)", "RMSE (m)"]
rmse_vmaxes = [5.0, 5.0, 80.0]

fig4, axes4 = plt.subplots(3, 6, figsize=(18, 10),
                            subplot_kw={"projection": proj})
fig4.subplots_adjust(left=0.05, right=0.91, bottom=0.04, top=0.91,
                     hspace=0.12, wspace=0.06)

for vi, (vname, unit, vmax) in enumerate(zip(rmse_vars, rmse_units, rmse_vmaxes)):
    da_rmse = sk[vname].values
    cmap_r = plt.cm.YlOrRd
    norm_r = mcolors.Normalize(vmin=0, vmax=vmax)
    for i, wk in enumerate(weeks):
        ax = axes4[vi, i]
        d = da_rmse[wk - 1]
        title = week_labels[i].replace("\n", " ") if vi == 0 else ""
        im4 = make_map(ax, d, lons, lats, cmap_r, norm_r,
                       title,
                       letter=f"({letters[vi*6+i]})",
                       text=f"μ={india_mean(d):.2f}")
    cbar_ax4_row = fig4.add_axes([0.925, axes4[vi, 0].get_position().y0,
                                  0.010,
                                  axes4[vi, 0].get_position().height])
    cb4 = fig4.colorbar(im4, cax=cbar_ax4_row, orientation="vertical", extend="max")
    cb4.set_label(unit, fontsize=8, labelpad=4)
    cb4.ax.tick_params(labelsize=7)

    fig4.text(0.025, axes4[vi, 0].get_position().y0 +
              axes4[vi, 0].get_position().height / 2,
              var_labels[vi], ha="center", va="center",
              fontsize=10, fontweight="bold", rotation=90)

fig4.suptitle("Spire JFM 2026 S2S  |  RMSE Skill Maps (90 inits)  |  India domain",
              fontsize=11, fontweight="bold", y=0.97)
fig4.savefig(f"{OUTPUT_DIR}/fig4_rmse_skill_maps.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig4)
print("  → fig4_rmse_skill_maps.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — India-mean Summary Line Plot (ACC / RMSE / Bias vs lead week)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: India-mean summary …")

x = np.arange(1, 7)
xlabels = ["W1\n(d1-7)", "W2\n(d8-14)", "W3\n(d15-21)",
           "W4\n(d22-28)", "W5\n(d29-35)", "W6\n(d36-42)"]

def india_series(arr4d):
    return [float(np.nanmean(arr4d[w - 1])) for w in weeks]

acc_t2m_s    = india_series(sk["acc_t2m"].values)
acc_pr_s     = india_series(sk["acc_precip"].values)
acc_z5_s     = india_series(sk["acc_z500"].values)
rmse_t2m_s   = india_series(sk["rmse_t2m"].values)
rmse_pr_s    = india_series(sk["rmse_precip"].values)
rmse_z5_s    = india_series(sk["rmse_z500"].values)
bias_t2m_s   = india_series(sk["bias_t2m"].values)
bias_pr_s    = india_series(sk["bias_precip"].values)

fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
fig5.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.88,
                     wspace=0.30)

# ACC
ax = axes5[0]
ax.plot(x, acc_t2m_s, "o-", color="#D73027", lw=2, ms=7, label="T2m")
ax.plot(x, acc_pr_s,  "s-", color="#4393C3", lw=2, ms=7, label="Precip")
ax.plot(x, acc_z5_s,  "^-", color="#1A7E3B", lw=2, ms=7, label="Z500")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.7)
ax.set_ylim(-0.3, 1.0)
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("ACC", fontsize=10)
ax.set_title("(a) Anomaly Correlation Coefficient", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.fill_between(x, 0.5, 1.0, alpha=0.05, color="green", label="_nolegend_")
ax.text(6.1, 0.51, "skill\nthreshold\n(ACC=0.5)", fontsize=6.5, color="gray", va="bottom")

# RMSE
ax = axes5[1]
ax_twin = ax.twinx()
ax.plot(x, rmse_t2m_s, "o-", color="#D73027", lw=2, ms=7, label="T2m (K)")
ax_twin.plot(x, rmse_z5_s, "^-", color="#1A7E3B", lw=2, ms=7, label="Z500 (m)")
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("RMSE  T2m (K)", fontsize=10, color="#D73027")
ax_twin.set_ylabel("RMSE  Z500 (m)", fontsize=10, color="#1A7E3B")
ax.set_title("(b) RMSE", fontsize=10, fontweight="bold")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
ax.grid(True, alpha=0.3)

# Bias
ax = axes5[2]
spire_t2m_mean = [float(np.nanmean(ds["spire_t2m_anom"].mean("init_time").values[w-1])) for w in weeks]
era5_t2m_mean  = [float(np.nanmean(ds["era5_t2m_anom"].mean("init_time").values[w-1]))  for w in weeks]
ax.plot(x, spire_t2m_mean, "o-", color="#D73027", lw=2, ms=7, label="Spire T2m forecast")
ax.plot(x, era5_t2m_mean,  "s--", color="#333333", lw=2, ms=7, label="ERA5 T2m observed")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("India-mean anomaly (K)", fontsize=10)
ax.set_title("(c) India-mean T2m anomaly", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig5.suptitle("Spire JFM 2026 S2S  |  India-domain Skill Summary  |  90 init dates",
              fontsize=12, fontweight="bold", y=0.97)
fig5.savefig(f"{OUTPUT_DIR}/fig5_india_skill_summary.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig5)
print("  → fig5_india_skill_summary.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Precipitation Anomaly Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: Precip anomaly comparison …")

spire_pr = ds["spire_precip_anom"].mean("init_time").values
era5_pr  = ds["era5_precip_anom"].mean("init_time").values

cmap_precip = plt.cm.BrBG
norm_pr = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

fig6, axes6 = plt.subplots(2, 6, figsize=(18, 7),
                            subplot_kw={"projection": proj})
fig6.subplots_adjust(left=0.04, right=0.91, bottom=0.06, top=0.88,
                     hspace=0.10, wspace=0.06)

for i, wk in enumerate(weeks):
    wk_idx = wk - 1
    im6 = make_map(axes6[0, i], spire_pr[wk_idx], lons, lats,
                   cmap_precip, norm_pr,
                   week_labels[i].replace("\n", " "),
                   letter=f"({letters[i]})",
                   text=f"India={india_mean(spire_pr[wk_idx]):+.2f}")
    im6 = make_map(axes6[1, i], era5_pr[wk_idx], lons, lats,
                   cmap_precip, norm_pr, "",
                   letter=f"({letters[i+6]})",
                   text=f"India={india_mean(era5_pr[wk_idx]):+.2f}")

fig6.text(0.015, 0.73, "Spire precip forecast\n(vs ERA5 1991–2020 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold", rotation=90)
fig6.text(0.015, 0.30, "ERA5 precip observed\n(vs WB2 1990–2019 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold", rotation=90)

cbar_ax6 = fig6.add_axes([0.925, 0.06, 0.013, 0.82])
cb6 = fig6.colorbar(im6, cax=cbar_ax6, orientation="vertical",
                    extend="both", ticks=[-4, -2, 0, 2, 4])
cb6.set_label("Precip anomaly (mm/day)", fontsize=9, labelpad=5)
cb6.ax.tick_params(labelsize=8)

fig6.suptitle("Spire JFM 2026 S2S  |  Precipitation Anomaly (90-init mean)  |  Both vs climatology",
              fontsize=11, fontweight="bold", y=0.96)
fig6.savefig(f"{OUTPUT_DIR}/fig6_precip_anomaly_comparison.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig6)
print("  → fig6_precip_anomaly_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Z500 Anomaly Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: Z500 anomaly comparison …")

spire_z5 = ds["spire_z500_anom"].mean("init_time").values
era5_z5  = ds["era5_z500_anom"].mean("init_time").values

cmap_z5  = plt.cm.RdBu_r
norm_z5  = TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)

fig7, axes7 = plt.subplots(2, 6, figsize=(18, 7),
                            subplot_kw={"projection": proj})
fig7.subplots_adjust(left=0.04, right=0.91, bottom=0.06, top=0.88,
                     hspace=0.10, wspace=0.06)

for i, wk in enumerate(weeks):
    wk_idx = wk - 1
    im7 = make_map(axes7[0, i], spire_z5[wk_idx], lons, lats,
                   cmap_z5, norm_z5,
                   week_labels[i].replace("\n", " "),
                   letter=f"({letters[i]})",
                   text=f"India={india_mean(spire_z5[wk_idx]):+.1f}m")
    im7 = make_map(axes7[1, i], era5_z5[wk_idx], lons, lats,
                   cmap_z5, norm_z5, "",
                   letter=f"({letters[i+6]})",
                   text=f"India={india_mean(era5_z5[wk_idx]):+.1f}m")

fig7.text(0.015, 0.73, "Spire Z500 forecast\n(vs ERA5 1991–2020 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold", rotation=90)
fig7.text(0.015, 0.30, "ERA5 Z500 observed\n(vs WB2 1990–2019 climo)",
          ha="center", va="center", fontsize=9, fontweight="bold", rotation=90)

cbar_ax7 = fig7.add_axes([0.925, 0.06, 0.013, 0.82])
cb7 = fig7.colorbar(im7, cax=cbar_ax7, orientation="vertical",
                    extend="both", ticks=[-80, -40, 0, 40, 80])
cb7.set_label("Z500 anomaly (m)", fontsize=9, labelpad=5)
cb7.ax.tick_params(labelsize=8)

fig7.suptitle("Spire JFM 2026 S2S  |  Z500 Anomaly (90-init mean)  |  Both vs climatology",
              fontsize=11, fontweight="bold", y=0.96)
fig7.savefig(f"{OUTPUT_DIR}/fig7_z500_anomaly_comparison.png",
             dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig7)
print("  → fig7_z500_anomaly_comparison.png")

print("\nAll figures saved to:", OUTPUT_DIR)
