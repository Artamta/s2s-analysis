"""
02_plot_daily.py

Spire S2S forecast vs ERA5 observed — publication-quality figures.
Init: 2026-01-01, lead 1–46 days (valid Jan 2 – Feb 16 2026)

Figures:
  fig1_india_line.png        — India-mean daily T2m line + uncertainty band
  fig2_weekly_maps_india.png — Weekly-mean spatial: Spire | ERA5 | Error (W1–W6, India)
  fig3_weekly_maps_global.png— Same, global (W1–W4)
  fig4_scatter_india.png     — Scatter Spire vs ERA5 per lead day (India), R²/MAE/RMSE + reg line
  fig5_scatter_global.png    — Same, global
  fig6_acc_india.png         — ACC, RMSE, MAE vs lead week (from 90-init weekly anomalies)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, Normalize
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from scipy.stats import pearsonr, linregress

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

DPI = 300
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
INIT_DATE = "2026-01-01"

# Fixed colorbar limits (as requested)
T2M_VMIN, T2M_VMAX = -30.0, 50.0     # T2m absolute range
BIAS_LIM = 8.0                         # symmetric bias range ±8 K

# ── India-only coastline (no disputed J&K line) ───────────────────────────────
def india_coastline_feature():
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_0_countries")
    india_geom = None
    for rec in shpreader.Reader(shp).records():
        if rec.attributes["ADMIN"] == "India":
            india_geom = rec.geometry
            break
    if india_geom is None:
        return cfeature.COASTLINE.with_scale("50m")
    return cfeature.ShapelyFeature([india_geom], ccrs.PlateCarree(),
                                   facecolor="none", edgecolor="#1a1a1a",
                                   linewidth=1.1)

def india_states_feature():
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_1_states_provinces")
    india_states = []
    for rec in shpreader.Reader(shp).records():
        if rec.attributes.get("admin", "") == "India":
            india_states.append(rec.geometry)
    if not india_states:
        return None
    return cfeature.ShapelyFeature(india_states, ccrs.PlateCarree(),
                                   facecolor="none", edgecolor="#555555",
                                   linewidth=0.35)

INDIA_FEATURE   = india_coastline_feature()
INDIA_STATES_FT = india_states_feature()
NEIGHBORS_FEATURE = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_0_countries",
    scale="50m", facecolor="none", edgecolor="#888888", linewidth=0.45)

def add_india_map(ax, extent=None):
    # ocean background
    ax.set_facecolor("#d0e8f5")
    # land background (light gray) — neighbor countries outline
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0ede8",
                   zorder=0)
    ax.add_feature(NEIGHBORS_FEATURE, zorder=2)
    if INDIA_STATES_FT is not None:
        ax.add_feature(INDIA_STATES_FT, zorder=3)
    ax.add_feature(INDIA_FEATURE, zorder=4)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

def add_global_map(ax):
    ax.set_facecolor("#d0e8f5")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f0ede8",
                   zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.5, edgecolor="#333333", zorder=4)
    ax.set_global()

def gridlines(ax, left=True, bottom=True):
    gl = ax.gridlines(linewidth=0.2, color="#aaaaaa", linestyle=":",
                      alpha=0.6, zorder=3, draw_labels=False)
    if left or bottom:
        gl2 = ax.gridlines(linewidth=0.2, color="#aaaaaa", linestyle=":",
                           alpha=0.6, zorder=3, draw_labels=True)
        gl2.top_labels = gl2.right_labels = False
        gl2.left_labels   = left
        gl2.bottom_labels = bottom
        gl2.xlabel_style  = {"size": 6.5, "color": "#333333"}
        gl2.ylabel_style  = {"size": 6.5, "color": "#333333"}
        return gl2
    return gl

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
sp_india   = xr.open_zarr("spire_forecast_jan1_india.zarr",  group="mean_stddev")
sp_pctl_i  = xr.open_zarr("spire_forecast_jan1_india.zarr",  group="percentiles")
e5_india   = xr.open_zarr("era5_observed_jan2_feb16_india.zarr")
sp_global  = xr.open_zarr("spire_forecast_jan1_global.zarr", group="mean_stddev")
e5_global  = xr.open_zarr("era5_observed_jan2_feb16_global.zarr")

# Compute valid-date index from init + step
init_ts     = pd.Timestamp(INIT_DATE)
valid_dates = pd.DatetimeIndex(
    [init_ts + pd.Timedelta(s) for s in sp_india["step"].values]
)
lead_days = np.arange(1, len(valid_dates) + 1)

# weekly-mean slices (days 1-7, 8-14, …)
WEEKS = {1:(0,7), 2:(7,14), 3:(14,21), 4:(21,28), 5:(28,35), 6:(35,42)}
WLABELS = {1:"W1 (d1–7)", 2:"W2 (d8–14)", 3:"W3 (d15–21)",
           4:"W4 (d22–28)", 5:"W5 (d29–35)", 6:"W6 (d36–42)"}

def week_mean(da, wk, time_dim="step"):
    s, e = WEEKS[wk]
    return da.isel({time_dim: slice(s, e)}).mean(time_dim)

# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(sp, e5):
    mask = ~(np.isnan(sp) | np.isnan(e5))
    a, b = sp[mask].astype(float), e5[mask].astype(float)
    if len(a) < 5:
        return np.nan, np.nan, np.nan
    r, _  = pearsonr(a, b)
    mae   = np.mean(np.abs(a - b))
    rmse  = np.sqrt(np.mean((a - b) ** 2))
    return r**2, mae, rmse

# ── area-weighted mean helpers ────────────────────────────────────────────────
def india_mean_series(da):
    lat = da["latitude"].values
    w   = np.cos(np.deg2rad(lat))
    w  /= w.mean()
    return (da * xr.DataArray(w, dims="latitude")).mean(["latitude","longitude"]).values

def india_mean_1d(da):
    lat = da["latitude"].values
    w   = np.cos(np.deg2rad(lat))
    w  /= w.mean()
    return float((da * xr.DataArray(w, dims="latitude")).mean(["latitude","longitude"]))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — India-mean daily T2m line plot
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: India-mean line plot …")

sp_m   = india_mean_series(sp_india["t2m_mean"])
e5_m   = india_mean_series(e5_india["t2m"])
sp_p10 = india_mean_series(sp_pctl_i["t2m_pctl"].sel(percentile=10))
sp_p90 = india_mean_series(sp_pctl_i["t2m_pctl"].sel(percentile=90))
sp_p20 = india_mean_series(sp_pctl_i["t2m_pctl"].sel(percentile=20))
sp_p80 = india_mean_series(sp_pctl_i["t2m_pctl"].sel(percentile=80))
sp_p50 = india_mean_series(sp_pctl_i["t2m_pctl"].sel(percentile=50))

r2, mae, rmse = metrics(sp_m, e5_m)
bias = sp_m - e5_m

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
fig.subplots_adjust(hspace=0.06)

# uncertainty bands
ax.fill_between(lead_days, sp_p10, sp_p90, alpha=0.15, color="#2166AC",
                label="Spire p10–p90 range")
ax.fill_between(lead_days, sp_p20, sp_p80, alpha=0.25, color="#2166AC",
                label="Spire p20–p80 range")
ax.plot(lead_days, sp_p50, lw=1.0, color="#2166AC", ls="-", alpha=0.5)
ax.plot(lead_days, sp_m,   lw=2.2, color="#2166AC", label="Spire ensemble mean")
ax.plot(lead_days, e5_m,   lw=2.2, color="#D73027", ls="--", label="ERA5 observed")

# week shading
for wk, (s, e) in WEEKS.items():
    if wk % 2 == 0:
        ax.axvspan(s+1, e, alpha=0.06, color="gray")
    ax.axvline(e, color="gray", lw=0.7, ls=":", alpha=0.6)
    ax.text((s+e)/2 + 1, ax.get_ylim()[1] if not ax.get_ylim()[1] else 17.5,
            f"W{wk}", fontsize=8, ha="center", color="0.5")

ax.set_ylabel("India-mean T2m (°C)", fontsize=11)
ax.legend(fontsize=9, ncol=4, loc="lower right",
          framealpha=0.9, edgecolor="0.7")
ax.grid(True, alpha=0.3, lw=0.5)
ax.set_title(
    f"Spire S2S Forecast vs ERA5 Observed  —  India-mean Daily T2m\n"
    f"Init: {INIT_DATE}   |   R²={r2:.3f}   MAE={mae:.2f}°C   RMSE={rmse:.2f}°C",
    fontsize=12, fontweight="bold", pad=8)

# bias panel
ax2.bar(lead_days, bias, color=np.where(bias >= 0, "#D73027", "#2166AC"),
        alpha=0.75, width=0.85)
ax2.axhline(0, color="k", lw=0.8)
ax2.axhline(np.nanmean(bias), color="#D73027", lw=1.2, ls="--",
            label=f"Mean bias = {np.nanmean(bias):+.2f}°C")
ax2.set_ylabel("Bias (°C)", fontsize=10)
ax2.set_xlabel(f"Lead day  (Init: {INIT_DATE})", fontsize=10)
ax2.legend(fontsize=8.5, loc="upper right")
ax2.grid(True, alpha=0.25, lw=0.5)
ax2.set_xticks(lead_days[::2])
ax2.set_xticklabels(
    [f"d{d}\n{valid_dates[d-1].strftime('%b%d')}" for d in lead_days[::2]],
    fontsize=7.5)

fig.savefig(f"{OUT}/fig1_india_line.png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"  → fig1_india_line.png  R²={r2:.3f} MAE={mae:.2f} RMSE={rmse:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Helper: weekly spatial figure (Spire | ERA5 | Error)
# ══════════════════════════════════════════════════════════════════════════════
PANEL_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

def weekly_spatial_fig(sp_da, e5_da, weeks_list, extent, add_map_fn,
                       fname, suptitle, valid_dates,
                       time_dim_sp="step", time_dim_e5="time",
                       t_vmin=T2M_VMIN, t_vmax=T2M_VMAX,
                       bias_lim=BIAS_LIM):

    n = len(weeks_list)
    proj = ccrs.PlateCarree()

    # tighter aspect for India (portrait), wider for global
    is_india = extent is not None
    col_w = 3.2 if is_india else 4.2
    fig, axes = plt.subplots(3, n,
                             figsize=(col_w * n + 1.2, 9.0),
                             subplot_kw={"projection": proj},
                             gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    fig.subplots_adjust(left=0.08, right=0.87, top=0.91, bottom=0.02)

    # gather data
    sp_maps   = [week_mean(sp_da, wk, time_dim_sp).values for wk in weeks_list]
    e5_maps   = [week_mean(e5_da, wk, time_dim_e5).values for wk in weeks_list]
    bias_maps = [sp_maps[j] - e5_maps[j] for j in range(n)]

    lats = sp_da["latitude"].values
    lons = sp_da["longitude"].values

    norm_t    = Normalize(vmin=t_vmin, vmax=t_vmax)
    norm_bias = TwoSlopeNorm(vmin=-bias_lim, vcenter=0, vmax=bias_lim)

    cmap_t    = plt.cm.RdYlBu_r
    cmap_bias = plt.cm.RdBu_r

    row_labels = ["Spire\nforecast", "ERA5\nobserved", "Error\n(Spire−ERA5)"]

    im_t = im_b = None
    panel_idx = 0
    for row in range(3):
        for j, wk in enumerate(weeks_list):
            ax = axes[row, j]
            s, e = WEEKS[wk]
            wk_dates = valid_dates[s:e]
            date_rng = (f"{wk_dates[0].strftime('%b %d')}–"
                        f"{wk_dates[-1].strftime('%b %d')}")

            if row == 0:
                data, cmap, norm = sp_maps[j], cmap_t, norm_t
            elif row == 1:
                data, cmap, norm = e5_maps[j], cmap_t, norm_t
            else:
                data, cmap, norm = bias_maps[j], cmap_bias, norm_bias

            im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                               transform=proj, rasterized=True, shading="auto")
            if row < 2:
                im_t = im
            else:
                im_b = im

            add_map_fn(ax)
            show_left   = (j == 0)
            show_bottom = (row == 2)
            gridlines(ax, left=show_left, bottom=show_bottom)

            # column header (only row 0)
            if row == 0:
                ax.set_title(f"{WLABELS[wk]}\n({date_rng})",
                             fontsize=8.5, fontweight="bold", pad=4,
                             color="#111111")

            # row label (only col 0)
            if j == 0:
                ax.text(-0.22, 0.5, row_labels[row],
                        transform=ax.transAxes, fontsize=8.5,
                        fontweight="bold", rotation=90,
                        va="center", ha="center", color="#222222")

            # panel letter top-left
            letter = PANEL_LETTERS[panel_idx]
            ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
                    fontsize=7.5, fontweight="bold", va="top", ha="left",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

            # mean annotation bottom-right
            mu = float(np.nanmean(data))
            unit = "K" if row == 2 else "°C"
            ax.text(0.97, 0.03, f"μ={mu:+.2f} {unit}",
                    transform=ax.transAxes, fontsize=6.8,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.90, linewidth=0.6))
            panel_idx += 1

    # ── colorbars ────────────────────────────────────────────────────────────
    # T2m colorbar (rows 0-1)
    if im_t is not None:
        cax1 = fig.add_axes([0.882, 0.36, 0.013, 0.53])
        ticks_t = np.arange(-30, 55, 10)
        cb1 = fig.colorbar(im_t, cax=cax1, norm=norm_t,
                           extend="both", ticks=ticks_t)
        cb1.set_label("T2m (°C)", fontsize=8.5, labelpad=4)
        cb1.ax.tick_params(labelsize=7.5)

    # Bias colorbar (row 2)
    if im_b is not None:
        cax2 = fig.add_axes([0.882, 0.02, 0.013, 0.26])
        ticks_b = np.arange(-bias_lim, bias_lim + 1, 2)
        cb2 = fig.colorbar(im_b, cax=cax2, norm=norm_bias,
                           extend="both", ticks=ticks_b)
        cb2.set_label("Bias (K)", fontsize=8.5, labelpad=4)
        cb2.ax.tick_params(labelsize=7.5)

    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.975,
                 color="#111111")
    fig.savefig(f"{OUT}/{fname}", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Weekly maps, India
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Weekly spatial maps, India …")
INDIA_EXTENT = [65, 100, 5, 40]

weekly_spatial_fig(
    sp_da=sp_india["t2m_mean"], e5_da=e5_india["t2m"],
    weeks_list=[1, 2, 3, 4, 5, 6],
    extent=INDIA_EXTENT,
    add_map_fn=lambda ax: add_india_map(ax, extent=INDIA_EXTENT),
    fname="fig2_weekly_maps_india.png",
    suptitle=(f"Spire vs ERA5  —  Weekly-mean T2m Spatial Maps  (India)\n"
              f"Init: {INIT_DATE}  |  Row 1: Spire  |  Row 2: ERA5  |  Row 3: Spire−ERA5 error"),
    valid_dates=valid_dates,
)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Weekly maps, Global
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Weekly spatial maps, Global …")
weekly_spatial_fig(
    sp_da=sp_global["t2m_mean"], e5_da=e5_global["t2m"],
    weeks_list=[1, 2, 3, 4],
    extent=None,
    add_map_fn=add_global_map,
    fname="fig3_weekly_maps_global.png",
    suptitle=(f"Spire vs ERA5  —  Weekly-mean T2m Spatial Maps  (Global)\n"
              f"Init: {INIT_DATE}  |  Row 1: Spire  |  Row 2: ERA5  |  Row 3: Spire−ERA5 error"),
    valid_dates=valid_dates,
)

# ══════════════════════════════════════════════════════════════════════════════
# Scatter helper — one panel per selected lead day
# ══════════════════════════════════════════════════════════════════════════════
def scatter_fig(sp_da, e5_da, label_days, valid_dates, fname, suptitle):
    n = len(label_days)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.8 * nrows))
    axes_flat  = np.array(axes).flatten()
    fig.subplots_adjust(hspace=0.42, wspace=0.32,
                        left=0.06, right=0.97, top=0.90, bottom=0.07)

    for k, d in enumerate(label_days):
        ax  = axes_flat[k]
        sp  = sp_da.isel(step=d-1).values.ravel().astype(float)
        e5  = e5_da.isel(time=d-1).values.ravel().astype(float)
        mask = ~(np.isnan(sp) | np.isnan(e5))
        sp, e5 = sp[mask], e5[mask]

        r2, mae, rmse = metrics(sp, e5)
        r = np.sqrt(r2) if not np.isnan(r2) else np.nan

        # hexbin density scatter
        hb = ax.hexbin(e5, sp, gridsize=60, cmap="plasma",
                       mincnt=1, linewidths=0.0,
                       bins="log")
        cb = plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("log₁₀(count)", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)

        # 1:1 line
        lmin = min(e5.min(), sp.min()) - 1
        lmax = max(e5.max(), sp.max()) + 1
        ax.plot([lmin, lmax], [lmin, lmax], "k-", lw=1.3, alpha=0.7,
                label="1:1 line", zorder=5)

        # regression line
        slope, intercept, *_ = linregress(e5, sp)
        x_reg = np.array([lmin, lmax])
        ax.plot(x_reg, slope * x_reg + intercept, "--", color="#F4A582",
                lw=1.5, label=f"Reg. (slope={slope:.2f})", zorder=5)

        ax.set_xlim(lmin, lmax)
        ax.set_ylim(lmin, lmax)
        ax.set_aspect("equal")

        date_str = valid_dates[d-1].strftime("%b %d")
        ax.set_title(f"Lead day {d}  ({date_str})", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("ERA5 observed (°C)", fontsize=9)
        ax.set_ylabel("Spire forecast (°C)", fontsize=9)

        # metrics box top-left
        txt = (f"R      = {r:.3f}\n"
               f"R²     = {r2:.3f}\n"
               f"MAE  = {mae:.2f} °C\n"
               f"RMSE = {rmse:.2f} °C")
        ax.text(0.04, 0.97, txt, transform=ax.transAxes,
                fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.45", fc="white",
                          ec="0.5", alpha=0.92))
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
        ax.tick_params(labelsize=8)

    for k in range(len(label_days), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.97)
    fig.savefig(f"{OUT}/{fname}", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Scatter, India
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Scatter plots, India …")
scatter_fig(
    sp_da=sp_india["t2m_mean"], e5_da=e5_india["t2m"],
    label_days=[1, 3, 7, 10, 14, 21, 28, 35, 42],
    valid_dates=valid_dates,
    fname="fig4_scatter_india.png",
    suptitle=(f"Spire vs ERA5  —  T2m Scatter (India: 5–40°N, 65–100°E)\n"
              f"Init: {INIT_DATE}  |  Each point = one 0.5° grid cell"),
)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Scatter, Global
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Scatter plots, Global …")
scatter_fig(
    sp_da=sp_global["t2m_mean"], e5_da=e5_global["t2m"],
    label_days=[1, 7, 14, 21, 28, 42],
    valid_dates=valid_dates,
    fname="fig5_scatter_global.png",
    suptitle=(f"Spire vs ERA5  —  T2m Scatter (Global)\n"
              f"Init: {INIT_DATE}  |  Each point = one 0.5° grid cell"),
)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — ACC / RMSE / MAE skill vs lead week
# Uses 90-init weekly anomalies (weekly_anomalies_v2.nc) for a proper
# cross-init ACC: r(spire_anom, era5_anom) across 90 init dates per grid cell,
# then spatial mean over India.
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: ACC / RMSE / MAE skill plot …")

ds_wk = xr.open_dataset("../s2s_verification/weekly_anomalies_v2.nc")
# Crop to India domain
ds_wk = ds_wk.sel(
    latitude=slice(5.0, 40.0),
    longitude=slice(65.0, 100.0),
)

weeks_list = [1, 2, 3, 4, 5, 6]
acc_mean   = np.full(6, np.nan)
acc_max    = np.full(6, np.nan)
rmse_mean  = np.full(6, np.nan)
rmse_max   = np.full(6, np.nan)
mae_mean   = np.full(6, np.nan)
mae_max    = np.full(6, np.nan)

for wi, wk in enumerate(weeks_list):
    # (init, lat, lon) arrays
    sp_m = ds_wk["spire_t2m_mean_anom"].sel(week=wk).values  # (90, lat, lon)
    e5_m = ds_wk["era5_t2m_mean_anom"].sel(week=wk).values
    sp_x = ds_wk["spire_t2m_max_anom"].sel(week=wk).values
    e5_x = ds_wk["era5_t2m_max_anom"].sel(week=wk).values

    for var_sp, var_e5, acc_arr, rmse_arr, mae_arr in [
        (sp_m, e5_m, acc_mean, rmse_mean, mae_mean),
        (sp_x, e5_x, acc_max,  rmse_max,  mae_max),
    ]:
        n_init, nlat, nlon = var_sp.shape
        npix = nlat * nlon
        sp_2d = var_sp.reshape(n_init, npix)
        e5_2d = var_e5.reshape(n_init, npix)

        # ACC: Pearson r across 90 inits at each pixel, then spatial mean
        rs = []
        for p in range(npix):
            a, b = sp_2d[:, p], e5_2d[:, p]
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                rs.append(pearsonr(a, b)[0])
        acc_arr[wi] = np.nanmean(rs) if rs else np.nan

        diff = (var_sp - var_e5).reshape(n_init, npix)
        rmse_arr[wi] = np.sqrt(np.nanmean(diff ** 2))
        mae_arr[wi]  = np.nanmean(np.abs(diff))

x = np.arange(1, 7)
wlabels_x = [f"W{w}\nd{(w-1)*7+1}–{w*7}" for w in range(1, 7)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.13, top=0.85, wspace=0.28)

col_mean = "#D73027"
col_max  = "#2166AC"

# (a) ACC
ax = axes[0]
ax.axhline(0.6, color="gray", lw=0.8, ls="--", alpha=0.6)
ax.axhline(0.5, color="gray", lw=0.8, ls=":",  alpha=0.5)
ax.fill_between([0.5, 6.5], 0.5, 1.0, color="green", alpha=0.06)
ax.text(6.35, 0.51, "Skillful\n(ACC≥0.5)", fontsize=7, color="0.4",
        va="bottom", ha="right")
ax.plot(x, acc_mean, "o-", color=col_mean, lw=2, ms=7, label="Mean T2m")
ax.plot(x, acc_max,  "s-", color=col_max,  lw=2, ms=7, label="Max T2m")
ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
ax.set_xlim(0.5, 6.5); ax.set_ylim(-0.15, 1.0)
ax.set_xticks(x); ax.set_xticklabels(wlabels_x, fontsize=8)
ax.set_ylabel("ACC (Pearson r)", fontsize=10)
ax.set_title("(a) Anomaly Correlation Coefficient", fontsize=10.5,
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (b) RMSE
ax = axes[1]
ax.plot(x, rmse_mean, "o-", color=col_mean, lw=2, ms=7, label="Mean T2m")
ax.plot(x, rmse_max,  "s-", color=col_max,  lw=2, ms=7, label="Max T2m")
ax.set_xlim(0.5, 6.5); ax.set_ylim(0)
ax.set_xticks(x); ax.set_xticklabels(wlabels_x, fontsize=8)
ax.set_ylabel("RMSE (K)", fontsize=10)
ax.set_title("(b) Root Mean Square Error", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (c) MAE
ax = axes[2]
ax.plot(x, mae_mean, "o-", color=col_mean, lw=2, ms=7, label="Mean T2m")
ax.plot(x, mae_max,  "s-", color=col_max,  lw=2, ms=7, label="Max T2m")
ax.set_xlim(0.5, 6.5); ax.set_ylim(0)
ax.set_xticks(x); ax.set_xticklabels(wlabels_x, fontsize=8)
ax.set_ylabel("MAE (K)", fontsize=10)
ax.set_title("(c) Mean Absolute Error", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Spire S2S Forecast Skill  —  India Domain (5–40°N, 65–100°E)\n"
    f"ACC / RMSE / MAE vs lead week  |  90 init dates  |  T2m anomalies vs ERA5",
    fontsize=11.5, fontweight="bold", y=0.97)

fig.savefig(f"{OUT}/fig6_acc_skill.png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("  → fig6_acc_skill.png")
print(f"\n  ACC mean-T2m: {acc_mean.round(3)}")
print(f"  ACC  max-T2m: {acc_max.round(3)}")

print(f"\nAll figures saved to ./{OUT}/  (DPI={DPI})")
