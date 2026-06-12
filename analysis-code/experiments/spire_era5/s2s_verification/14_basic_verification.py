"""
14_basic_verification.py — "Basic" model vs ground-truth verification.

Per Prof. Manmeet Singh's steer: do NOT compare anomalies. Compare the ABSOLUTE
fields — Spire model vs ERA5 ground truth — as (1) spatial maps + difference and
(2) scatter plots. Coastlines only (no political borders).

Computes weekly-mean ABSOLUTE temperature (°C) for each init date:
  Spire air_temperature_max  (daily max, native Spire product)
  Spire air_temperature      (daily mean)
  ERA5  daily-max  2m T
  ERA5  daily-mean 2m T
Weekly windows W1..W6 (d1-7 … d36-42), India domain 0-50N, 55-105E, 0.5°.

Outputs:
  weekly_absolute_v2.nc                      (the raw weekly means)
  ../../spire-s2s-paper/figures/fig21_basic_spatial_tmax.png   (Spire | ERA5 | diff)
  ../../spire-s2s-paper/figures/fig22_basic_scatter_tmax.png   (model vs truth scatter)
  ../../spire-s2s-paper/figures/fig23_basic_spatial_tmean.png
  ../../spire-s2s-paper/figures/fig24_basic_scatter_tmean.png
"""
import os, string, warnings
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 12, "savefig.dpi": 300, "figure.dpi": 130,
})

ABS_FILE = "weekly_absolute_v2.nc"
FIGD = "../../spire-s2s-paper/figures"
os.makedirs(FIGD, exist_ok=True)
PROJ = ccrs.PlateCarree(); EXT = [55, 105, 0, 50]
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 0.0, 50.0, 55.0, 105.0
WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, 35), 6: (36, 42)}
MAX_LEAD = 42
let = list(string.ascii_lowercase)

# ── BUILD or LOAD absolute weekly means ──────────────────────────────────────
if os.path.exists(ABS_FILE):
    print(f"Loading cached {ABS_FILE}")
    dsa = xr.open_dataset(ABS_FILE)
else:
    from arraylake import Client
    print("Opening Spire …")
    session = Client().get_repo("artamta/s2s-research").readonly_session("main")
    ds_mean = (xr.open_zarr(session.store, group="mean_stddev")
               .isel(latitude=slice(None, None, -1))
               .sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX)))
    spire_lat = ds_mean["latitude"].values; spire_lon = ds_mean["longitude"].values
    init_times = pd.DatetimeIndex(ds_mean["reference_time"].values)
    n = len(init_times)
    print(f"  {n} inits, {len(spire_lat)}×{len(spire_lon)}")

    ws = init_times[0] + pd.Timedelta(1, "D"); we = init_times[-1] + pd.Timedelta(MAX_LEAD, "D")
    all_dates = pd.date_range(ws, we, freq="D"); d2i = {d: i for i, d in enumerate(all_dates)}

    print("Opening ARCO-ERA5 & fetching daily T2m (max & mean) …")
    ds_e = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                        storage_options={"token": "anon"})
    da = ds_e["2m_temperature"].sel(latitude=slice(LAT_MAX+1, LAT_MIN-1),
                                    longitude=slice(LON_MIN-1, LON_MAX+1),
                                    time=slice(f"{ws.date()}T00:00", f"{we.date()}T23:00"))
    def daily(agg):
        d = (da.resample(time="1D").max("time") if agg == "max"
             else da.resample(time="1D").mean("time")).compute() - 273.15
        d = d.interp(latitude=spire_lat, longitude=spire_lon, method="linear").reindex(time=all_dates)
        return d.values.astype(np.float32)
    e5_max_d = daily("max"); e5_mean_d = daily("mean")
    print("  ERA5 daily fields ready.")

    print("Spire weekly means …")
    shp = (n, len(WEEKS), len(spire_lat), len(spire_lon))
    o = {k: np.full(shp, np.nan, np.float32) for k in
         ["spire_tmax", "era5_tmax", "spire_tmean", "era5_tmean"]}
    sp_mx, sp_mn = {}, {}
    for wk, (d0, d1) in WEEKS.items():
        steps = [np.timedelta64(d, "D") for d in range(d0, d1+1)]
        sp_mx[wk] = ds_mean["air_temperature_max"].sel(step=steps).mean("step").compute().values - 273.15
        sp_mn[wk] = ds_mean["air_temperature"].sel(step=steps).mean("step").compute().values - 273.15
    for i, idt in enumerate(init_times):
        for wi, (wk, (d0, d1)) in enumerate(WEEKS.items()):
            vd = pd.date_range(idt + pd.Timedelta(d0, "D"), idt + pd.Timedelta(d1, "D"))
            ix = [d2i[d] for d in vd]
            o["spire_tmax"][i, wi]  = sp_mx[wk][i]
            o["spire_tmean"][i, wi] = sp_mn[wk][i]
            o["era5_tmax"][i, wi]   = e5_max_d[ix].mean(0)
            o["era5_tmean"][i, wi]  = e5_mean_d[ix].mean(0)
    mk = lambda a, ln: xr.DataArray(a, dims=["init_time","week","latitude","longitude"],
            coords={"init_time": init_times, "week": list(WEEKS), "latitude": spire_lat,
                    "longitude": spire_lon}, attrs={"long_name": ln, "units": "degC"})
    dsa = xr.Dataset({"spire_tmax": mk(o["spire_tmax"], "Spire weekly-mean daily-max T2m"),
                      "era5_tmax":  mk(o["era5_tmax"],  "ERA5 weekly-mean daily-max T2m"),
                      "spire_tmean":mk(o["spire_tmean"],"Spire weekly-mean daily-mean T2m"),
                      "era5_tmean": mk(o["era5_tmean"], "ERA5 weekly-mean daily-mean T2m")},
                     attrs={"description": "Absolute weekly-mean T2m, Spire vs ERA5, JFM 2026 India"})
    dsa.to_netcdf(ABS_FILE); print(f"Saved {ABS_FILE}")

lats = dsa["latitude"].values; lons = dsa["longitude"].values
it = pd.DatetimeIndex(dsa["init_time"].values); N = len(it)
RNG = f"{it[0]:%Y-%m-%d} to {it[-1]:%Y-%m-%d}"
WL = ["W1 (d1–7)", "W2 (d8–14)", "W3 (d15–21)", "W4 (d22–28)", "W5 (d29–35)", "W6 (d36–42)"]

def cmap_map(ax, data, cmap, vmin, vmax, title="", letter=None, corner=None,
             ll=False, bl=False, norm=None):
    if norm is None:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=PROJ, shading="auto", rasterized=True)
    else:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                           transform=PROJ, shading="auto", rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, edgecolor="k", zorder=4)
    ax.set_extent(EXT, crs=PROJ)
    gl = ax.gridlines(lw=0.2, color="gray", ls=":", alpha=0.5)
    gl.xlocator = mticker.FixedLocator([60,70,80,90,100]); gl.ylocator = mticker.FixedLocator([10,20,30,40])
    gl.top_labels = gl.right_labels = False; gl.left_labels = ll; gl.bottom_labels = bl
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "0.35"}
    if title: ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    if letter: ax.text(0.03, 0.95, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
                       va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.9))
    if corner: ax.text(0.97, 0.05, corner, transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.9))
    return im

# ════════ Spatial: model | ground truth | difference (rows = W1/W3/W6) ════════
def fig_spatial(spv, e5v, label, fname):
    sp = dsa[spv].mean("init_time").values   # (week,lat,lon)
    e5 = dsa[e5v].mean("init_time").values
    rows = [0, 2, 5]; rlab = ["W1", "W3", "W6"]
    vmin = float(np.nanpercentile(np.concatenate([sp[rows], e5[rows]]), 2))
    vmax = float(np.nanpercentile(np.concatenate([sp[rows], e5[rows]]), 98))
    dmax = float(np.nanpercentile(np.abs(sp[rows]-e5[rows]), 98)); dmax = max(dmax, 1.0)
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), subplot_kw={"projection": PROJ})
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.06, top=0.9, hspace=0.12, wspace=0.08)
    for r, (wi, wl) in enumerate(zip(rows, rlab)):
        imL = cmap_map(axes[r,0], sp[wi], "turbo", vmin, vmax,
                       title=("Spire (model)" if r==0 else ""), letter=f"({let[r*3]})",
                       corner=f"μ={np.nanmean(sp[wi]):.1f}°C", ll=True, bl=(r==2))
        cmap_map(axes[r,1], e5[wi], "turbo", vmin, vmax,
                 title=("ERA5 (ground truth)" if r==0 else ""), letter=f"({let[r*3+1]})",
                 corner=f"μ={np.nanmean(e5[wi]):.1f}°C", bl=(r==2))
        imD = cmap_map(axes[r,2], sp[wi]-e5[wi], "RdBu_r", None, None, norm=dnorm,
                       title=("Spire − ERA5" if r==0 else ""), letter=f"({let[r*3+2]})",
                       corner=f"μ={np.nanmean(sp[wi]-e5[wi]):+.1f}°C", bl=(r==2))
        axes[r,0].text(-0.16, 0.5, wl, transform=axes[r,0].transAxes, rotation=90,
                       fontsize=12, fontweight="bold", va="center", ha="center")
    cax1 = fig.add_axes([0.915, 0.40, 0.013, 0.5]); cb1 = fig.colorbar(imL, cax=cax1)
    cb1.set_label(f"{label} (°C)", fontsize=10)
    cax2 = fig.add_axes([0.915, 0.06, 0.013, 0.28]); cb2 = fig.colorbar(imD, cax=cax2, extend="both")
    cb2.set_label("Difference (°C)", fontsize=10)
    fig.suptitle(f"Spire vs ERA5  |  {label}  |  90-init mean ({RNG})  |  coastline only",
                 fontsize=13, fontweight="bold", y=0.95)
    p = f"{FIGD}/{fname}"; fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  → {p}")

# ════════ Scatter: model vs ground truth (per week, gridcell) ════════
def fig_scatter(spv, e5v, label, fname):
    sp = dsa[spv].mean("init_time").values; e5 = dsa[e5v].mean("init_time").values
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.9, hspace=0.32, wspace=0.28)
    allv = np.concatenate([sp.ravel(), e5.ravel()])
    lo = float(np.nanpercentile(allv, 1)); hi = float(np.nanpercentile(allv, 99))
    for i, wk in enumerate(WEEKS):
        r, c = divmod(i, 3); ax = axes[r, c]; wi = wk-1
        x = e5[wi].ravel(); y = sp[wi].ravel()
        m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
        rr, _ = pearsonr(x, y); rmse = np.sqrt(np.mean((y-x)**2)); bias = np.mean(y-x); mae = np.mean(np.abs(y-x))
        ax.scatter(x, y, s=5, alpha=0.25, color="#C0392B", rasterized=True, linewidths=0)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6)
        coef = np.polyfit(x, y, 1); xf = np.linspace(lo, hi, 100)
        ax.plot(xf, np.polyval(coef, xf), "-", color="#2166AC", lw=1.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
        ax.set_xlabel("ERA5 ground truth (°C)", fontsize=9.5)
        ax.set_ylabel("Spire model (°C)", fontsize=9.5)
        ax.set_title(f"({let[i]}) {WL[i]}", fontsize=11, fontweight="bold")
        ax.text(0.04, 0.96, f"R² = {rr**2:.3f}\nRMSE = {rmse:.2f}°C\nMAE  = {mae:.2f}°C\n"
                f"bias = {bias:+.2f}°C\nslope = {coef[0]:.2f}",
                transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="0.55", alpha=0.95))
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Spire model vs ERA5 ground truth  |  {label}  |  grid-cell scatter, 90-init mean  |  {RNG}",
                 fontsize=13, fontweight="bold", y=0.97)
    p = f"{FIGD}/{fname}"; fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  → {p}")

print("Plotting basic verification (coastline only) …")
fig_spatial("spire_tmax", "era5_tmax", "Daily-max 2 m temperature", "fig21_basic_spatial_tmax.png")
fig_scatter("spire_tmax", "era5_tmax", "Daily-max 2 m temperature", "fig22_basic_scatter_tmax.png")
fig_spatial("spire_tmean", "era5_tmean", "Daily-mean 2 m temperature", "fig23_basic_spatial_tmean.png")
fig_scatter("spire_tmean", "era5_tmean", "Daily-mean 2 m temperature", "fig24_basic_scatter_tmean.png")
print("Done ✓")
