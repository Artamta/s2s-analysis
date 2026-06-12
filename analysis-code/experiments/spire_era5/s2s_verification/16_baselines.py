"""
16_baselines.py — Persistence & climatology baselines for the Spire verification.

Gives the trivial-forecast references every verification paper needs, so Spire's
skill is interpretable (does it beat persistence and climatology?).

  • CLIMATOLOGY forecast: anomaly ≡ 0.  → ACC undefined; RMSE = sqrt(mean(obs_anom²)).
    (Spire's anomaly RMSE should be BELOW this line to have skill.)
  • PERSISTENCE forecast: the observed anomaly in the 7 days ENDING at init,
    carried forward to all leads (standard S2S persistence). → ACC decays with lead.

Compares against Spire on the SAME 90 inits, for T2m-mean and T2m-max, using
weekly_anomalies_v2.nc (Spire & ERA5 anomalies) + cached ERA5 DOY climatologies.

Outputs:
  ../../spire-s2s-paper/figures/fig27_baseline_acc.png   (Spire vs persistence ACC)
  ../../spire-s2s-paper/figures/fig28_baseline_rmse.png  (Spire vs persistence vs climatology RMSE)
"""
import os, warnings
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "savefig.dpi": 300, "figure.dpi": 130})

FIGD = "../../spire-s2s-paper/figures"; os.makedirs(FIGD, exist_ok=True)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 0.0, 50.0, 55.0, 105.0
WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, 35), 6: (36, 42)}

# ── Spire & ERA5 verifying anomalies (already computed) ──────────────────────
ds = xr.open_dataset("weekly_anomalies_v2.nc")
lats, lons = ds.latitude.values, ds.longitude.values
it = pd.DatetimeIndex(ds.init_time.values); N = len(it)
nW = len(WEEKS)

# ── Cached ERA5 DOY climatologies (mean & max) ───────────────────────────────
clim_mean = xr.open_dataset("era5_tmean_climo_india.nc")["tmean_climo"]   # (doy,lat,lon)
clim_max  = xr.open_dataset("era5_tmax_climo_india.nc")["tmax_climo"]
def climo_doy(clim, doys):
    # Climo DOY index covers ~Jan–mid-May. The persistence pre-init week of early-Jan
    # inits reaches into late Dec (doy>360); map those (and any gaps) to the nearest
    # available DOY — late Dec ≈ earliest winter DOY climatologically.
    avail = clim.doy.values; aset = set(int(a) for a in avail)
    vals = []
    for d in doys:
        d = int(d)
        if d not in aset:
            d = int(avail.min()) if d > 200 else int(avail[np.argmin(np.abs(avail - d))])
        vals.append(clim.sel(doy=d).values)
    return np.mean(vals, axis=0)

# ── ERA5 pre-init observed week → persistence anomaly ────────────────────────
print("Fetching ERA5 for pre-init persistence week …")
de = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                  storage_options={"token": "anon"})
pre_start = it[0] - pd.Timedelta(7, "D"); pre_end = it[-1]
pdates = pd.date_range(pre_start, pre_end, freq="D"); p2i = {d: i for i, d in enumerate(pdates)}
def daily(agg):
    da = de["2m_temperature"].sel(latitude=slice(LAT_MAX+1, LAT_MIN-1),
                                  longitude=slice(LON_MIN-1, LON_MAX+1),
                                  time=slice(f"{pre_start.date()}T00:00", f"{pre_end.date()}T23:00"))
    d = (da.resample(time="1D").max("time") if agg == "max" else da.resample(time="1D").mean("time"))
    return ((d.compute() - 273.15).interp(latitude=lats, longitude=lons, method="linear")
            .reindex(time=pdates).values.astype(np.float32))
era5_mean_pre = daily("mean"); era5_max_pre = daily("max")

def persistence_anom(pre_daily, clim):
    """Observed anomaly over the 7 days ending at each init (one field per init)."""
    out = np.full((N, len(lats), len(lons)), np.nan, np.float32)
    for i, idt in enumerate(it):
        wd = pd.date_range(idt - pd.Timedelta(6, "D"), idt)
        obs = pre_daily[[p2i[d] for d in wd if d in p2i]].mean(0)
        out[i] = obs - climo_doy(clim, [d.day_of_year for d in wd])
    return out
PERS = {"mean": persistence_anom(era5_mean_pre, clim_mean),
        "max":  persistence_anom(era5_max_pre,  clim_max)}

# ── Metrics: ACC (per-gridpoint corr across inits, area-mean) & RMSE ──────────
def acc_curve(fc, ob):
    """fc: (init,lat,lon) constant-with-lead OR (init,week,lat,lon); ob:(init,week,lat,lon)"""
    acc = np.full(nW, np.nan); rmse = np.full(nW, np.nan)
    for w in range(nW):
        f = fc[:, w] if fc.ndim == 4 else fc          # (init,lat,lon)
        o = ob[:, w]
        ff = f.reshape(N, -1); oo = o.reshape(N, -1)
        rs = [pearsonr(ff[:, p], oo[:, p])[0] for p in range(ff.shape[1])
              if np.std(ff[:, p]) > 1e-6 and np.std(oo[:, p]) > 1e-6]
        acc[w] = np.nanmean(rs) if rs else np.nan
        rmse[w] = float(np.sqrt(np.nanmean((ff - oo) ** 2)))
    return acc, rmse

curves = {}
for var, short in [("t2m_mean", "mean"), ("t2m_max", "max")]:
    sp = ds[f"spire_{var}_anom"].values; e5 = ds[f"era5_{var}_anom"].values
    acc_sp, rmse_sp = acc_curve(sp, e5)
    acc_pe, rmse_pe = acc_curve(PERS[short], e5)
    rmse_cl = np.array([float(np.sqrt(np.nanmean(e5[:, w] ** 2))) for w in range(nW)])  # climo=0
    curves[short] = dict(acc_sp=acc_sp, rmse_sp=rmse_sp, acc_pe=acc_pe, rmse_pe=rmse_pe, rmse_cl=rmse_cl)
    print(f"{short}: ACC Spire {acc_sp.round(2)}  Persist {acc_pe.round(2)}")
    print(f"{short}: RMSE Spire {rmse_sp.round(2)}  Persist {rmse_pe.round(2)}  Climo {rmse_cl.round(2)}")

x = np.arange(1, nW + 1); WL = [f"W{w}" for w in WEEKS]
SP_C = {"mean": "#D73027", "max": "#F46D43"}

# ── fig27: ACC — Spire vs persistence ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.2))
for short, lab in [("mean", "T2m-mean"), ("max", "T2m-max")]:
    c = SP_C[short]
    ax.plot(x, curves[short]["acc_sp"], "o-", color=c, lw=2.2, ms=8, label=f"Spire {lab}", mec="white", mew=0.5)
    ax.plot(x, curves[short]["acc_pe"], "s--", color=c, lw=1.6, ms=6, alpha=0.7, label=f"Persistence {lab}")
ax.axhline(0.5, color="gray", ls="-.", lw=1, alpha=0.7); ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax.set_xlim(0.5, nW+0.5); ax.set_ylim(-0.3, 1.0); ax.set_xticks(x); ax.set_xticklabels(WL)
ax.set_xlabel("Forecast lead week"); ax.set_ylabel("India-mean ACC (r)")
ax.set_title("Spire vs Persistence — Anomaly Correlation\nIndia, JFM 2026 (Spire should beat persistence)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIGD}/fig27_baseline_acc.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"→ {FIGD}/fig27_baseline_acc.png")

# ── fig28: RMSE — Spire vs persistence vs climatology ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.2))
for short, lab in [("mean", "T2m-mean"), ("max", "T2m-max")]:
    c = SP_C[short]
    ax.plot(x, curves[short]["rmse_sp"], "o-", color=c, lw=2.2, ms=8, label=f"Spire {lab}", mec="white", mew=0.5)
    ax.plot(x, curves[short]["rmse_pe"], "s--", color=c, lw=1.6, ms=6, alpha=0.7, label=f"Persistence {lab}")
    ax.plot(x, curves[short]["rmse_cl"], ":", color=c, lw=1.6, alpha=0.6, label=f"Climatology {lab}")
ax.set_xlim(0.5, nW+0.5); ax.set_xticks(x); ax.set_xticklabels(WL)
ax.set_xlabel("Forecast lead week"); ax.set_ylabel("RMSE (K)")
ax.set_title("Spire vs Persistence vs Climatology — RMSE\nIndia, JFM 2026 (Spire should be lowest)", fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIGD}/fig28_baseline_rmse.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"→ {FIGD}/fig28_baseline_rmse.png")
print("Done ✓")
