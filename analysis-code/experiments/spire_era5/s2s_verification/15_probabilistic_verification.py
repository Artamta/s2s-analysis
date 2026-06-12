"""
15_probabilistic_verification.py — Probabilistic S2S verification of Spire.

Adds the verification S2S referees expect: tercile RPS / RPSS (ranked probability
skill score vs a climatological forecast) and reliability diagrams, using Spire's
native `probabilities` group (forecast tercile probabilities) verified against
ERA5 tercile categories (1991-2020 thresholds).

  • RPSS = 1 − RPS_forecast / RPS_climatology.  RPSS > 0 ⇒ Spire beats a
    climatological (1/3,1/3,1/3) forecast → the CLIMATOLOGY BASELINE is built in.
  • Reliability diagram (upper tercile): forecast probability vs observed frequency.

Variables: air_temperature (mean T2m), air_temperature_max.  Weekly windows W1..W6.
Domain: India 0-50N, 55-105E, 0.5°.

Heavy step (cached): ERA5 1991-2020 daily-mean & daily-max T2m over the Jan–mid-May
window, used to derive per-window tercile thresholds.  Cache: era5_30yr_daily_india.nc

Outputs:
  prob_scores.nc
  ../../spire-s2s-paper/figures/fig25_rpss_vs_lead.png
  ../../spire-s2s-paper/figures/fig26_reliability_tercile.png
"""
import os, warnings
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "savefig.dpi": 300, "figure.dpi": 130})

FIGD = "../../spire-s2s-paper/figures"; os.makedirs(FIGD, exist_ok=True)
CACHE = "era5_30yr_daily_india.nc"
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 0.0, 50.0, 55.0, 105.0
WEEKS = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, 35), 6: (36, 42)}
CLIMO_YEARS = range(1991, 2021)
VARS = {"air_temperature": ("tmean", "mean"), "air_temperature_max": ("tmax", "max")}

# ── Spire forecast tercile probabilities (weekly mean of daily probs) ─────────
from arraylake import Client
print("Opening Spire probabilities …")
sess = Client().get_repo("artamta/s2s-research").readonly_session("main")
dp = (xr.open_zarr(sess.store, group="probabilities")
      .isel(latitude=slice(None, None, -1))
      .sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX)))
slat = dp["latitude"].values; slon = dp["longitude"].values
init_times = pd.DatetimeIndex(dp["reference_time"].values); N = len(init_times)
print(f"  {N} inits, {len(slat)}×{len(slon)}")

def weekly_prob(varbase, bound, wk):
    d0, d1 = WEEKS[wk]
    steps = [np.timedelta64(d, "D") for d in range(d0, d1 + 1)]  # step is timedelta64[D]
    da = dp[f"{varbase}_prob_{bound}_tercile"].sel(step=steps).mean("step") / 100.0
    return da.values  # (init, lat, lon)

# ── ERA5: 2026 obs daily + 1991-2020 climo daily (cached) ────────────────────
print("Opening ARCO-ERA5 …")
de = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                  storage_options={"token": "anon"})
ws = init_times[0] + pd.Timedelta(1, "D"); we = init_times[-1] + pd.Timedelta(42, "D")
obs_dates = pd.date_range(ws, we, freq="D"); o2i = {d: i for i, d in enumerate(obs_dates)}

def fetch_daily(t0, t1, agg):
    da = de["2m_temperature"].sel(latitude=slice(LAT_MAX+1, LAT_MIN-1),
                                  longitude=slice(LON_MIN-1, LON_MAX+1),
                                  time=slice(t0, t1))
    d = (da.resample(time="1D").max("time") if agg == "max" else da.resample(time="1D").mean("time"))
    d = (d.compute() - 273.15).interp(latitude=slat, longitude=slon, method="linear")
    return d

print("Fetching ERA5 2026 obs (mean & max) …")
obs = {}
for varb, (short, agg) in VARS.items():
    d = fetch_daily(f"{ws.date()}T00:00", f"{we.date()}T23:00", agg).reindex(time=obs_dates)
    obs[short] = d.values.astype(np.float32)

# canonical month-day list (non-leap), Jan1 .. mid-May, to align climo years
canon = pd.date_range("2025-01-01", "2025-05-20")
md_list = [(d.month, d.day) for d in canon]
md_idx = {md: i for i, md in enumerate(md_list)}

if os.path.exists(CACHE):
    print(f"Loading climo cache ← {CACHE}")
    cds = xr.open_dataset(CACHE)
    climo = {s: cds[f"{s}_30yr"].values for s in ["tmean", "tmax"]}  # (year, md, lat, lon)
else:
    print(f"Fetching ERA5 {CLIMO_YEARS.start}-{CLIMO_YEARS.stop-1} daily (heavy, one-time) …")
    climo = {s: np.full((len(list(CLIMO_YEARS)), len(md_list), len(slat), len(slon)), np.nan, np.float32)
             for s in ["tmean", "tmax"]}
    for yi, yr in enumerate(CLIMO_YEARS):
        for short, agg in [("tmean", "mean"), ("tmax", "max")]:
            d = fetch_daily(f"{yr}-01-01T00:00", f"{yr}-05-20T23:00", agg)
            dts = pd.DatetimeIndex(d["time"].values)
            for k, t in enumerate(dts):
                j = md_idx.get((t.month, t.day))
                if j is not None:
                    climo[short][yi, j] = d.isel(time=k).values
        print(f"  {yr} done ({yi+1}/30)", flush=True)
    xr.Dataset({f"{s}_30yr": (["year", "md", "latitude", "longitude"], climo[s]) for s in climo},
               coords={"year": list(CLIMO_YEARS), "md": np.arange(len(md_list)),
                       "latitude": slat, "longitude": slon}).to_netcdf(CACHE)
    print(f"  cached → {CACHE}")

# ── Verification loop: build obs category + forecast probs per init×week ──────
print("Computing tercile categories, RPS, RPSS …")
nW = len(WEEKS)
# accumulators for RPSS (per week): sum RPS_fcst, RPS_climo over (init, gridcell)
results = {}
rel_bins = np.linspace(0, 1, 11)  # reliability bins for upper tercile
for varb, (short, agg) in VARS.items():
    rps_f = np.zeros(nW); rps_c = np.zeros(nW); cnt = np.zeros(nW)
    # reliability accumulators (upper tercile), per week: per-bin sum of obs-occurrence & count
    rel_occ = np.zeros((nW, 10)); rel_cnt = np.zeros((nW, 10)); rel_fsum = np.zeros((nW, 10))
    for wi, (wk, (d0, d1)) in enumerate(WEEKS.items()):
        pu = weekly_prob(varb, "upper", wk)  # (init, lat, lon)
        pl = weekly_prob(varb, "lower", wk)
        pm = np.clip(1 - pu - pl, 0, 1)
        for i, idt in enumerate(init_times):
            vd = pd.date_range(idt + pd.Timedelta(d0, "D"), idt + pd.Timedelta(d1, "D"))
            mds = [md_idx.get((d.month, d.day)) for d in vd]
            mds = [m for m in mds if m is not None]
            if not mds: continue
            # observed 2026 weekly mean
            obs_wk = obs[short][[o2i[d] for d in vd if d in o2i]].mean(0)
            # climo weekly mean per year → terciles
            clim_wk = np.nanmean(climo[short][:, mds], axis=1)  # (year, lat, lon)
            t33 = np.nanpercentile(clim_wk, 100/3, axis=0)
            t67 = np.nanpercentile(clim_wk, 200/3, axis=0)
            cat = (obs_wk > t33).astype(int) + (obs_wk > t67).astype(int)  # 0,1,2
            # RPS (3-category cumulative). Forecast CDF: F1=P(lower), F2=P(lower)+P(mid)
            f_cdf1 = pl[i]; f_cdf2 = pl[i] + pm[i]
            o_cdf1 = (cat <= 0).astype(float)  # observed step CDF P(obs<=cat0)
            o_cdf2 = (cat <= 1).astype(float)  # P(obs<=cat1)
            rps = (f_cdf1 - o_cdf1)**2 + (f_cdf2 - o_cdf2)**2
            # climatology forecast CDF: 1/3, 2/3
            rps_clim = (1/3 - o_cdf1)**2 + (2/3 - o_cdf2)**2
            rps_f[wi] += np.nansum(rps); rps_c[wi] += np.nansum(rps_clim); cnt[wi] += np.isfinite(rps).sum()
            # reliability for upper tercile
            occ = (cat == 2).astype(float)
            bins = np.clip((pu[i] * 10).astype(int), 0, 9)
            for b in range(10):
                msk = bins == b
                rel_cnt[wi, b] += msk.sum(); rel_occ[wi, b] += occ[msk].sum(); rel_fsum[wi, b] += pu[i][msk].sum()
        print(f"  {short} W{wk} done")
    rpss = 1 - (rps_f / np.maximum(rps_c, 1e-9))
    results[short] = dict(rpss=rpss, rps_f=rps_f/np.maximum(cnt,1), rps_c=rps_c/np.maximum(cnt,1),
                          rel_occ=rel_occ, rel_cnt=rel_cnt, rel_fsum=rel_fsum)
    print(f"  {short} RPSS: " + " ".join(f"W{w+1}={rpss[w]:+.3f}" for w in range(nW)))

# ── fig25: RPSS vs lead ──────────────────────────────────────────────────────
x = np.arange(1, nW+1); WL = [f"W{w}" for w in WEEKS]
fig, ax = plt.subplots(figsize=(8, 5.2))
sty = {"tmean": ("#D73027", "o", "T2m-mean"), "tmax": ("#F46D43", "s", "T2m-max")}
for short, r in results.items():
    c, mk, lb = sty[short]
    ax.plot(x, r["rpss"], mk+"-", color=c, lw=2, ms=8, label=lb, mec="white", mew=0.5)
ax.axhline(0, color="k", ls="--", lw=1)
ax.fill_between([0.5, nW+0.5], 0, 1, color="#4caf50", alpha=0.06)
ax.fill_between([0.5, nW+0.5], -1, 0, color="#f44336", alpha=0.05)
ax.text(nW+0.05, 0.02, "skill > climatology", fontsize=8, color="green", va="bottom")
ax.text(nW+0.05, -0.02, "worse than climatology", fontsize=8, color="#b71c1c", va="top")
ax.set_xlim(0.5, nW+0.5); ax.set_ylim(-0.5, 1.0)
ax.set_xticks(x); ax.set_xticklabels(WL); ax.set_xlabel("Forecast lead week")
ax.set_ylabel("RPSS (tercile, vs climatology)")
ax.set_title("Spire AI-S2S | Probabilistic skill — tercile RPSS\n"
             f"India, JFM 2026, {N} inits (RPSS>0 ⇒ beats climatology)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIGD}/fig25_rpss_vs_lead.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"→ {FIGD}/fig25_rpss_vs_lead.png")

# ── fig26: reliability diagram (upper tercile, W1/W2/W3) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (short, r) in zip(axes, results.items()):
    for wi, wlab, col in [(0, "W1", "#1a9850"), (1, "W2", "#2166AC"), (2, "W3", "#762a83")]:
        f = r["rel_fsum"][wi] / np.maximum(r["rel_cnt"][wi], 1)
        o = r["rel_occ"][wi] / np.maximum(r["rel_cnt"][wi], 1)
        m = r["rel_cnt"][wi] > 0
        ax.plot(f[m], o[m], "o-", color=col, lw=1.8, ms=6, label=wlab, mec="white", mew=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.axhline(1/3, color="gray", ls=":", lw=0.8); ax.axvline(1/3, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.set_xlabel("Forecast probability (upper tercile)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{sty[short][2]}", fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
fig.suptitle("Spire AI-S2S | Reliability — upper-tercile probability (India, JFM 2026)",
             fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(f"{FIGD}/fig26_reliability_tercile.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"→ {FIGD}/fig26_reliability_tercile.png")

# save scores
xr.Dataset({f"rpss_{s}": ("week", results[s]["rpss"]) for s in results},
           coords={"week": list(WEEKS)}).to_netcdf("prob_scores.nc")
print("Saved prob_scores.nc\nDone ✓")
