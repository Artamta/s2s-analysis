"""
make_lead_degradation.py — Skill vs. lead for ONE fixed valid week.

The classic S2S diagnostic: hold the verifying week fixed (12-18 Feb 2026) and
forecast it from progressively earlier initializations, so lead time is the only
thing that changes.

    Lead   Init        Forecast days (of that init)
    Week 1 2026-02-12  1-7
    Week 2 2026-02-05  8-14
    Week 3 2026-01-29  15-21
    Week 4 2026-01-22  22-28
    Week 5 2026-01-15  29-35
    Week 6 2026-01-08  36-42

Z500/TP anomalies come from the pre-aggregated product; T2M is computed fresh
for all six inits (cached). Outputs a PCC-vs-lead curve (Z500, TP, T2M) and a
CSV of the numbers.

CAVEAT (printed and captioned): this is a SINGLE event (N=1), so the curves are
one realization, not an expectation — week-to-week reordering of the dynamical
models is sampling noise. T2M anomalies are referenced to ERA5 climatology, which
disadvantages the dynamical models (own-climatology bias contaminates the score);
read T2M skill qualitatively. See PRESENTATION_BRIEF.md / the diagnostic.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import case1_analysis as c1  # noqa: E402

OUT = c1.CONFIG["outdir"]

# (lead#, init, day0, day1) — all target the valid week 12-18 Feb 2026
LEAD_PAIRS = [
    (1, "2026-02-12", 1, 7),
    (2, "2026-02-05", 8, 14),
    (3, "2026-01-29", 15, 21),
    (4, "2026-01-22", 22, 28),
    (5, "2026-01-15", 29, 35),
    (6, "2026-01-08", 36, 42),
]
MODELS = ["SPIRE", "FuXi", "ECMWF", "NCEP"]
COL = {"SPIRE": "#E8720C", "FuXi": "#5B8FB9", "ECMWF": "#4FA06A", "NCEP": "#9B7FC2"}
STY = {"SPIRE": ("-", "o"), "FuXi": ("-", "s"), "ECMWF": ("--", "^"), "NCEP": ("-.", "D")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def t2m_forecast(grid, clim_t2m, init, d0, d1):
    """Per-model T2M anomaly (deg C) for one (init, day-range), India-masked."""
    init_str = pd.to_datetime(init).strftime("%Y%m%d")
    raw = {
        "SPIRE": c1._spire_t2m(init, d0, d1),
        "FuXi": c1._fuxi_t2m(init_str, d0, d1),
        "ECMWF": c1._op_t2m("ecmwf", init_str, d0, d1),
        "NCEP": c1._op_t2m("ncep", init_str, d0, d1),
    }
    out = {}
    for m in MODELS:
        out[m] = None if raw[m] is None else grid.apply_land(grid.regrid(raw[m]) - clim_t2m)
    return out


def build():
    base = xr.open_dataset(c1.CONFIG["weekly_fields"])
    grid = c1.Grid(base.lat.values, base.lon.values)
    clim = c1.load_climatology(grid)
    ds = xr.open_dataset(c1.CONFIG["weekly_fields"])
    inits = [str(x)[:10] for x in ds.init.values]

    # ERA5 truth (same valid week for every lead)
    dailyT = xr.open_dataset(c1.CONFIG["era5_daily_t2m"])["t2m"]
    obsT = grid.apply_land(grid.regrid(
        dailyT.sel(time=slice(*c1.TARGET_WEEK)).mean("time")) - clim["t2m"])

    # T2M cache keyed by init
    cache = os.path.join(OUT, "t2m_leads_cache.nc")
    cached = xr.open_dataset(cache) if os.path.exists(cache) else None
    new_vars = {}

    rows = []
    fields = {"Z500": {}, "TP": {}, "T2M": {}}
    for (lead, init, d0, d1) in LEAD_PAIRS:
        wk = f"Week {lead}"
        ii = inits.index(init)
        obsZ = grid.apply_land(ds["z_obs"].isel(init=ii).sel(week=wk).values)
        obsP = grid.apply_land(ds["tp_obs"].isel(init=ii).sel(week=wk).values)

        # T2M forecasts (from cache if present, else compute)
        t2m = {}
        need = any(cached is None or f"L{lead}__{m}" not in cached for m in MODELS)
        if need:
            print(f"  computing T2M  lead {lead}  init {init} (days {d0}-{d1}) ...", flush=True)
            t2m = t2m_forecast(grid, clim["t2m"], init, d0, d1)
            for m in MODELS:
                if t2m[m] is not None:
                    new_vars[f"L{lead}__{m}"] = (("lat", "lon"), np.asarray(t2m[m], float))
        else:
            for m in MODELS:
                v = f"L{lead}__{m}"
                t2m[m] = grid.apply_land(cached[v].values) if v in cached else None

        for var, obs in [("Z500", obsZ), ("TP", obsP), ("T2M", None)]:
            fields[var][lead] = {}
            for m in MODELS:
                if var == "Z500":
                    fc = grid.apply_land(ds["z_fcst"].sel(model=m).isel(init=ii).sel(week=wk).values)
                elif var == "TP":
                    fc = grid.apply_land(ds["tp_fcst"].sel(model=m).isel(init=ii).sel(week=wk).values)
                else:
                    fc = t2m[m]
                o = {"Z500": obsZ, "TP": obsP, "T2M": obsT}[var]
                s = c1.spatial_metrics(fc, o, grid.weights)
                fields[var][lead][m] = s["pcc"]
                rows.append(dict(Lead=lead, Init=init, Variable=var, Model=m,
                                 PCC=s["pcc"], RMSE=s["rmse"], Bias=s["bias"]))

    # save T2M cache (unmasked) if we computed anything new
    if new_vars:
        merged = {} if cached is None else {k: cached[k] for k in cached.data_vars}
        merged.update({k: xr.DataArray(v[1], dims=v[0]) for k, v in new_vars.items()})
        xr.Dataset(merged, coords={"lat": grid.lat, "lon": grid.lon}).to_netcdf(cache)
        print(f"  cached T2M leads -> {cache}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lead_degradation.csv"), index=False)
    return fields, df


def plot(fields):
    variables = [("Z500", "Z500 anomaly PCC"), ("TP", "Precip anomaly PCC"),
                 ("T2M", "T2M anomaly PCC")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    leads = [p[0] for p in LEAD_PAIRS]
    for ax, (var, title) in zip(axes, variables):
        for m in MODELS:
            ys = [fields[var][l][m] for l in leads]
            ls, mk = STY[m]
            lw = 3.0 if m == "SPIRE" else 1.9
            z = 5 if m == "SPIRE" else 3
            ax.plot(leads, ys, ls, marker=mk, color=COL[m], lw=lw, ms=8,
                    label=m, zorder=z, markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(0.5, color="#555", lw=1.0, ls=(0, (6, 3)))
        ax.axhline(0.0, color="#222", lw=0.8)
        ax.set_xticks(leads); ax.set_xticklabels([f"W{l}" for l in leads])
        ax.set_xlabel("Lead (verifying 12-18 Feb 2026)")
        ax.set_ylim(-0.45, 1.05); ax.set_title(title, pad=8)
        ax.grid(axis="y", ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Pattern correlation (PCC)")
    axes[0].legend(loc="lower left", fontsize=11, frameon=True, edgecolor="#CCC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pres_06_skill_vs_lead.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote pres_06_skill_vs_lead", flush=True)


if __name__ == "__main__":
    print("Building lead-degradation (6 leads, fixed valid week) ...", flush=True)
    fields, df = build()
    plot(fields)
    print("\nPCC by lead (rounded):", flush=True)
    print(df.pivot_table(index=["Variable", "Model"], columns="Lead",
                         values="PCC").round(2).to_string(), flush=True)
    print("\nLEAD_DEGRADATION_DONE", flush=True)
