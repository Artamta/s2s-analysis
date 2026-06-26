#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a5_wetdry_spells.py  —  Weekly all-India rainfall: ERA5 vs SPIRE/FuXi/ECMWF
                        with wet-spell / dry-spell background shading.
================================================================================
The JJAS deck has a popular figure: weekly all-India monsoon rainfall, ERA5 truth
vs the S2S models, with blue/red background bands marking ERA5 active (wet) and
break (suppressed) weeks. This is the JFM2026 (winter) analogue.

JFM is the DRY half-year, so "active/break monsoon" becomes "wet-spell / dry-spell"
(winter rain over India is driven by western disturbances). The construction is
identical to the JJAS script:

    index(week) = ( weekly all-India TP  −  ERA5 normal for that calendar week )
                  / ERA5 inter-annual std for that calendar week              [σ]

  index > +1σ  -> wet-spell week   (blue background)
  index < −1σ  -> dry-spell week   (red background)

ERA5 normal + std come from the WeatherBench2 ERA5 archive (total_precipitation_24hr
at 06Z, 1990–2020), area-averaged over the India box and standardised PER CALENDAR
WEEK — cached to a CSV so only the first run touches the cloud.

The forecast + truth weekly means reuse the SAME verified adapters as the rest of
new_anal, so SPIRE is included here (the JJAS version only had FuXi/ECMWF).

Three figures (mirroring the JJAS set):
  A5_rainfall_lines  weekly all-India rainfall lines + wet/dry background  ← the ask
  A5_index_lines     the standardised σ-index lines (truth vs each model)
  A5_event_scores    wet/dry-spell detection skill (POD / FAR / hits) per model

  python a5_wetdry_spells.py
  python a5_wetdry_spells.py --clim-years 1995 2020 --no-cache
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from core import grid as G
from core import metrics as M
from core.config import WEEKS
from core.aggregate import valid_dates_for

# WeatherBench2 ERA5 (same archive + variable the JJAS script uses)
WB2 = ("gs://weatherbench2/datasets/era5/"
       "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
BOX = dict(s=5.0, n=38.0, w=65.0, e=100.0)            # India box (matches grid)
CLIM_CACHE = os.path.join(C.HERE, "era5_all_india_daily_1990_2020.csv")

COLORS = {"ERA5": "#1f2937", "SPIRE": "#0072B2", "FuXi": "#D55E00", "ECMWF": "#009E73"}
MODELS = ["SPIRE", "FuXi", "ECMWF"]
SIGMA = 1.0                                           # wet/dry-spell threshold


# ============================================================== ERA5 climatology
def load_era5_daily_ai(years, no_cache=False):
    """All-India daily mean TP [mm/day] for the climatology years (JFM..May season).
       Cached; only the first run pulls from WeatherBench2."""
    if os.path.exists(CLIM_CACHE) and not no_cache:
        s = pd.read_csv(CLIM_CACHE, parse_dates=["time"])
        print(f"  ERA5 clim: using cache {os.path.basename(CLIM_CACHE)} ({len(s)} days)")
        return s.set_index("time")["tp_mm_day"].sort_index()
    y0, y1 = years
    print(f"  ERA5 clim: building all-India daily TP {y0}-{y1} from WeatherBench2...")
    ds = xr.open_zarr(WB2, storage_options=dict(token="anon"))
    tp = ds["total_precipitation_24hr"].sel(
        latitude=slice(BOX["s"], BOX["n"]), longitude=slice(BOX["w"], BOX["e"]))
    tp = tp.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    tp = tp.sel(time=tp["time.hour"] == 6)                       # 24-h total valid 06Z
    tp = tp.sel(time=tp["time.month"].isin([1, 2, 3, 4, 5]))     # JFM..early-May season
    w = np.cos(np.deg2rad(tp.latitude))
    ai = (tp.weighted(w).mean(["latitude", "longitude"]) * 1000.0).load()
    s = ai.to_series().rename("tp_mm_day").sort_index()
    s.to_csv(CLIM_CACHE, index_label="time")
    print(f"  ERA5 clim: cached -> {os.path.basename(CLIM_CACHE)} ({len(s)} days)")
    return s


def same_month_day_window(series, start, end, years):
    """ERA5 all-India mean TP for the same calendar window in every climatology year."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for year in range(years[0], years[1] + 1):
        s = pd.Timestamp(year=year, month=start.month, day=start.day)
        e = pd.Timestamp(year=year, month=end.month, day=end.day)
        if e < s:
            continue
        vals = series.loc[s:e]
        if len(vals) >= 5:
            out.append(float(vals.mean()))
    return np.asarray(out, dtype=float)


# ============================================================== weekly model means
def collect_weekly(cfg, GC, truth, years, clim_series):
    """One row per (init, week): ERA5 + each model all-India weekly TP, plus the
       ERA5 calendar-week normal and inter-annual std -> standardised indices."""
    w = GC["weights"]
    rda = G.region_da("All India", GC)

    def ai_mean(field):
        return float(M.wmean(field.where(rda), w)) if field is not None else np.nan

    rows = []
    for init in cfg.init_dates:
        for wi in range(6):
            wn, ds, de = WEEKS[wi]
            valid = valid_dates_for(init, ds, de, cfg.valid_end)
            if not valid:
                continue
            o = C.truth_weekly_mean(cfg, truth, GC, init, "TP", wi)
            if o is None:
                continue
            fields = C.model_weekly_mean(cfg, GC, init, "TP", wi)
            vstart, vend = pd.Timestamp(valid[0]), pd.Timestamp(valid[-1])
            clim_vals = same_month_day_window(clim_series, vstart, vend, years)
            cmean = float(np.nanmean(clim_vals)) if clim_vals.size else np.nan
            cstd = float(np.nanstd(clim_vals, ddof=1)) if clim_vals.size > 1 else np.nan
            row = dict(init=init, week=wi + 1, valid_start=vstart, valid_end=vend,
                       valid_mid=vstart + (vend - vstart) / 2,
                       clim_tp=cmean, clim_std=cstd, ERA5_tp=ai_mean(o))
            for m in MODELS:
                row[f"{m}_tp"] = ai_mean(fields.get(m))
            rows.append(row)
    df = pd.DataFrame(rows)
    std = df["clim_std"].replace(0, np.nan)
    for m in ["ERA5"] + MODELS:
        df[f"{m}_index"] = (df[f"{m}_tp"] - df["clim_tp"]) / std
    return df


def calendar_average(df):
    """Average the matched windows that share a valid week -> one point per week."""
    keys = ["valid_start", "valid_end", "valid_mid"]
    agg = {f"{m}_tp": "mean" for m in ["ERA5"] + MODELS}
    agg.update({f"{m}_index": "mean" for m in ["ERA5"] + MODELS})
    agg["clim_tp"] = "mean"
    cal = df.groupby(keys, as_index=False).agg(agg).sort_values("valid_mid")
    return cal.reset_index(drop=True)


# ============================================================== event scoring
def classify(x):
    if pd.isna(x):
        return "none"
    if x > SIGMA:
        return "wet"
    if x < -SIGMA:
        return "dry"
    return "normal"


def score_events(df):
    rows = []
    obs = df["ERA5_index"].map(classify)
    for m in MODELS:
        pred = df[f"{m}_index"].map(classify)
        for event in ["wet", "dry"]:
            oe, pe = obs == event, pred == event
            hits = int((oe & pe).sum()); misses = int((oe & ~pe).sum())
            fa = int((~oe & pe).sum()); cn = int((~oe & ~pe).sum())
            pod = hits / (hits + misses) if hits + misses else np.nan
            far = fa / (hits + fa) if hits + fa else np.nan
            acc = (hits + cn) / len(df) if len(df) else np.nan
            rows.append(dict(model=m, event=event, hits=hits, misses=misses,
                             false_alarms=fa, pod=pod, far=far, accuracy=acc))
    return pd.DataFrame(rows)


# ============================================================== theme + figures
def theme():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 320, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.24, "grid.linestyle": ":",
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _shade_spells(ax, cal):
    for r in cal.itertuples(index=False):
        if r.ERA5_index > SIGMA:
            ax.axvspan(r.valid_start, r.valid_end, color="#dbeafe", alpha=0.55, zorder=0)
        elif r.ERA5_index < -SIGMA:
            ax.axvspan(r.valid_start, r.valid_end, color="#fee2e2", alpha=0.60, zorder=0)


def fig_rainfall_lines(cal):
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    yvals = cal[[f"{m}_tp" for m in ["ERA5"] + MODELS] + ["clim_tp"]].to_numpy()
    ymax = max(1.0, float(np.nanmax(yvals)) * 1.18)
    _shade_spells(ax, cal)
    for m in ["ERA5"] + MODELS:
        ax.plot(cal["valid_mid"], cal[f"{m}_tp"], marker="o", ms=5.0,
                lw=2.6 if m == "ERA5" else 2.1, color=COLORS[m], label=m)
    ax.plot(cal["valid_mid"], cal["clim_tp"], color="0.45", lw=1.5, ls="--",
            label="ERA5 normal")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("All-India weekly rainfall (mm day$^{-1}$)")
    ax.set_xlabel("Valid-window midpoint date")
    ax.set_title("Weekly winter rainfall over India: ERA5 truth vs SPIRE / FuXi / ECMWF  (JFM2026)")
    ax.text(0.012, 0.93, "Blue background: ERA5 wet-spell week (> +1σ)",
            transform=ax.transAxes, color="#1d4ed8", fontweight="bold")
    ax.text(0.012, 0.05, "Red background: ERA5 dry-spell week (< −1σ)",
            transform=ax.transAxes, color="#b91c1c", fontweight="bold")
    ax.legend(loc="upper right", ncol=5)
    fig.text(0.015, 0.012,
             "Wet/dry background = ERA5 weekly all-India TP anomaly standardised by the "
             "1990–2020 ERA5 calendar-week distribution.",
             fontsize=8.8, color="0.30")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    C.savefig(fig, "A5_rainfall_lines.png")


def fig_index_lines(cal):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    yv = cal[[f"{m}_index" for m in ["ERA5"] + MODELS]].to_numpy()
    ymin = min(-2.4, float(np.nanmin(yv)) - 0.35)
    ymax = max(2.4, float(np.nanmax(yv)) + 0.35)
    ax.axhspan(SIGMA, ymax, color="#dbeafe", alpha=0.55, zorder=0)
    ax.axhspan(ymin, -SIGMA, color="#fee2e2", alpha=0.55, zorder=0)
    ax.axhline(SIGMA, color="#2563eb", ls="--", lw=1.0)
    ax.axhline(-SIGMA, color="#dc2626", ls="--", lw=1.0)
    ax.axhline(0, color="0.20", lw=0.9)
    for m in ["ERA5"] + MODELS:
        ax.plot(cal["valid_mid"], cal[f"{m}_index"], marker="o", ms=5.0,
                lw=2.5 if m == "ERA5" else 2.0, color=COLORS[m], label=m)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Weekly rainfall index (σ)")
    ax.set_xlabel("Valid-window midpoint date")
    ax.set_title("Weekly wet/dry-spell index over India: ERA5 truth vs SPIRE / FuXi / ECMWF  (JFM2026)")
    ax.text(0.012, 0.93, "Wet-spell week (> +1σ)", transform=ax.transAxes,
            color="#1d4ed8", fontweight="bold")
    ax.text(0.012, 0.05, "Dry-spell week (< −1σ)", transform=ax.transAxes,
            color="#b91c1c", fontweight="bold")
    ax.legend(loc="upper right", ncol=5)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    C.savefig(fig, "A5_index_lines.png")


def fig_event_scores(scores):
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(scores))
    bar_colors = [COLORS[r.model] for r in scores.itertuples()]
    ax.bar(x, scores["pod"], color=bar_colors, alpha=0.85, edgecolor="white", label="Hit rate (POD)")
    ax.scatter(x, scores["far"], color="0.15", marker="x", s=90, label="False-alarm ratio (FAR)")
    for xi, r in enumerate(scores.itertuples()):
        ax.text(xi, min((r.pod if np.isfinite(r.pod) else 0) + 0.04, 0.98),
                f"H={r.hits}\nM={r.misses}\nFA={r.false_alarms}",
                ha="center", va="bottom", fontsize=8.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.model}\n{r.event}-spell" for r in scores.itertuples()])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Event score")
    ax.set_title("Wet/dry-spell week detection skill  (vs ERA5, JFM2026)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    C.savefig(fig, "A5_event_scores.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clim-years", nargs=2, type=int, default=[1990, 2020])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    theme()
    cfg = C.get_cfg(1.5)
    GC = C.grid_ctx(cfg)
    truth = C.open_truth(cfg)
    years = tuple(args.clim_years)
    print(f"A5 wet/dry-spell rainfall  (clim {years[0]}-{years[1]})")
    clim_series = load_era5_daily_ai(years, no_cache=args.no_cache)
    df = collect_weekly(cfg, GC, truth, years, clim_series)
    cal = calendar_average(df)
    scores = score_events(df)

    fig_rainfall_lines(cal)
    fig_index_lines(cal)
    fig_event_scores(scores)

    df.to_csv(os.path.join(C.FIGS, "A5_weekly_windows.csv"), index=False)
    cal.to_csv(os.path.join(C.FIGS, "A5_calendar_average.csv"), index=False)
    scores.to_csv(os.path.join(C.FIGS, "A5_event_scores.csv"), index=False)
    print(f"  wrote A5_weekly_windows.csv / A5_calendar_average.csv / A5_event_scores.csv")
    print("\n=== wet/dry-spell detection (vs ERA5) ===")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
