#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare ERA5, FuXi-S2S, and ECMWF-S2S using a weekly monsoon active/break index.

Classic active/break spells are daily. S2S verification here is weekly, so this
script makes a weekly analogue:

    index = (weekly all-India TP - ERA5 1979-2022 normal for that calendar week)
            / ERA5 interannual std for that calendar week

Positive values are wetter-than-normal monsoon weeks; negative values are
suppressed weeks. +/-1 sigma is used as an active/break-style threshold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/raj.ayush/s2s/s2s_anlysis")
HERE = ROOT / "final_analysis/jjas/plots"
FIGS = HERE / "figs"
DEFAULT_2019 = FIGS / "matched_monsoon_study"
DEFAULT_ALL = FIGS / "matched_monsoon_study_all_years"
WB2_ZARR_15 = (
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
LAT_S, LAT_N = 5.0, 38.0
LON_W, LON_E = 65.0, 100.0
CLIM_Y0, CLIM_Y1 = 1979, 2022
COLORS = {"ERA5": "#1f2937", "FuXi": "#D55E00", "ECMWF": "#009E73"}


def choose_source(source):
    if source == "all-years":
        return DEFAULT_ALL
    if source == "2019":
        return DEFAULT_2019
    if source != "auto":
        return Path(source)
    if (DEFAULT_ALL / "matched_pairwise_deltas.csv").exists():
        return DEFAULT_ALL
    return DEFAULT_2019


def load_pairwise(source):
    path = Path(source) / "matched_pairwise_deltas.csv"
    df = pd.read_csv(path, parse_dates=["valid_start", "valid_end", "fuxi_init", "ecmwf_init"])
    df = df[(df["variable"] == "TP") & (df["region"] == "All India")].copy()
    if "lead_week" not in df:
        df["lead_week"] = pd.to_numeric(df["week"].str.extract(r"(\d+)")[0], errors="coerce")
    df["valid_mid"] = df["valid_start"] + (df["valid_end"] - df["valid_start"]) / 2
    df["obs_mean"] = df["obs_mean_FuXi"]
    return df.sort_values(["valid_mid", "lead_week", "fuxi_init"])


def cos_weighted_area_mean(da):
    w = np.cos(np.deg2rad(da["latitude"]))
    weights = xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": da["latitude"]})
    return da.weighted(weights).mean(dim=["latitude", "longitude"])


def load_era5_daily_ai(cache):
    cache = Path(cache)
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["time"])
        return s.set_index("time")["tp_mm_day"].sort_index()

    ds = xr.open_zarr(WB2_ZARR_15)
    tp = ds["total_precipitation_24hr"]
    tp = tp.sel(latitude=slice(LAT_S, LAT_N), longitude=slice(LON_W, LON_E))
    tp = tp.sel(time=slice(f"{CLIM_Y0}-01-01", f"{CLIM_Y1}-12-31"))
    tp = tp.sel(time=tp["time.hour"] == 6)
    tp = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    ai = (cos_weighted_area_mean(tp) * 1000.0).load()
    s = ai.to_series().rename("tp_mm_day").sort_index()
    s.to_csv(cache, index_label="time")
    return s


def same_month_day_window(series, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    out = []
    for year in range(CLIM_Y0, CLIM_Y1 + 1):
        s = pd.Timestamp(year=year, month=start.month, day=start.day)
        e = pd.Timestamp(year=year, month=end.month, day=end.day)
        if e < s:
            continue
        vals = series.loc[s:e]
        if len(vals) >= 5:
            out.append(float(vals.mean()))
    return np.asarray(out, dtype=float)


def add_climatology(df, era5_series):
    unique = df[["valid_start", "valid_end"]].drop_duplicates().copy()
    rows = []
    for row in unique.itertuples(index=False):
        vals = same_month_day_window(era5_series, row.valid_start, row.valid_end)
        rows.append({
            "valid_start": row.valid_start,
            "valid_end": row.valid_end,
            "clim_mean": float(np.nanmean(vals)),
            "clim_std": float(np.nanstd(vals, ddof=1)),
            "clim_n": int(np.isfinite(vals).sum()),
        })
    clim = pd.DataFrame(rows)
    out = df.merge(clim, on=["valid_start", "valid_end"], how="left")
    std = out["clim_std"].replace(0, np.nan)
    out["ERA5_index"] = (out["obs_mean"] - out["clim_mean"]) / std
    out["FuXi_index"] = (out["fcst_mean_FuXi"] - out["clim_mean"]) / std
    out["ECMWF_index"] = (out["fcst_mean_ECMWF"] - out["clim_mean"]) / std
    return out


def classify(x):
    if pd.isna(x):
        return "missing"
    if x > 1.0:
        return "active"
    if x < -1.0:
        return "break"
    return "normal"


def make_calendar_average(df):
    rows = []
    keys = ["valid_start", "valid_end", "valid_mid"]
    for key, sub in df.groupby(keys, dropna=False):
        base = dict(zip(keys, key))
        rows.append({
            **base,
            "ERA5": sub["ERA5_index"].mean(),
            "FuXi": sub["FuXi_index"].mean(),
            "ECMWF": sub["ECMWF_index"].mean(),
            "ERA5_tp": sub["obs_mean"].mean(),
            "FuXi_tp": sub["fcst_mean_FuXi"].mean(),
            "ECMWF_tp": sub["fcst_mean_ECMWF"].mean(),
            "clim_tp": sub["clim_mean"].mean(),
            "n_forecasts": len(sub),
        })
    return pd.DataFrame(rows).sort_values("valid_mid")


def score_events(df):
    rows = []
    for model, col in [("FuXi", "FuXi_index"), ("ECMWF", "ECMWF_index")]:
        obs = df["ERA5_index"].map(classify)
        pred = df[col].map(classify)
        for event in ["active", "break"]:
            obs_event = obs == event
            pred_event = pred == event
            hits = int((obs_event & pred_event).sum())
            misses = int((obs_event & ~pred_event).sum())
            false_alarms = int((~obs_event & pred_event).sum())
            correct_neg = int((~obs_event & ~pred_event).sum())
            pod = hits / (hits + misses) if hits + misses else np.nan
            far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
            accuracy = (hits + correct_neg) / len(df) if len(df) else np.nan
            rows.append({
                "model": model,
                "event": event,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_neg,
                "pod": pod,
                "far": far,
                "accuracy": accuracy,
            })
    return pd.DataFrame(rows)


def apply_theme():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def corr_label(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return "r=NA"
    return f"r={np.corrcoef(x[mask], y[mask])[0, 1]:.2f}"


def plot(df, calendar, scores, out, tag):
    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.axhspan(1, 3.5, color="#dbeafe", alpha=0.55, zorder=0)
    ax0.axhspan(-3.5, -1, color="#fee2e2", alpha=0.55, zorder=0)
    ax0.axhline(1, color="#2563eb", ls="--", lw=1.0)
    ax0.axhline(-1, color="#dc2626", ls="--", lw=1.0)
    ax0.axhline(0, color="0.25", lw=0.9)
    for model in ["ERA5", "FuXi", "ECMWF"]:
        ax0.plot(calendar["valid_mid"], calendar[model], marker="o", ms=4.2,
                 lw=2.1 if model == "ERA5" else 1.8,
                 color=COLORS[model], label=model, alpha=0.95)
    ax0.set_ylabel("Weekly monsoon index (sigma)")
    ax0.set_title("Calendar-valid weekly active/break index: ERA5 truth vs FuXi-S2S vs ECMWF-S2S")
    ax0.legend(loc="upper right", ncol=3)
    ax0.text(0.01, 0.92, "Active-style wet week", transform=ax0.transAxes,
             color="#1d4ed8", fontweight="bold")
    ax0.text(0.01, 0.05, "Break-style suppressed week", transform=ax0.transAxes,
             color="#b91c1c", fontweight="bold")

    lead = df["lead_week"].to_numpy()
    for model, col, marker in [("FuXi", "FuXi_index", "o"), ("ECMWF", "ECMWF_index", "s")]:
        ax1.scatter(df["ERA5_index"], df[col], c=lead, cmap="viridis", vmin=1, vmax=6,
                    s=48, alpha=0.82, marker=marker, edgecolor="white", linewidth=0.45,
                    label=f"{model} ({corr_label(df['ERA5_index'].to_numpy(), df[col].to_numpy())})")
    lim = np.nanmax(np.abs(df[["ERA5_index", "FuXi_index", "ECMWF_index"]].to_numpy())) + 0.35
    lim = max(2.0, float(lim))
    ax1.plot([-lim, lim], [-lim, lim], color="0.30", ls="--", lw=1.0)
    ax1.axhline(1, color="#2563eb", ls=":", lw=0.9)
    ax1.axhline(-1, color="#dc2626", ls=":", lw=0.9)
    ax1.axvline(1, color="#2563eb", ls=":", lw=0.9)
    ax1.axvline(-1, color="#dc2626", ls=":", lw=0.9)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("ERA5 weekly index")
    ax1.set_ylabel("Forecast weekly index")
    ax1.set_title("Matched-window amplitude agreement")
    ax1.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(ax1.collections[0], ax=ax1, pad=0.02)
    cb.set_label("Lead week")
    cb.set_ticks(range(1, 7))

    x = np.arange(len(scores))
    labels = [f"{r.model}\n{r.event}" for r in scores.itertuples()]
    ax2.bar(x, scores["pod"], color=["#D55E00", "#D55E00", "#009E73", "#009E73"],
            alpha=0.86, edgecolor="white", label="Hit rate")
    ax2.scatter(x, scores["far"], color="0.15", marker="x", s=80, label="False alarm ratio")
    for xi, row in enumerate(scores.itertuples()):
        ax2.text(xi, min(row.pod + 0.04, 0.98), f"H={row.hits}\nM={row.misses}",
                 ha="center", va="bottom", fontsize=8.5, color="0.25")
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Event score")
    ax2.set_title("Active/break-style event detection")
    ax2.legend(loc="upper right")

    years = sorted(pd.to_datetime(df["valid_start"]).dt.year.unique())
    year_text = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
    fig.suptitle(f"Weekly monsoon active/break comparison ({year_text}; matched valid windows)",
                 fontsize=14.5, fontweight="bold")
    fig.text(
        0.02, 0.012,
        "Index = weekly all-India TP anomaly normalized by ERA5 1979-2022 interannual std for the same calendar week. "
        "Thresholds +/-1 sigma are an S2S weekly analogue of daily active/break spells.",
        fontsize=8.7, color="0.28",
    )
    path = out / f"active_break_weekly_index_fuxi_ecmwf_era5_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_line_only(calendar, out, tag):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    yvals = calendar[["ERA5", "FuXi", "ECMWF"]].to_numpy()
    ymin = min(-2.4, float(np.nanmin(yvals)) - 0.35)
    ymax = max(2.4, float(np.nanmax(yvals)) + 0.35)
    ax.axhspan(1, ymax, color="#dbeafe", alpha=0.55, zorder=0)
    ax.axhspan(ymin, -1, color="#fee2e2", alpha=0.55, zorder=0)
    ax.axhline(1, color="#2563eb", ls="--", lw=1.0)
    ax.axhline(-1, color="#dc2626", ls="--", lw=1.0)
    ax.axhline(0, color="0.20", lw=0.9)
    for model in ["ERA5", "FuXi", "ECMWF"]:
        ax.plot(calendar["valid_mid"], calendar[model],
                marker="o", ms=5.0, lw=2.4 if model == "ERA5" else 2.1,
                color=COLORS[model], label=model)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Weekly monsoon index (sigma)")
    ax.set_xlabel("Valid-window midpoint date")
    ax.set_title("Weekly active/break-style monsoon index: ERA5 truth vs FuXi-S2S vs ECMWF-S2S",
                 fontsize=13.0, fontweight="bold")
    ax.text(0.012, 0.92, "Active-style wet week (> +1 sigma)", transform=ax.transAxes,
            color="#1d4ed8", fontweight="bold")
    ax.text(0.012, 0.055, "Break-style suppressed week (< -1 sigma)", transform=ax.transAxes,
            color="#b91c1c", fontweight="bold")
    ax.legend(loc="upper right", ncol=3)
    fig.text(
        0.015, 0.015,
        "Index = weekly all-India TP anomaly normalized by ERA5 1979-2022 interannual std for the same calendar week.",
        fontsize=8.8, color="0.30",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = out / f"active_break_weekly_index_lines_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rainfall_lines(calendar, out, tag):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    yvals = calendar[["ERA5_tp", "FuXi_tp", "ECMWF_tp", "clim_tp"]].to_numpy()
    ymax = max(1.0, float(np.nanmax(yvals)) * 1.18)
    for row in calendar.itertuples(index=False):
        if row.ERA5 > 1:
            ax.axvspan(row.valid_start, row.valid_end, color="#dbeafe", alpha=0.55, zorder=0)
        elif row.ERA5 < -1:
            ax.axvspan(row.valid_start, row.valid_end, color="#fee2e2", alpha=0.60, zorder=0)
    for model, col in [("ERA5", "ERA5_tp"), ("FuXi", "FuXi_tp"), ("ECMWF", "ECMWF_tp")]:
        ax.plot(calendar["valid_mid"], calendar[col],
                marker="o", ms=5.0, lw=2.5 if model == "ERA5" else 2.1,
                color=COLORS[model], label=model)
    ax.plot(calendar["valid_mid"], calendar["clim_tp"], color="0.45", lw=1.4,
            ls="--", label="ERA5 normal")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("All-India weekly rainfall (mm day$^{-1}$)")
    ax.set_xlabel("Valid-window midpoint date")
    ax.set_title("Weekly monsoon rainfall: ERA5 truth vs FuXi-S2S vs ECMWF-S2S",
                 fontsize=13.0, fontweight="bold")
    ax.text(0.012, 0.92, "Blue background: ERA5 active-style wet week",
            transform=ax.transAxes, color="#1d4ed8", fontweight="bold")
    ax.text(0.012, 0.055, "Red background: ERA5 break-style suppressed week",
            transform=ax.transAxes, color="#b91c1c", fontweight="bold")
    ax.legend(loc="upper right", ncol=4)
    fig.text(
        0.015, 0.015,
        "Active/break background is based on ERA5 weekly anomaly index standardized by the 1979-2022 ERA5 calendar-week distribution.",
        fontsize=8.8, color="0.30",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = out / f"active_break_weekly_rainfall_lines_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_readme(out, source, tag, scores, fig_path, line_path, rainfall_path):
    score_lines = [
        "| model | event | hits | misses | false alarms | POD | FAR | accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores.itertuples(index=False):
        score_lines.append(
            f"| {row.model} | {row.event} | {row.hits} | {row.misses} | "
            f"{row.false_alarms} | {row.pod:.2f} | {row.far:.2f} | {row.accuracy:.2f} |"
        )
    lines = [
        "# Weekly Active/Break-Style Model Comparison",
        "",
        f"Source: `{source}`",
        f"Figure: `{fig_path.name}`",
        f"Line-only figure: `{line_path.name}`",
        f"Rainfall line figure: `{rainfall_path.name}`",
        "",
        "This is a weekly S2S analogue of active/break analysis, not the classic daily spell definition.",
        "The index is computed from all-India weekly mean TP, standardized by the ERA5 1979-2022 distribution for the same calendar week.",
        "",
        "Positive values mean wetter-than-normal monsoon weeks. Negative values mean suppressed weeks.",
        "Thresholds `> +1` and `< -1` mark active-style and break-style weeks.",
        "",
        "## Event Scores",
        "\n".join(score_lines),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--era5-cache", default=str(FIGS / "era5_all_india_jjas_daily_1979_2022.csv"))
    args = ap.parse_args()

    apply_theme()
    source = choose_source(args.source)
    df = load_pairwise(source)
    tag = "all_years" if "all_years" in str(source) else str(pd.to_datetime(df["valid_start"]).dt.year.min())
    out = Path(args.out) if args.out else FIGS / f"active_break_model_compare_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    era5 = load_era5_daily_ai(args.era5_cache)
    df = add_climatology(df, era5)
    calendar = make_calendar_average(df)
    scores = score_events(df)
    fig_path = plot(df, calendar, scores, out, tag)
    line_path = plot_line_only(calendar, out, tag)
    rainfall_path = plot_rainfall_lines(calendar, out, tag)

    df.to_csv(out / "active_break_weekly_index_windows.csv", index=False)
    calendar.to_csv(out / "active_break_weekly_index_calendar_average.csv", index=False)
    scores.to_csv(out / "active_break_event_scores.csv", index=False)
    write_readme(out, source, tag, scores, fig_path, line_path, rainfall_path)
    print(f"WROTE {fig_path}")
    print(f"WROTE {line_path}")
    print(f"WROTE {rainfall_path}")
    print(f"WROTE tables -> {out}")


if __name__ == "__main__":
    main()
