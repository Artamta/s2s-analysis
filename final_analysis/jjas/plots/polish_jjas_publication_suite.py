#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
polish_jjas_publication_suite.py
================================
Curate and polish JJAS monsoon verification plots for the paper.

This script reads the matched-valid-window outputs produced by
matched_monsoon_study.py / aggregate_matched_monsoon_studies.py and creates a
compact publication-facing figure set. It is deliberately data-light: it does
not reopen raw NetCDF fields, so it can run quickly while the all-year compute
chain is still progressing.

Default behaviour:
  1. Use all-year matched outputs if they exist.
  2. Otherwise use the available 2019 matched study.

Outputs are written to:
  final_analysis/jjas/plots/figs/jjas_publication_suite[_all_years]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
DEFAULT_2019 = FIGS / "matched_monsoon_study"
DEFAULT_ALL = FIGS / "matched_monsoon_study_all_years"

MODELS = ["FuXi", "ECMWF"]
VARIABLES = ["TP", "Z500"]
REGIONS = [
    "All India",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
]
REGION_LABELS = {
    "All India": "All India",
    "northwest_india": "Northwest",
    "central_india": "Central",
    "south_peninsula": "South Pen.",
    "east_northeast_india": "East & NE",
}
MODEL_COLORS = {"FuXi": "#D55E00", "ECMWF": "#009E73"}
VAR_UNITS = {"TP": "mm day$^{-1}$", "Z500": "gpm"}


def apply_theme():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def lead_week(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")


def choose_source(mode):
    if mode == "all-years":
        return DEFAULT_ALL
    if mode == "2019":
        return DEFAULT_2019
    all_metrics = DEFAULT_ALL / "matched_case_metrics_all_years.csv"
    if all_metrics.exists():
        return DEFAULT_ALL
    return DEFAULT_2019


def read_source(source):
    source = Path(source)
    if (source / "matched_case_metrics_all_years.csv").exists():
        metrics = pd.read_csv(source / "matched_case_metrics_all_years.csv")
        cases = pd.read_csv(source / "matched_cases_all_years.csv")
        delta = pd.read_csv(source / "matched_pairwise_deltas.csv")
        label = "all_years"
    else:
        metrics = pd.read_csv(source / "matched_case_metrics.csv")
        cases = pd.read_csv(source / "matched_cases.csv")
        delta = pd.read_csv(source / "matched_pairwise_deltas.csv")
        years = sorted(metrics["year"].unique()) if "year" in metrics else []
        label = str(years[0]) if len(years) == 1 else "matched"

    for df in (metrics, delta):
        if "lead_week" not in df.columns:
            df["lead_week"] = lead_week(df["week"]).astype(int)
    for col in ["fuxi_init", "ecmwf_init", "valid_start", "valid_end"]:
        if col in cases:
            cases[col] = pd.to_datetime(cases[col])
    return metrics, cases, delta, label


def bootstrap_stats(values, seed=42, n_boot=5000):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    win = float(np.mean(vals > 0))
    if len(vals) == 1 or n_boot <= 0:
        return mean, mean, mean, win
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi), win


def grouped_ci(df, keys, value, n_boot=5000):
    rows = []
    for group_key, sub in df.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        mean, lo, hi, win = bootstrap_stats(sub[value], n_boot=n_boot)
        row = dict(zip(keys, group_key))
        row.update(mean=mean, ci_low=lo, ci_high=hi,
                   win_fraction=win, n=int(pd.to_numeric(sub[value], errors="coerce").notna().sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")


def fig_lead_skill(metrics, out, tag):
    d = metrics[metrics.region == "All India"].copy()
    if d.empty:
        return []
    rows = []
    for metric in ["pcc", "rmse"]:
        g = grouped_ci(d, ["variable", "model", "lead_week"], metric)
        g["metric"] = metric
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)
    for col, var in enumerate(VARIABLES):
        for row, metric in enumerate(["pcc", "rmse"]):
            ax = axes[row, col]
            sub = g[(g.variable == var) & (g.metric == metric)]
            for model in MODELS:
                dm = sub[sub.model == model].sort_values("lead_week")
                if dm.empty:
                    continue
                x = dm.lead_week.to_numpy(float)
                y = dm["mean"].to_numpy(float)
                lo = dm["ci_low"].to_numpy(float)
                hi = dm["ci_high"].to_numpy(float)
                ax.plot(x, y, marker="o", lw=2.2, ms=5.5,
                        color=MODEL_COLORS[model], label=model)
                ax.fill_between(x, lo, hi, color=MODEL_COLORS[model], alpha=0.16, linewidth=0)
            ax.set_title(var)
            ax.set_xticks(range(1, 7), [f"W{i}" for i in range(1, 7)])
            if metric == "pcc":
                ax.set_ylim(0.35, 1.02)
                ax.set_ylabel("Pattern correlation")
                ax.axhline(0.5, color="0.35", lw=1.0, ls="--")
            else:
                ax.set_ylabel(f"RMSE ({VAR_UNITS[var]})")
            if row == 1:
                ax.set_xlabel("Lead week")
            if row == 0 and col == 1:
                ax.legend(loc="lower right")
            panel_label(ax, chr(ord("a") + row * 2 + col))
    fig.suptitle(f"Matched-valid-window JJAS skill ({tag.replace('_', ' ')})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / "01_lead_skill_polished.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_scorecard(metrics, out, tag):
    d = metrics.copy()
    g = (
        d.groupby(["variable", "model", "region", "lead_week"], as_index=False)
        .agg(pcc=("pcc", "mean"), n=("pcc", "count"))
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True, sharey=True)
    for i, var in enumerate(VARIABLES):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            sub = g[(g.variable == var) & (g.model == model)]
            mat = np.full((len(REGIONS), 6), np.nan)
            for r, region in enumerate(REGIONS):
                for week in range(1, 7):
                    hit = sub[(sub.region == region) & (sub.lead_week == week)]
                    if not hit.empty:
                        mat[r, week - 1] = hit.pcc.iloc[0]
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
            ax.set_title(f"{var} | {model}")
            ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
            ax.set_yticks(range(len(REGIONS)), [REGION_LABELS[r] for r in REGIONS])
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    if np.isfinite(val):
                        color = "white" if val > 0.68 else "black"
                        ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                                fontsize=8.5, fontweight="bold", color=color)
            panel_label(ax, chr(ord("a") + i * 2 + j))
    fig.subplots_adjust(right=0.88, top=0.88, hspace=0.25, wspace=0.10)
    cax = fig.add_axes([0.90, 0.22, 0.018, 0.56])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pattern correlation")
    fig.suptitle(f"Regional JJAS skill scorecard ({tag.replace('_', ' ')})",
                 fontsize=14, fontweight="bold")
    path = out / "02_regional_skill_scorecard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_paired_advantage(delta, out, tag):
    d = delta[delta.region == "All India"].copy()
    metrics = [
        ("pcc_gain_fuxi_minus_ecmwf", "PCC gain", "FuXi - ECMWF", ""),
        ("rmse_reduction_fuxi_vs_ecmwf", "RMSE reduction", "ECMWF - FuXi", ""),
        ("abs_bias_reduction_fuxi_vs_ecmwf", "|bias| reduction", "ECMWF - FuXi", ""),
    ]
    fig, axes = plt.subplots(len(VARIABLES), len(metrics), figsize=(14.5, 7.4), sharex=True)
    for row, var in enumerate(VARIABLES):
        sub_var = d[d.variable == var]
        for col, (metric, title, ylabel, _) in enumerate(metrics):
            ax = axes[row, col]
            if metric not in sub_var:
                ax.axis("off")
                continue
            stats = grouped_ci(sub_var, ["lead_week"], metric)
            stats = stats.sort_values("lead_week")
            x = stats.lead_week.to_numpy(float)
            y = stats["mean"].to_numpy(float)
            lo = stats["ci_low"].to_numpy(float)
            hi = stats["ci_high"].to_numpy(float)
            colors = np.where(y >= 0, "#0B7F79", "#B36B00")
            ax.bar(x, y, width=0.70, color=colors, edgecolor="white", linewidth=0.9)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none",
                        ecolor="0.15", lw=1.0, capsize=3)
            ax.axhline(0, color="0.20", lw=1.0)
            ax.set_title(f"{var} | {title}")
            ax.set_xticks(range(1, 7), [f"W{i}" for i in range(1, 7)])
            ax.set_ylabel(ylabel + (f" ({VAR_UNITS[var]})" if "rmse" in metric or "bias" in metric else ""))
            for r in stats.itertuples():
                ax.text(r.lead_week, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{100*r.win_fraction:.0f}%", ha="center", va="bottom", fontsize=8,
                        color="0.25")
            if row == len(VARIABLES) - 1:
                ax.set_xlabel("Lead week")
            panel_label(ax, chr(ord("a") + row * len(metrics) + col))
    fig.suptitle(
        f"Paired FuXi advantage by exact valid window ({tag.replace('_', ' ')}); labels = FuXi win rate",
        fontsize=13.5, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "03_paired_advantage_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_regional_gain_ci(delta, out, tag):
    if "pcc_gain_fuxi_minus_ecmwf" not in delta:
        return []
    stats = grouped_ci(delta, ["variable", "region", "lead_week"], "pcc_gain_fuxi_minus_ecmwf")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
    max_abs = max(0.05, float(np.nanpercentile(np.abs(stats["mean"]), 95)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs)
    im = None
    for i, var in enumerate(VARIABLES):
        ax = axes[i]
        mat = np.full((len(REGIONS), 6), np.nan)
        sig = np.zeros((len(REGIONS), 6), dtype=bool)
        wins = np.full((len(REGIONS), 6), np.nan)
        sub = stats[stats.variable == var]
        for r, region in enumerate(REGIONS):
            for week in range(1, 7):
                hit = sub[(sub.region == region) & (sub.lead_week == week)]
                if not hit.empty:
                    row = hit.iloc[0]
                    mat[r, week - 1] = row["mean"]
                    sig[r, week - 1] = (row["ci_low"] > 0) or (row["ci_high"] < 0)
                    wins[r, week - 1] = row["win_fraction"]
        im = ax.imshow(mat, cmap="BrBG", norm=norm, aspect="auto")
        ax.set_title(var)
        ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
        ax.set_yticks(range(len(REGIONS)), [REGION_LABELS[r] for r in REGIONS])
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                if np.isfinite(mat[r, c]):
                    star = "*" if sig[r, c] else ""
                    ax.text(c, r, f"{mat[r,c]:+.2f}{star}\n{100*wins[r,c]:.0f}%",
                            ha="center", va="center", fontsize=8.0, linespacing=0.9)
        panel_label(ax, chr(ord("a") + i))
    fig.subplots_adjust(right=0.88, top=0.82, bottom=0.14, wspace=0.08)
    cax = fig.add_axes([0.90, 0.22, 0.018, 0.48])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PCC gain (FuXi - ECMWF)")
    fig.suptitle(
        f"Regional paired PCC gain ({tag.replace('_', ' ')}): mean, win-rate, and bootstrap signal",
        fontsize=13.5, fontweight="bold",
    )
    fig.text(0.12, 0.035, "* 95% bootstrap CI excludes zero; second line in each cell is FuXi win rate.",
             fontsize=9, color="0.25")
    path = out / "04_regional_pcc_gain_with_ci.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_rain_intensity(metrics, out, tag):
    d = metrics[(metrics.variable == "TP") & (metrics.region == "All India")].copy()
    if d.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    markers = {"FuXi": "o", "ECMWF": "^"}
    for model in MODELS:
        sub = d[d.model == model]
        axes[0].scatter(sub.obs_mean, sub.fcst_mean, s=22 + 8 * sub.lead_week,
                        color=MODEL_COLORS[model], marker=markers[model],
                        alpha=0.72, edgecolor="white", linewidth=0.35, label=model)
        axes[1].scatter(sub.obs_mean, sub.pcc, s=22 + 8 * sub.lead_week,
                        color=MODEL_COLORS[model], marker=markers[model],
                        alpha=0.72, edgecolor="white", linewidth=0.35, label=model)
    lim = [
        0,
        max(float(d[["obs_mean", "fcst_mean"]].max().max()) * 1.08, 1.0),
    ]
    axes[0].plot(lim, lim, color="0.25", lw=1.1, ls="--")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("ERA5 All-India rainfall (mm day$^{-1}$)")
    axes[0].set_ylabel("Forecast rainfall (mm day$^{-1}$)")
    axes[0].set_title("Mean-rainfall calibration")
    axes[1].axhline(0.5, color="0.25", lw=1.0, ls="--")
    axes[1].set_xlabel("ERA5 All-India rainfall (mm day$^{-1}$)")
    axes[1].set_ylabel("Pattern correlation")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Skill under wetter monsoon windows")
    axes[1].legend(loc="lower left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(
        f"Rainfall-intensity conditioned JJAS forecast behaviour ({tag.replace('_', ' ')})",
        fontsize=13.5, fontweight="bold",
    )
    fig.text(0.50, 0.02, "Larger markers indicate later lead weeks.", ha="center", fontsize=9, color="0.30")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    path = out / "05_rainfall_intensity_conditioning.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_error_phase_space(metrics, out, tag):
    d = metrics[metrics.region == "All India"].copy()
    if d.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    for ax, var in zip(axes, VARIABLES):
        sub = d[d.variable == var]
        for model in MODELS:
            dm = sub[sub.model == model]
            sc = ax.scatter(dm.bias, dm.rmse, c=dm.lead_week, cmap="viridis",
                            vmin=1, vmax=6, s=48, marker="o" if model == "FuXi" else "^",
                            edgecolor="white", linewidth=0.45, alpha=0.82, label=model)
        ax.axvline(0, color="0.30", lw=1.0, ls="--")
        ax.set_xlabel(f"Mean bias ({VAR_UNITS[var]})")
        ax.set_ylabel(f"RMSE ({VAR_UNITS[var]})")
        ax.set_title(var)
        ax.legend(loc="upper left")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_ticks(range(1, 7))
    cbar.set_label("Lead week")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(f"Error phase space: bias drift vs amplitude error ({tag.replace('_', ' ')})",
                 fontsize=13.5, fontweight="bold")
    fig.savefig(out / "06_error_phase_space.png", bbox_inches="tight")
    plt.close(fig)
    return [out / "06_error_phase_space.png"]


def fig_case_ranking(delta, out, tag):
    d = delta[(delta.region == "All India") & (delta.variable == "TP")].copy()
    if d.empty:
        return []
    d["label"] = (
        pd.to_datetime(d["valid_start"]).dt.strftime("%d %b") + "-" +
        pd.to_datetime(d["valid_end"]).dt.strftime("%d %b") +
        " | W" + d["lead_week"].astype(str) +
        " | obs " + d["obs_mean_FuXi"].round(1).astype(str)
    )
    pick = pd.concat([
        d.nlargest(8, "pcc_gain_fuxi_minus_ecmwf"),
        d.nsmallest(8, "pcc_gain_fuxi_minus_ecmwf"),
    ], ignore_index=True).sort_values("pcc_gain_fuxi_minus_ecmwf")
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    colors = np.where(pick["pcc_gain_fuxi_minus_ecmwf"] >= 0, "#0B7F79", "#B36B00")
    ax.barh(range(len(pick)), pick["pcc_gain_fuxi_minus_ecmwf"], color=colors, edgecolor="white")
    ax.axvline(0, color="0.25", lw=1.0)
    ax.set_yticks(range(len(pick)), pick["label"])
    ax.set_xlabel("TP PCC gain (FuXi - ECMWF)")
    ax.set_title("Windows where FuXi most helps or hurts All-India rainfall skill")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.text(0.02, 0.02, "Labels show valid window, lead week, and ERA5 mean rainfall (mm day$^{-1}$).",
             fontsize=9, color="0.30")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = out / "07_top_bottom_tp_windows.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_calendar(cases, out, tag):
    c = cases.copy()
    if c.empty:
        return []
    c["lead_week"] = lead_week(c["week"]).astype(int)
    years = sorted(pd.to_datetime(c["valid_start"]).dt.year.unique())
    if len(years) == 1 and c["fuxi_init"].nunique() <= 35:
        init_order = sorted(c["fuxi_init"].dropna().unique())
        ymap = {init: i for i, init in enumerate(init_order)}
        fig, ax = plt.subplots(figsize=(11.2, max(4.8, 0.36 * len(init_order) + 1.8)))
        cmap = plt.get_cmap("turbo", 6)
        for row in c.itertuples():
            y = ymap[row.fuxi_init]
            color = cmap(row.lead_week - 1)
            ax.plot([row.valid_start, row.valid_end], [y, y], color=color, lw=6,
                    solid_capstyle="butt", alpha=0.86)
        ax.scatter(pd.to_datetime(init_order), [ymap[i] for i in init_order],
                   marker="o", s=20, color="#D55E00", label="FuXi init")
        ecmwf = c.drop_duplicates(["fuxi_init", "ecmwf_init"])
        ax.scatter(ecmwf["ecmwf_init"], ecmwf["fuxi_init"].map(ymap),
                   marker="^", s=24, color="#009E73", label="Matched ECMWF init")
        ax.set_yticks(range(len(init_order)), [pd.to_datetime(i).strftime("%d %b") for i in init_order])
        ax.set_xlabel("Calendar date")
        ax.set_ylabel("FuXi initialization")
        ax.set_title("Exact valid windows used for fair FuXi-ECMWF comparison")
        ax.legend(loc="upper left", ncol=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 6))
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
        cbar.set_ticks(np.linspace(1.4, 5.6, 6))
        cbar.set_ticklabels([f"W{i}" for i in range(1, 7)])
        fig.tight_layout()
        path = out / "08_valid_window_calendar.png"
    else:
        c["year"] = pd.to_datetime(c["valid_start"]).dt.year
        c["month"] = pd.to_datetime(c["valid_start"]).dt.month
        table = c.drop_duplicates(["year", "fuxi_init"]).pivot_table(
            index="year", columns="month", values="fuxi_init", aggfunc="count", fill_value=0
        )
        table = table.reindex(columns=[6, 7, 8, 9], fill_value=0)
        fig, ax = plt.subplots(figsize=(7.0, max(5.0, 0.28 * len(table) + 1.5)))
        im = ax.imshow(table.values, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(4), ["Jun", "Jul", "Aug", "Sep"])
        ax.set_yticks(range(len(table.index)), table.index.astype(str))
        ax.set_xlabel("Initialization month")
        ax.set_ylabel("Year")
        ax.set_title("JJAS FuXi initialization coverage")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, int(table.values[i, j]), ha="center", va="center",
                        color="white" if table.values[i, j] > table.values.max() * 0.55 else "black",
                        fontsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label("FuXi inits")
        fig.tight_layout()
        path = out / "08_valid_window_calendar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def write_report(out, source, metrics, cases, delta, paths):
    years = sorted(metrics["year"].unique()) if "year" in metrics else []
    ai = metrics[metrics.region == "All India"]
    lines = [
        "# JJAS Publication Plot Suite",
        "",
        f"Source: `{source}`",
        f"Years: {years[0]}-{years[-1]} ({len(years)} years)" if years else "Years: unavailable",
        f"Matched valid windows: {len(cases)}",
        f"Metric rows: {len(metrics)}",
        f"Paired comparison rows: {len(delta)}",
        "",
        "## Headline All-India Means",
    ]
    for var in VARIABLES:
        lines.append("")
        lines.append(f"### {var}")
        for model in MODELS:
            sub = ai[(ai.variable == var) & (ai.model == model)]
            if sub.empty:
                continue
            unit = VAR_UNITS[var].replace("$", "")
            lines.append(
                f"- {model}: PCC {sub.pcc.mean():.2f}; "
                f"RMSE {sub.rmse.mean():.2f} {unit}; bias {sub.bias.mean():+.2f}"
            )
    lines.extend([
        "",
        "## New Figures",
    ])
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend([
        "",
        "## Reading Notes",
        "- All model comparisons are matched by ERA5 valid date window.",
        "- Positive PCC gain means FuXi has higher pattern correlation than ECMWF.",
        "- Positive RMSE or absolute-bias reduction means FuXi has lower error.",
        "- Bootstrap intervals resample matched windows within each plotted group.",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto",
                        help="auto, 2019, all-years, or explicit matched-study output dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    apply_theme()
    source = choose_source(args.source) if args.source in {"auto", "2019", "all-years"} else Path(args.source)
    metrics, cases, delta, label = read_source(source)
    if args.out:
        out = Path(args.out)
    else:
        suffix = "all_years" if label == "all_years" else label
        out = FIGS / f"jjas_publication_suite_{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    paths += fig_lead_skill(metrics, out, label)
    paths += fig_scorecard(metrics, out, label)
    paths += fig_paired_advantage(delta, out, label)
    paths += fig_regional_gain_ci(delta, out, label)
    paths += fig_rain_intensity(metrics, out, label)
    paths += fig_error_phase_space(metrics, out, label)
    paths += fig_case_ranking(delta, out, label)
    paths += fig_calendar(cases, out, label)
    write_report(out, source, metrics, cases, delta, paths)
    print(f"WROTE {len(paths)} polished JJAS figures -> {out}")


if __name__ == "__main__":
    main()
