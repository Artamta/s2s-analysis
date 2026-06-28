#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
statistical_comparison_pack.py
==============================
Meeting-ready paired statistical comparison plots for JJAS.

The goal is not just "nice plots"; it is defensible model comparison:
  - same ERA5 valid date window for FuXi and ECMWF
  - paired deltas: FuXi - ECMWF for PCC, ECMWF - FuXi for error reductions
  - bootstrap confidence intervals over matched windows
  - Wilcoxon/sign-test p-values and Benjamini-Hochberg FDR q-values
  - clear statements of what is statistically supported

The script uses all-year matched outputs if present; otherwise it uses the 2019
matched study. Current caveat for 2019: inference is over matched windows within
one monsoon season, so it supports a case-study comparison, not climatological
generality.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
DEFAULT_2019 = FIGS / "matched_monsoon_study"
DEFAULT_ALL = FIGS / "matched_monsoon_study_all_years"

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
METRICS = [
    ("pcc_gain_fuxi_minus_ecmwf", "PCC gain", "FuXi - ECMWF", ""),
    ("rmse_reduction_fuxi_vs_ecmwf", "RMSE reduction", "ECMWF - FuXi", ""),
    ("abs_bias_reduction_fuxi_vs_ecmwf", "|bias| reduction", "ECMWF - FuXi", ""),
]
UNITS = {"TP": "mm day$^{-1}$", "Z500": "gpm"}
MODEL_COLORS = {"FuXi": "#D55E00", "ECMWF": "#009E73"}


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


def choose_source(source):
    if source == "all-years":
        return DEFAULT_ALL
    if source == "2019":
        return DEFAULT_2019
    if source != "auto":
        return Path(source)
    if (DEFAULT_ALL / "matched_pairwise_deltas.csv").exists() and (
        DEFAULT_ALL / "matched_case_metrics_all_years.csv"
    ).exists():
        return DEFAULT_ALL
    return DEFAULT_2019


def read_source(source):
    source = Path(source)
    if (source / "matched_case_metrics_all_years.csv").exists():
        metrics = pd.read_csv(source / "matched_case_metrics_all_years.csv")
        cases = pd.read_csv(source / "matched_cases_all_years.csv")
        delta = pd.read_csv(source / "matched_pairwise_deltas.csv")
        tag = "all_years"
    else:
        metrics = pd.read_csv(source / "matched_case_metrics.csv")
        cases = pd.read_csv(source / "matched_cases.csv")
        delta = pd.read_csv(source / "matched_pairwise_deltas.csv")
        years = sorted(metrics["year"].unique()) if "year" in metrics.columns else []
        tag = str(years[0]) if len(years) == 1 else "matched"
    for df in (metrics, delta):
        if "lead_week" not in df:
            df["lead_week"] = pd.to_numeric(
                df["week"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            ).astype(int)
    for col in ["valid_start", "valid_end", "fuxi_init", "ecmwf_init"]:
        if col in delta:
            delta[col] = pd.to_datetime(delta[col])
        if col in cases:
            cases[col] = pd.to_datetime(cases[col])
    return metrics, cases, delta, tag


def bootstrap_ci(values, n_boot=10000, seed=7):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, median=np.nan, ci_low=np.nan, ci_high=np.nan,
                    win_rate=np.nan, sign_p=np.nan, wilcoxon_p=np.nan)
    mean = float(np.mean(x))
    median = float(np.median(x))
    win_rate = float(np.mean(x > 0))
    sign_p = float(stats.binomtest(int(np.sum(x > 0)), n, p=0.5).pvalue)
    if np.allclose(x, 0):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
    if n == 1:
        lo = hi = mean
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = x[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, mean=mean, median=median, ci_low=float(lo), ci_high=float(hi),
                win_rate=win_rate, sign_p=sign_p, wilcoxon_p=wilcoxon_p)


def bh_fdr(p_values):
    """Benjamini-Hochberg q-values, equivalent to statsmodels fdr_bh."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def make_stats(delta, n_boot):
    rows = []
    groupings = {
        "overall_region": ["variable", "region"],
        "lead_region": ["variable", "region", "lead_week"],
        "all_india_lead": ["variable", "lead_week"],
    }
    for scope, keys in groupings.items():
        for key, sub in delta.groupby(keys, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(keys, key))
            base["scope"] = scope
            for metric, label, direction, _ in METRICS:
                if metric not in sub:
                    continue
                row = {**base, "metric": metric, "metric_label": label,
                       "positive_direction": direction}
                row.update(bootstrap_ci(sub[metric], n_boot=n_boot))
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        p = out["wilcoxon_p"].fillna(1.0).to_numpy()
        out["wilcoxon_q_bh"] = bh_fdr(p)
        p2 = out["sign_p"].fillna(1.0).to_numpy()
        out["sign_q_bh"] = bh_fdr(p2)
        out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
        out["fuxi_better_mean"] = out["mean"] > 0
    return out


def sig_marker(q, ci):
    if not ci:
        return ""
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "†"


def panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="right", va="bottom")


def fig_forest(stats_df, out, tag):
    d = stats_df[stats_df.scope == "overall_region"].copy()
    d = d[d.metric.isin([m[0] for m in METRICS])]
    paths = []
    for metric, label, direction, _ in METRICS:
        sub = d[d.metric == metric].copy()
        if sub.empty:
            continue
        sub["row"] = sub["variable"] + " | " + sub["region"].map(REGION_LABELS)
        order = []
        for var in VARIABLES:
            for region in REGIONS:
                name = f"{var} | {REGION_LABELS[region]}"
                if name in set(sub["row"]):
                    order.append(name)
        sub["row"] = pd.Categorical(sub["row"], categories=order[::-1], ordered=True)
        sub = sub.sort_values("row")
        fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.33 * len(sub) + 1.2)))
        y = np.arange(len(sub))
        colors = np.where(sub["mean"] >= 0, "#0B7F79", "#B36B00")
        x_min = float(np.nanmin(sub["ci_low"]))
        x_max = float(np.nanmax(sub["ci_high"]))
        span = max(x_max - x_min, 0.01)
        ax.set_xlim(x_min - 0.08 * span, x_max + 0.32 * span)
        ax.errorbar(sub["mean"], y,
                    xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
                    fmt="none", ecolor="0.20", elinewidth=1.0, capsize=3, zorder=1)
        ax.scatter(sub["mean"], y, s=70, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
        ax.axvline(0, color="0.25", lw=1.0)
        ax.set_yticks(y, sub["row"])
        unit = ""
        if "rmse" in metric or "bias" in metric:
            unit = " (TP: mm day$^{-1}$; Z500: gpm)"
        ax.set_xlabel(f"{label}: {direction}{unit}")
        ax.set_title(f"Paired effect size with 95% bootstrap CI ({tag.replace('_', ' ')})")
        for yi, row in enumerate(sub.itertuples()):
            mark = sig_marker(row.wilcoxon_q_bh, row.ci_excludes_zero)
            ax.text(row.ci_high + 0.015 * span, yi,
                    f"{mark} {100*row.win_rate:.0f}% wins",
                    va="center", fontsize=8.5, color="0.25", ha="left")
        fig.text(0.02, 0.02,
                 "*, **: 95% bootstrap CI excludes zero and Wilcoxon BH-FDR q<0.05/0.01; †: CI excludes zero only.",
                 fontsize=8.5, color="0.30")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        path = out / f"01_forest_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def fig_fdr_heatmap(stats_df, out, tag):
    d = stats_df[stats_df.scope == "lead_region"].copy()
    paths = []
    for metric, label, direction, _ in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
        subm = d[d.metric == metric]
        vmax = max(0.05, float(np.nanpercentile(np.abs(subm["mean"]), 95)))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = None
        for ax, var in zip(axes, VARIABLES):
            sub = subm[subm.variable == var]
            mat = np.full((len(REGIONS), 6), np.nan)
            text = [["" for _ in range(6)] for _ in REGIONS]
            for i, region in enumerate(REGIONS):
                for week in range(1, 7):
                    row = sub[(sub.region == region) & (sub.lead_week == week)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    mat[i, week - 1] = row["mean"]
                    mark = sig_marker(row["wilcoxon_q_bh"], row["ci_excludes_zero"])
                    text[i][week - 1] = f"{row['mean']:+.2f}{mark}\n{100*row['win_rate']:.0f}%"
            im = ax.imshow(mat, cmap="BrBG", norm=norm, aspect="auto")
            ax.set_title(var)
            ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
            ax.set_yticks(range(len(REGIONS)), [REGION_LABELS[r] for r in REGIONS])
            for i in range(len(REGIONS)):
                for j in range(6):
                    if text[i][j]:
                        ax.text(j, i, text[i][j], ha="center", va="center",
                                fontsize=7.8, linespacing=0.9)
        fig.subplots_adjust(right=0.88, top=0.82, bottom=0.14, wspace=0.08)
        cax = fig.add_axes([0.90, 0.22, 0.018, 0.48])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{label}: {direction}")
        fig.suptitle(f"{label} by region and lead ({tag.replace('_', ' ')}): mean delta + win rate",
                     fontsize=13.5, fontweight="bold")
        fig.text(0.08, 0.035,
                 "Positive cells favour FuXi. *, **: CI excludes zero and Wilcoxon BH-FDR q<0.05/0.01; †: CI excludes zero only.",
                 fontsize=8.5, color="0.30")
        path = out / f"02_fdr_heatmap_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def fig_paired_scatter(delta, out, tag):
    d = delta[delta.region == "All India"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
    for col, var in enumerate(VARIABLES):
        sub = d[d.variable == var]
        ax = axes[0, col]
        ax.scatter(sub["pcc_ECMWF"], sub["pcc_FuXi"], c=sub["lead_week"], cmap="viridis",
                   vmin=1, vmax=6, s=42, edgecolor="white", linewidth=0.5, alpha=0.86)
        lim = [min(sub["pcc_ECMWF"].min(), sub["pcc_FuXi"].min(), 0.0),
               max(sub["pcc_ECMWF"].max(), sub["pcc_FuXi"].max(), 1.0)]
        ax.plot(lim, lim, color="0.25", ls="--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{var} PCC")
        ax.set_xlabel("ECMWF")
        ax.set_ylabel("FuXi")
        panel_label(ax, chr(ord("a") + col))
        ax = axes[1, col]
        ax.scatter(sub["rmse_ECMWF"], sub["rmse_FuXi"], c=sub["lead_week"], cmap="viridis",
                   vmin=1, vmax=6, s=42, edgecolor="white", linewidth=0.5, alpha=0.86)
        lim = [0, max(sub["rmse_ECMWF"].max(), sub["rmse_FuXi"].max()) * 1.05]
        ax.plot(lim, lim, color="0.25", ls="--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{var} RMSE")
        ax.set_xlabel(f"ECMWF ({UNITS[var]})")
        ax.set_ylabel(f"FuXi ({UNITS[var]})")
        panel_label(ax, chr(ord("c") + col))
    cbar = fig.colorbar(axes[0, 0].collections[0], ax=axes, shrink=0.88, pad=0.02)
    cbar.set_ticks(range(1, 7))
    cbar.set_label("Lead week")
    fig.suptitle(f"Paired window-by-window comparison ({tag.replace('_', ' ')}): points above/below diagonal show winner",
                 fontsize=13.5, fontweight="bold")
    path = out / "03_paired_scatter_pcc_rmse.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_delta_distributions(delta, out, tag):
    d = delta[delta.region == "All India"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=False)
    for row, var in enumerate(VARIABLES):
        sub = d[d.variable == var]
        for col, (metric, label, direction, _) in enumerate(METRICS):
            ax = axes[row, col]
            vals = [sub[sub.lead_week == week][metric].dropna().to_numpy() for week in range(1, 7)]
            parts = ax.violinplot(vals, positions=np.arange(1, 7), widths=0.72,
                                  showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#92C5DE")
                body.set_edgecolor("0.25")
                body.set_alpha(0.82)
            parts["cmeans"].set_color("0.10")
            parts["cmeans"].set_linewidth(1.2)
            ax.axhline(0, color="0.25", lw=1.0)
            ax.set_xticks(range(1, 7), [f"W{i}" for i in range(1, 7)])
            ax.set_title(f"{var} | {label}")
            ax.set_ylabel(direction)
            if row == 1:
                ax.set_xlabel("Lead week")
            panel_label(ax, chr(ord("a") + row * 3 + col))
    fig.suptitle(f"Distribution of paired FuXi-ECMWF deltas by lead ({tag.replace('_', ' ')})",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "04_delta_distribution_by_lead.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_rainfall_terciles(metrics, out, tag):
    d = metrics[(metrics.variable == "TP") & (metrics.region == "All India")].copy()
    if d.empty:
        return []
    # Terciles are assigned by valid-window observed rainfall, shared by both models.
    key_cols = ["fuxi_init", "valid_start", "valid_end", "week"]
    obs = d.drop_duplicates(key_cols)[key_cols + ["obs_mean"]].copy()
    obs["rainfall_tercile"] = pd.qcut(obs["obs_mean"], 3, labels=["Dry", "Normal", "Wet"])
    d = d.merge(obs[key_cols + ["rainfall_tercile"]], on=key_cols, how="left")
    rows = []
    for (tercile, model), sub in d.groupby(["rainfall_tercile", "model"], observed=False):
        for metric in ["pcc", "rmse", "bias"]:
            s = bootstrap_ci(sub[metric], n_boot=5000)
            rows.append(dict(tercile=str(tercile), model=model, metric=metric, **s))
    g = pd.DataFrame(rows)
    order = ["Dry", "Normal", "Wet"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharex=True)
    for ax, metric in zip(axes, ["pcc", "rmse", "bias"]):
        sub = g[g.metric == metric]
        x = np.arange(len(order))
        width = 0.36
        for i, model in enumerate(["FuXi", "ECMWF"]):
            dm = sub[sub.model == model].set_index("tercile").reindex(order)
            offset = (i - 0.5) * width
            ax.bar(x + offset, dm["mean"], width=width, color=MODEL_COLORS[model],
                   edgecolor="white", label=model)
            ax.errorbar(x + offset, dm["mean"],
                        yerr=[dm["mean"] - dm["ci_low"], dm["ci_high"] - dm["mean"]],
                        fmt="none", ecolor="0.15", lw=1.0, capsize=3)
        ax.set_xticks(x, order)
        ax.set_title(metric.upper() if metric != "pcc" else "PCC")
        if metric == "pcc":
            ax.set_ylabel("Pattern correlation")
            ax.set_ylim(0, 1.02)
            ax.axhline(0.5, color="0.35", lw=1.0, ls="--")
        elif metric == "rmse":
            ax.set_ylabel("RMSE (mm day$^{-1}$)")
        else:
            ax.set_ylabel("Bias (mm day$^{-1}$)")
            ax.axhline(0, color="0.35", lw=1.0)
        panel_label(ax, chr(ord("a") + list(["pcc", "rmse", "bias"]).index(metric)))
    axes[0].legend(loc="lower left")
    fig.suptitle(f"All-India rainfall-tercile conditioned TP skill ({tag.replace('_', ' ')})",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out / "05_tp_skill_by_rainfall_tercile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]


def fig_claim_matrix(stats_df, out, tag):
    d = stats_df[(stats_df.scope == "overall_region") & (stats_df.region == "All India")].copy()
    if d.empty:
        return []
    rows = []
    for var in VARIABLES:
        for metric, label, _, _ in METRICS:
            row = d[(d.variable == var) & (d.metric == metric)].iloc[0]
            if row["ci_low"] > 0 and row["wilcoxon_q_bh"] < 0.05:
                verdict = "FuXi better"
            elif row["ci_high"] < 0 and row["wilcoxon_q_bh"] < 0.05:
                verdict = "ECMWF better"
            elif row["ci_low"] > 0:
                verdict = "FuXi suggestive"
            elif row["ci_high"] < 0:
                verdict = "ECMWF suggestive"
            else:
                verdict = "No robust difference"
            rows.append((var, label, row["mean"], row["ci_low"], row["ci_high"],
                         row["win_rate"], row["wilcoxon_q_bh"], verdict))
    table = pd.DataFrame(rows, columns=["Variable", "Comparison", "Mean", "CI low", "CI high",
                                        "FuXi win rate", "BH q", "Verdict"])
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")
    cell_text = []
    for _, r in table.iterrows():
        cell_text.append([
            r["Variable"],
            r["Comparison"],
            f"{r['Mean']:+.2f} [{r['CI low']:+.2f}, {r['CI high']:+.2f}]",
            f"{100*r['FuXi win rate']:.0f}%",
            f"{r['BH q']:.3f}",
            r["Verdict"],
        ])
    cols = ["Var", "Metric", "Mean delta [95% CI]", "Win", "q", "Defensible statement"]
    tab = ax.table(cellText=cell_text, colLabels=cols, loc="center",
                   cellLoc="center", colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.55)
    for (i, j), cell in tab.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#23415A")
        elif j == 5:
            txt = cell.get_text().get_text()
            if "FuXi better" in txt:
                cell.set_facecolor("#C7EAE5")
            elif "ECMWF better" in txt:
                cell.set_facecolor("#F6E8C3")
            elif "suggestive" in txt:
                cell.set_facecolor("#EFEFEF")
    ax.set_title(f"Meeting claim matrix: All-India paired JJAS comparison ({tag.replace('_', ' ')})",
                 fontsize=13.0, fontweight="bold", pad=12)
    path = out / "06_meeting_claim_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path], table


def write_report(out, source, tag, metrics, cases, delta, stats_df, claim_table, paths):
    years = sorted(metrics["year"].unique()) if "year" in metrics else []
    if len(years) == 1:
        caveat = (
            "This is a statistically paired 2019 case-study result. It is defensible "
            "for the 2019 JJAS windows, but do not claim climatological superiority "
            "until the all-year pipeline finishes."
        )
    else:
        caveat = (
            "This uses multiple years of matched windows. Remaining dependence between "
            "overlapping windows should still be stated, but it is much stronger than "
            "single-year evidence."
        )
    lines = [
        "# Defensible JJAS Statistical Comparison Pack",
        "",
        f"Source: `{source}`",
        f"Years: {years[0]}-{years[-1]} ({len(years)} years)" if years else "Years: unknown",
        f"Matched valid windows: {len(cases)}",
        f"Paired region-variable rows: {len(delta)}",
        "",
        "## What You Can Say In The Meeting",
        caveat,
        "",
    ]
    for _, r in claim_table.iterrows():
        lines.append(
            f"- **{r['Variable']} {r['Comparison']}**: mean delta {r['Mean']:+.2f} "
            f"[{r['CI low']:+.2f}, {r['CI high']:+.2f}], "
            f"FuXi win-rate {100*r['FuXi win rate']:.0f}%, "
            f"BH-FDR q={r['BH q']:.3f} -> **{r['Verdict']}**."
        )
    lines.extend([
        "",
        "## Statistical Design",
        "- Pairing unit: same ERA5 valid date window.",
        "- PCC delta: FuXi - ECMWF, so positive means FuXi better.",
        "- RMSE and |bias| reduction: ECMWF - FuXi, so positive means FuXi lower-error.",
        "- Confidence interval: bootstrap over matched windows.",
        "- Tests: Wilcoxon signed-rank plus sign-test; q-values use Benjamini-Hochberg FDR.",
        "",
        "## Figures",
    ])
    for p in paths:
        lines.append(f"- `{p.name}`")
    lines.append("")
    (out / "MEETING_DEFENSE.md").write_text("\n".join(lines))
    stats_df.to_csv(out / "paired_statistical_tests.csv", index=False)
    claim_table.to_csv(out / "meeting_claim_matrix.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    args = ap.parse_args()

    apply_theme()
    source = choose_source(args.source)
    metrics, cases, delta, tag = read_source(source)
    out = Path(args.out) if args.out else FIGS / f"jjas_statistical_comparison_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    stats_df = make_stats(delta, n_boot=args.bootstrap_samples)
    paths = []
    paths += fig_forest(stats_df, out, tag)
    paths += fig_fdr_heatmap(stats_df, out, tag)
    paths += fig_paired_scatter(delta, out, tag)
    paths += fig_delta_distributions(delta, out, tag)
    paths += fig_rainfall_terciles(metrics, out, tag)
    claim_paths, claim_table = fig_claim_matrix(stats_df, out, tag)
    paths += claim_paths
    write_report(out, source, tag, metrics, cases, delta, stats_df, claim_table, paths)
    print(f"WROTE defensible statistical comparison pack -> {out}")
    print(f"figures={len(paths)}")


if __name__ == "__main__":
    main()
