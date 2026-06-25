#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_matched_monsoon_studies.py
====================================
Aggregate per-year matched JJAS monsoon verification outputs into one
publication-level multi-year summary.

This script does not reopen raw model/truth fields. It reads the yearly
`matched_case_metrics.csv` and `matched_cases.csv` files produced by
matched_monsoon_study.py, concatenates them, and regenerates compact lead,
paired-delta, and regional figures.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matched_monsoon_study as M


BOOT_METRICS = {
    "pcc_gain_fuxi_minus_ecmwf": "PCC gain (FuXi - ECMWF)",
    "rmse_reduction_fuxi_vs_ecmwf": "RMSE reduction (ECMWF - FuXi)",
    "abs_bias_reduction_fuxi_vs_ecmwf": "Abs bias reduction (ECMWF - FuXi)",
}


def read_year_outputs(base, years):
    metrics, cases = [], []
    missing = []
    for year in years:
        root = base / f"matched_monsoon_study_{year}"
        mp = root / "matched_case_metrics.csv"
        cp = root / "matched_cases.csv"
        if not (mp.exists() and cp.exists()):
            missing.append(year)
            continue
        metrics.append(pd.read_csv(mp))
        cases.append(pd.read_csv(cp))
    if missing:
        raise SystemExit(f"missing yearly outputs for: {missing}")
    return pd.concat(metrics, ignore_index=True), pd.concat(cases, ignore_index=True)


def bootstrap_mean_ci(values, rng, n_boot):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = len(vals)
    if n == 0:
        return {
            "n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "p_positive": np.nan, "p_two_sided": np.nan, "win_fraction": np.nan,
        }
    mean = float(np.mean(vals))
    win_fraction = float(np.mean(vals > 0))
    if n == 1 or n_boot <= 0:
        return {
            "n": n, "mean": mean, "ci_low": mean, "ci_high": mean,
            "p_positive": float(mean > 0), "p_two_sided": np.nan,
            "win_fraction": win_fraction,
        }
    samples = rng.integers(0, n, size=(n_boot, n))
    boot = vals[samples].mean(axis=1)
    p_positive = float(np.mean(boot > 0))
    p_two_sided = float(2 * min(p_positive, 1 - p_positive))
    return {
        "n": n,
        "mean": mean,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_positive": p_positive,
        "p_two_sided": p_two_sided,
        "win_fraction": win_fraction,
    }


def make_bootstrap_ci(delta, out, n_boot=10000, seed=42):
    rows = []
    rng = np.random.default_rng(seed)
    keys = ["variable", "region", "lead_week"]
    for group_key, sub in delta.groupby(keys, dropna=False):
        base = dict(zip(keys, group_key))
        for metric, label in BOOT_METRICS.items():
            if metric not in sub:
                continue
            stats = bootstrap_mean_ci(sub[metric], rng, n_boot)
            rows.append({**base, "metric": metric, "metric_label": label, **stats})
    boot = pd.DataFrame(rows)
    boot.to_csv(out / "matched_pairwise_bootstrap_ci.csv", index=False)
    return boot


def fig_bootstrap_pcc_gain(boot, out):
    if boot.empty:
        return
    d = boot[boot.metric == "pcc_gain_fuxi_minus_ecmwf"].copy()
    if d.empty:
        return
    variables = [v for v in M.VAR_ORDER if v in set(d.variable)]
    regions = [r for r in M.REGION_ORDER if r in set(d.region)]
    vmax = np.nanpercentile(np.abs(d["mean"]), 95)
    vmax = max(0.05, float(vmax)) if np.isfinite(vmax) else 0.2
    fig, axes = M.plt.subplots(1, len(variables), figsize=(7.2 * len(variables), 5.7),
                               squeeze=False, sharey=True)
    for col, (ax, var) in enumerate(zip(axes[0], variables)):
        mat = np.full((len(regions), 6), np.nan)
        sig = np.zeros((len(regions), 6), dtype=bool)
        sub = d[d.variable == var]
        for i, region in enumerate(regions):
            for week in range(1, 7):
                row = sub[(sub.region == region) & (sub.lead_week == week)]
                if row.empty:
                    continue
                row = row.iloc[0]
                mat[i, week - 1] = row["mean"]
                sig[i, week - 1] = (row["ci_low"] > 0) or (row["ci_high"] < 0)
        im = ax.imshow(mat, aspect="auto", cmap="BrBG", vmin=-vmax, vmax=vmax)
        ax.set_title(var)
        ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
        ax.set_yticks(range(len(regions)), [M.REGION_LABELS.get(r, r) for r in regions])
        ax.tick_params(labelleft=(col == 0))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    mark = "*" if sig[i, j] else ""
                    ax.text(j, i, f"{mat[i, j]:+.2f}{mark}", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.82, bottom=0.14, wspace=0.10)
    cax = fig.add_axes([0.885, 0.22, 0.018, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PCC gain (FuXi - ECMWF)")
    fig.suptitle("Regional PCC gain with paired-bootstrap significance", fontweight="bold")
    fig.text(0.16, 0.06, "* 95% bootstrap CI excludes zero", fontsize=9)
    fig.savefig(out / "matched_pcc_gain_bootstrap_significance.png", bbox_inches="tight", dpi=260)
    M.plt.close(fig)


def write_summary(df, cases, delta, boot, out):
    ai = df[df.region == "All India"]
    years = sorted(cases["year"].unique())
    fuxi_inits = cases["fuxi_init"].nunique()
    lines = [
        "# Multi-Year Matched-Valid-Window JJAS Monsoon Study",
        "",
        f"Years: {years[0]}-{years[-1]} ({len(years)} years)",
        f"FuXi init dates scored: {fuxi_inits}",
        f"Matched forecast windows scored: {len(cases)}",
        f"Metric rows: {len(df)}",
        "",
        "## All-India Mean Skill",
    ]
    for var in [v for v in M.VAR_ORDER if v in set(ai.variable)]:
        lines.extend(["", f"### {var}"])
        for model in ["FuXi", "ECMWF"]:
            sub = ai[(ai.variable == var) & (ai.model == model)]
            if sub.empty:
                continue
            unit = "mm/day" if var == "TP" else "gpm"
            lines.append(
                f"- {model}: PCC {sub.pcc.mean():.2f}, "
                f"RMSE {sub.rmse.mean():.2f} {unit}, bias {sub.bias.mean():+.2f}"
            )

    lines.extend(["", "## Paired FuXi Advantage"])
    dai = delta[delta.region == "All India"]
    for var in [v for v in M.VAR_ORDER if v in set(dai.variable)]:
        sub = dai[dai.variable == var]
        if sub.empty:
            continue
        unit = "mm/day" if var == "TP" else "gpm"
        lines.append(
            f"- {var}: mean PCC gain {sub.pcc_gain_fuxi_minus_ecmwf.mean():+.2f}; "
            f"mean RMSE reduction {sub.rmse_reduction_fuxi_vs_ecmwf.mean():+.2f} {unit}; "
            f"FuXi wins {100 * (sub.pcc_gain_fuxi_minus_ecmwf > 0).mean():.0f}% "
            "of All-India paired windows by PCC."
        )

    if boot is not None and not boot.empty:
        lines.extend(["", "## Paired Bootstrap Confidence"])
        bai = boot[(boot.region == "All India") & (boot.metric == "pcc_gain_fuxi_minus_ecmwf")]
        for var in [v for v in M.VAR_ORDER if v in set(bai.variable)]:
            sub = bai[bai.variable == var].sort_values("lead_week")
            pieces = []
            for row in sub.itertuples():
                pieces.append(
                    f"W{int(row.lead_week)} {row.mean:+.2f} "
                    f"[{row.ci_low:+.2f}, {row.ci_high:+.2f}]"
                )
            if pieces:
                lines.append(f"- {var} All-India PCC gain 95% CI: " + "; ".join(pieces))

    lines.extend([
        "",
        "## Calendar Handling",
        "FuXi and ECMWF initialization dates are offset. Metrics match forecasts",
        "by ERA5 valid date window, not by exact initialization date.",
    ])
    (out / "MULTIYEAR_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2002)
    p.add_argument("--end-year", type=int, default=2021)
    p.add_argument("--base", default=str(HERE / "figs"))
    p.add_argument("--out", default=str(HERE / "figs" / "matched_monsoon_study_all_years"))
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    M.set_plot_style()
    df, cases = read_year_outputs(base, years)
    df.to_csv(out / "matched_case_metrics_all_years.csv", index=False)
    cases.to_csv(out / "matched_cases_all_years.csv", index=False)
    skill, delta = M.make_summary_tables(df, out)
    boot = make_bootstrap_ci(delta, out, n_boot=args.bootstrap_samples, seed=args.seed)
    M.fig_pcc_lines(df, out)
    M.fig_rmse_lines(df, out)
    M.fig_skill_delta_bars(delta, out)
    M.fig_region_pcc_gain(delta, out)
    fig_bootstrap_pcc_gain(boot, out)
    write_summary(df, cases, delta, boot, out)
    print(f"WROTE all-year summary -> {out}")
    print(f"years={len(years)} cases={len(cases)} rows={len(df)}")


if __name__ == "__main__":
    main()
