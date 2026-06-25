#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matched_monsoon_study.py
========================
Matched-valid-window JJAS monsoon study for FuXi-S2S, ECMWF-S2S, and ERA5.

FuXi and ECMWF hindcast initialization calendars are offset. This script avoids
an unfair same-lead comparison by matching forecasts on the ERA5 valid dates:

  FuXi init + lead window -> valid date window
  ECMWF init + adjusted lead window -> the exact same valid date window

For every compact FuXi init available under /storage/.../jjas/fuxi_combined, it
computes TP and Z500 metrics over All-India and the four IMD homogeneous regions.

Outputs
-------
  matched_case_metrics.csv
  matched_cases.csv
  matched_skill_by_lead.csv
  matched_pairwise_deltas.csv
  matched_pcc_heatmap.png
  matched_bias_heatmap.png
  matched_pcc_lines.png
  matched_rmse_lines.png
  matched_skill_delta_bars.png
  matched_pcc_gain_regions.png
  STUDY_SUMMARY.md

Run
---
conda run --no-capture-output -n s2s-hind python -u matched_monsoon_study.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JJAS_DIR = HERE.parent
FA_ROOT = JJAS_DIR.parent
sys.path[:0] = [str(FA_ROOT), str(JJAS_DIR)]

from core import Physics
from core import grid as G
from core import truth as T
from core.adapters import get_adapter
from core.config import REGIONS, WEEKS
from core.metrics import rmse as w_rmse, bias as w_bias
import adapters_jjas  # noqa: F401
import adapters_fuxi  # noqa: F401
from config import DATA_ROOT, build_config

MODEL_COLORS = {"FuXi": "#D55E00", "ECMWF": "#009E73"}
VAR_ORDER = ["TP", "Z500"]
REGION_ORDER = ["All India", "northwest_india", "central_india",
                "south_peninsula", "east_northeast_india"]
REGION_LABELS = {
    "All India": "All India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}
_TRUTH_CACHE = {}


def set_plot_style():
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 10,
        "savefig.facecolor": "white",
    })


def compact_fuxi_inits(year):
    root = Path(DATA_ROOT) / "fuxi_combined"
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob(f"{year}*.nc")):
        s = path.stem
        out.append(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return out


def lead_for_window(init, valid_start, valid_end):
    ds = (pd.to_datetime(valid_start) - pd.to_datetime(init)).days
    de = (pd.to_datetime(valid_end) - pd.to_datetime(init)).days
    return ds, de


def find_ecmwf_match(cfg, fuxi_init, ds, de):
    valid_start = (pd.to_datetime(fuxi_init) + pd.Timedelta(days=ds)).strftime("%Y-%m-%d")
    valid_end = (pd.to_datetime(fuxi_init) + pd.Timedelta(days=de)).strftime("%Y-%m-%d")
    candidates = []
    for init in cfg.init_dates:
        eds, ede = lead_for_window(init, valid_start, valid_end)
        if 1 <= eds <= ede <= 42:
            candidates.append((init, eds, ede))
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(pd.to_datetime(x[0]) - pd.to_datetime(fuxi_init)))


def weighted_corr(a, b, w):
    """Cosine-weighted spatial correlation over finite grid cells."""
    av, bv, wv = a.values, b.values, w.values
    if wv.ndim == 1:
        wv = np.broadcast_to(wv[:, None], av.shape)
    ok = np.isfinite(av) & np.isfinite(bv) & np.isfinite(wv)
    if ok.sum() < 3:
        return np.nan
    x, y, ww = av[ok], bv[ok], wv[ok]
    ww = ww / np.nansum(ww)
    xm, ym = np.nansum(ww * x), np.nansum(ww * y)
    xa, ya = x - xm, y - ym
    den = np.sqrt(np.nansum(ww * xa * xa) * np.nansum(ww * ya * ya))
    return float(np.nansum(ww * xa * ya) / den) if den > 0 else np.nan


def region_mean(da, w):
    try:
        return float(da.weighted(w).mean(["lat", "lon"]).item())
    except Exception:
        return np.nan


def score_region(var, model, field, obs, rg, GC, week_label, case):
    rda = G.region_da(rg, GC)
    f = field.where(rda)
    o = obs.where(rda)
    w = GC["weights"]
    return dict(
        year=case["year"],
        fuxi_init=case["fuxi_init"],
        ecmwf_init=case["ecmwf_init"],
        valid_start=case["valid_start"],
        valid_end=case["valid_end"],
        model=model,
        variable=var,
        region=rg,
        week=week_label,
        fuxi_lead_start=case["fuxi_ds"],
        fuxi_lead_end=case["fuxi_de"],
        ecmwf_lead_start=case["ecmwf_ds"],
        ecmwf_lead_end=case["ecmwf_de"],
        pcc=weighted_corr(f, o, w),
        rmse=w_rmse(f, o, w),
        bias=w_bias(f, o, w),
        obs_mean=region_mean(o, w),
        fcst_mean=region_mean(f, w),
    )


def load_fields(cfg, GC, case, var):
    phys = cfg.physics
    truth_key = (str(cfg.paths.wb2_zarr), case["valid_start"], case["valid_end"])
    if truth_key not in _TRUTH_CACHE:
        _TRUTH_CACHE[truth_key] = T.open_truth_wb2(
            cfg.paths.wb2_zarr, phys, case["valid_start"], case["valid_end"]
        )
    truth = _TRUTH_CACHE[truth_key]
    valid_dates = pd.date_range(case["valid_start"], case["valid_end"]).strftime("%Y-%m-%d").tolist()
    obs = T.truth_period_mean(var, truth, valid_dates, GC)
    fields = {}
    for model, init, ds, de in [
        ("FuXi", case["fuxi_init"], case["fuxi_ds"], case["fuxi_de"]),
        ("ECMWF", case["ecmwf_init"], case["ecmwf_ds"], case["ecmwf_de"]),
    ]:
        spec = cfg.model(model)
        cube = get_adapter(spec.adapter)(init, var, spec, phys)
        if cube is None or not cube.has_week(de):
            fields[model] = None
            continue
        fields[model] = cube.weekly(ds, de, GC)[0]
    return obs, fields


def build_cases(cfg, fuxi_inits, max_cases=None):
    cases = []
    for f_init in fuxi_inits:
        for week_label, ds, de in WEEKS:
            match = find_ecmwf_match(cfg, f_init, ds, de)
            if match is None:
                continue
            e_init, eds, ede = match
            valid_start = (pd.to_datetime(f_init) + pd.Timedelta(days=ds)).strftime("%Y-%m-%d")
            valid_end = (pd.to_datetime(f_init) + pd.Timedelta(days=de)).strftime("%Y-%m-%d")
            cases.append(dict(
                year=int(f_init[:4]), fuxi_init=f_init, ecmwf_init=e_init,
                week=week_label, fuxi_ds=ds, fuxi_de=de,
                ecmwf_ds=eds, ecmwf_de=ede,
                valid_start=valid_start, valid_end=valid_end,
            ))
    if max_cases:
        # max_cases counts FuXi initialization dates, not week rows.
        keep = set(fuxi_inits[:max_cases])
        cases = [c for c in cases if c["fuxi_init"] in keep]
    return cases


def compute_metrics(cfg, GC, cases, variables):
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] FuXi {case['fuxi_init']} {case['week']} "
              f"valid {case['valid_start']}..{case['valid_end']} "
              f"ECMWF {case['ecmwf_init']} lead {case['ecmwf_ds']}-{case['ecmwf_de']}",
              flush=True)
        for var in variables:
            obs, fields = load_fields(cfg, GC, case, var)
            if obs is None:
                continue
            for model, field in fields.items():
                if field is None:
                    continue
                for rg in REGIONS:
                    rows.append(score_region(var, model, field, obs, rg, GC, case["week"], case))
    return pd.DataFrame(rows)


def _prep_plot_df(df, metric, region):
    d = df[df.region == region].copy()
    d["case"] = d["fuxi_init"].str.replace("-", "") + " " + d["week"].str.replace("Week ", "W")
    g = d.groupby(["case", "variable", "model"], as_index=False)[metric].mean()
    return g


def fig_heatmap(df, metric, out, region="All India"):
    g = _prep_plot_df(df, metric, region)
    if g.empty:
        return
    cases = list(dict.fromkeys(g["case"]))
    variables = [v for v in VAR_ORDER if v in set(g.variable)]
    models = ["FuXi", "ECMWF"]
    fig, axes = plt.subplots(len(variables), len(models), figsize=(4.8 * len(models), 0.36 * len(cases) + 2.2),
                             squeeze=False)
    cmap = "RdYlBu_r" if metric == "pcc" else "RdBu_r"
    if metric == "pcc":
        vmin, vmax = -0.2, 1.0
    else:
        lim = np.nanpercentile(np.abs(g[metric]), 95) if metric == "bias" else np.nanpercentile(g[metric], 95)
        if metric == "bias":
            vmin, vmax = -max(1.0, lim), max(1.0, lim)
        else:
            vmin, vmax = 0, max(1.0, lim)
    for i, var in enumerate(variables):
        for j, model in enumerate(models):
            ax = axes[i, j]
            mat = np.full((len(cases), 1), np.nan)
            sub = g[(g.variable == var) & (g.model == model)].set_index("case")
            for k, c in enumerate(cases):
                if c in sub.index:
                    mat[k, 0] = sub.loc[c, metric]
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{var} | {model}")
            ax.set_xticks([])
            ax.set_yticks(range(len(cases)), cases if j == 0 else [])
            for k, val in enumerate(mat[:, 0]):
                if np.isfinite(val):
                    ax.text(0, k, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cb.set_label(metric.upper())
    fig.suptitle(f"Matched-valid-window {metric.upper()} ({region})", fontweight="bold")
    fig.savefig(out / f"matched_{metric}_heatmap.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def fig_rmse_lines(df, out, region="All India"):
    fig_metric_lines(df, out, metric="rmse", region=region)


def fig_pcc_lines(df, out, region="All India"):
    fig_metric_lines(df, out, metric="pcc", region=region)


def fig_metric_lines(df, out, metric, region="All India"):
    d = df[df.region == region].copy()
    if d.empty:
        return
    d["lead_week"] = d["week"].str.extract(r"(\d+)")[0].astype(int)
    g = d.groupby(["variable", "model", "lead_week"], as_index=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n=(metric, "count"),
    )
    variables = [v for v in VAR_ORDER if v in set(g.variable)]
    fig, axes = plt.subplots(1, len(variables), figsize=(5.6 * len(variables), 4.2), squeeze=False)
    for ax, var in zip(axes[0], variables):
        for model in ["FuXi", "ECMWF"]:
            sub = g[(g.variable == var) & (g.model == model)].sort_values("lead_week")
            if sub.empty:
                continue
            y = sub["mean"].to_numpy()
            x = sub.lead_week.to_numpy()
            sem = (sub["std"].fillna(0) / np.sqrt(sub["n"].clip(lower=1))).to_numpy()
            ax.plot(x, y, marker="o", lw=2.2, color=MODEL_COLORS[model], label=model)
            ax.fill_between(x, y - sem, y + sem, color=MODEL_COLORS[model], alpha=0.16, linewidth=0)
        ax.set_title(var)
        ax.set_xlabel("Lead week")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(range(1, 7))
        if metric == "pcc":
            ax.set_ylim(-0.05, 1.02)
        ax.legend(frameon=False)
    fig.suptitle(f"Matched-valid-window {metric.upper()} vs lead ({region})", fontweight="bold")
    fig.savefig(out / f"matched_{metric}_lines.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def make_summary_tables(df, out):
    d = df.copy()
    d["lead_week"] = d["week"].str.extract(r"(\d+)")[0].astype(int)
    skill = (
        d.groupby(["variable", "region", "week", "lead_week", "model"], as_index=False)
        .agg(
            n=("pcc", "count"),
            pcc_mean=("pcc", "mean"),
            pcc_std=("pcc", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            bias_mean=("bias", "mean"),
            bias_std=("bias", "std"),
            obs_mean=("obs_mean", "mean"),
            fcst_mean=("fcst_mean", "mean"),
        )
        .sort_values(["variable", "region", "lead_week", "model"])
    )
    skill.to_csv(out / "matched_skill_by_lead.csv", index=False)

    keys = ["fuxi_init", "ecmwf_init", "valid_start", "valid_end",
            "variable", "region", "week"]
    metric_cols = ["pcc", "rmse", "bias", "obs_mean", "fcst_mean"]
    wide = d.pivot_table(index=keys, columns="model", values=metric_cols, aggfunc="mean")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    if {"pcc_FuXi", "pcc_ECMWF"}.issubset(wide.columns):
        wide["pcc_gain_fuxi_minus_ecmwf"] = wide["pcc_FuXi"] - wide["pcc_ECMWF"]
    if {"rmse_FuXi", "rmse_ECMWF"}.issubset(wide.columns):
        wide["rmse_reduction_fuxi_vs_ecmwf"] = wide["rmse_ECMWF"] - wide["rmse_FuXi"]
    if {"bias_FuXi", "bias_ECMWF"}.issubset(wide.columns):
        wide["abs_bias_reduction_fuxi_vs_ecmwf"] = wide["bias_ECMWF"].abs() - wide["bias_FuXi"].abs()
    wide["lead_week"] = wide["week"].str.extract(r"(\d+)")[0].astype(int)
    wide = wide.sort_values(["variable", "region", "lead_week", "fuxi_init"])
    wide.to_csv(out / "matched_pairwise_deltas.csv", index=False)
    return skill, wide


def fig_skill_delta_bars(delta, out, region="All India"):
    d = delta[delta.region == region].copy()
    if d.empty:
        return
    variables = [v for v in VAR_ORDER if v in set(d.variable)]
    fig, axes = plt.subplots(2, len(variables), figsize=(5.6 * len(variables), 7.2), squeeze=False,
                             sharex="col")
    metrics = [
        ("pcc_gain_fuxi_minus_ecmwf", "PCC gain", "FuXi - ECMWF"),
        ("rmse_reduction_fuxi_vs_ecmwf", "RMSE reduction", "ECMWF - FuXi"),
    ]
    for col, var in enumerate(variables):
        sub_var = d[d.variable == var]
        for row, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            if metric not in sub_var:
                ax.axis("off")
                continue
            g = sub_var.groupby("lead_week", as_index=False).agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
                n=(metric, "count"),
            )
            x = g.lead_week.to_numpy()
            y = g["mean"].to_numpy()
            err = (g["std"].fillna(0) / np.sqrt(g["n"].clip(lower=1))).to_numpy()
            colors = np.where(y >= 0, "#D55E00", "#4C78A8")
            ax.bar(x, y, color=colors, width=0.68, edgecolor="white", linewidth=0.8)
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#222222", lw=1.0, capsize=3)
            ax.axhline(0, color="#222222", lw=0.9)
            ax.set_title(f"{var} | {title}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(1, 7))
            if row == 1:
                ax.set_xlabel("Lead week")
            ax.tick_params(labelbottom=True)
    fig.suptitle(f"Matched-valid-window paired FuXi advantage ({region})", fontweight="bold")
    fig.subplots_adjust(hspace=0.38, wspace=0.25)
    fig.savefig(out / "matched_skill_delta_bars.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def fig_region_pcc_gain(delta, out):
    if delta.empty or "pcc_gain_fuxi_minus_ecmwf" not in delta:
        return
    g = (
        delta.groupby(["variable", "region", "lead_week"], as_index=False)
        ["pcc_gain_fuxi_minus_ecmwf"].mean()
    )
    variables = [v for v in VAR_ORDER if v in set(g.variable)]
    regions = [r for r in REGION_ORDER if r in set(g.region)]
    fig, axes = plt.subplots(1, len(variables), figsize=(7.1 * len(variables), 5.6),
                             squeeze=False, sharey=True)
    vmax = np.nanpercentile(np.abs(g["pcc_gain_fuxi_minus_ecmwf"]), 95)
    vmax = max(0.05, float(vmax)) if np.isfinite(vmax) else 0.2
    for col, (ax, var) in enumerate(zip(axes[0], variables)):
        mat = np.full((len(regions), 6), np.nan)
        sub = g[g.variable == var]
        for i, region in enumerate(regions):
            for week in range(1, 7):
                val = sub[(sub.region == region) & (sub.lead_week == week)]
                if not val.empty:
                    mat[i, week - 1] = val["pcc_gain_fuxi_minus_ecmwf"].iloc[0]
        im = ax.imshow(mat, aspect="auto", cmap="BrBG", vmin=-vmax, vmax=vmax)
        ax.set_title(var)
        ax.set_xticks(range(6), [f"W{i}" for i in range(1, 7)])
        ax.set_yticks(range(len(regions)), [REGION_LABELS.get(r, r) for r in regions])
        ax.tick_params(labelleft=(col == 0))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.82, bottom=0.14, wspace=0.10)
    cax = fig.add_axes([0.885, 0.22, 0.018, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PCC gain (FuXi - ECMWF)")
    fig.suptitle("Regional matched-window PCC gain", fontweight="bold")
    fig.savefig(out / "matched_pcc_gain_regions.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def write_summary(df, cases, fuxi_inits, out, skill=None, delta=None):
    lines = [
        "# Matched-Valid-Window JJAS Monsoon Study",
        "",
        f"Compact FuXi init dates used: {len(fuxi_inits)}",
        ", ".join(fuxi_inits) if fuxi_inits else "(none)",
        "",
        f"Matched forecast windows scored: {len(cases)}",
        "",
    ]
    if not df.empty:
        ai = df[df.region == "All India"]
        lines.append("## All-India Mean Skill")
        for var in [v for v in VAR_ORDER if v in set(ai.variable)]:
            lines.append("")
            lines.append(f"### {var}")
            for model in ["FuXi", "ECMWF"]:
                sub = ai[(ai.variable == var) & (ai.model == model)]
                if sub.empty:
                    continue
                lines.append(
                    f"- {model}: PCC {sub.pcc.mean():.2f}, "
                    f"RMSE {sub.rmse.mean():.2f}, bias {sub.bias.mean():+.2f}"
                )
        if delta is not None and not delta.empty:
            lines.append("")
            lines.append("## Paired FuXi Advantage")
            dai = delta[delta.region == "All India"]
            for var in [v for v in VAR_ORDER if v in set(dai.variable)]:
                sub = dai[dai.variable == var]
                if sub.empty:
                    continue
                lines.append(
                    f"- {var}: mean PCC gain {sub.pcc_gain_fuxi_minus_ecmwf.mean():+.2f}; "
                    f"mean RMSE reduction {sub.rmse_reduction_fuxi_vs_ecmwf.mean():+.2f}; "
                    f"FuXi wins {100 * (sub.pcc_gain_fuxi_minus_ecmwf > 0).mean():.0f}% "
                    "of All-India paired windows by PCC."
                )
        if skill is not None and not skill.empty:
            lines.append("")
            lines.append("## Lead-Week Coverage")
            cov = skill[(skill.region == "All India") & (skill.model == "FuXi")]
            for var in [v for v in VAR_ORDER if v in set(cov.variable)]:
                sub = cov[cov.variable == var]
                counts = ", ".join(
                    f"W{int(r.lead_week)} n={int(r.n)}" for r in sub.itertuples()
                )
                lines.append(f"- {var}: {counts}")
    lines.extend([
        "",
        "## Calendar Handling",
        "FuXi and ECMWF initialization dates are offset. Metrics and figures match",
        "forecasts by ERA5 valid date window, not by nominal lead week alone.",
    ])
    (out / "STUDY_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--vars", nargs="+", default=["TP", "Z500"], choices=["TP", "Z500"])
    p.add_argument("--max-inits", type=int, default=None,
                   help="limit number of compact FuXi init dates for a quick run")
    p.add_argument("--out", default=str(HERE / "figs" / "matched_monsoon_study"))
    args = p.parse_args()

    set_plot_style()
    cfg = build_config(args.year)
    GC = G.build_grid_context(cfg.grid, cfg.paths.region_mask_nc)
    fuxi_inits = compact_fuxi_inits(args.year)
    if args.max_inits:
        fuxi_inits = fuxi_inits[:args.max_inits]
    if not fuxi_inits:
        raise SystemExit("No compact FuXi files found. Run jjas/preprocess_fuxi.py first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(cfg, fuxi_inits)
    pd.DataFrame(cases).to_csv(out / "matched_cases.csv", index=False)
    df = compute_metrics(cfg, GC, cases, args.vars)
    df.to_csv(out / "matched_case_metrics.csv", index=False)

    if not df.empty:
        skill, delta = make_summary_tables(df, out)
        fig_heatmap(df, "pcc", out)
        fig_heatmap(df, "bias", out)
        fig_pcc_lines(df, out)
        fig_rmse_lines(df, out)
        fig_skill_delta_bars(delta, out)
        fig_region_pcc_gain(delta, out)
    else:
        skill, delta = None, None
    write_summary(df, cases, fuxi_inits, out, skill=skill, delta=delta)
    print(f"WROTE study -> {out}")
    print(f"rows={len(df)} cases={len(cases)} fuxi_inits={len(fuxi_inits)}")


if __name__ == "__main__":
    main()
