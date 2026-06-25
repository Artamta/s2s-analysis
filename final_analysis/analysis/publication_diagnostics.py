#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
publication_diagnostics.py
==========================
Publication-facing diagnostics for any final_analysis results directory.

Reads the standard verification outputs:
  skill_deterministic.csv
  skill_probabilistic.csv
  skill_brier.csv

and writes:
  summary_bootstrap.csv       bootstrap means and 95% CIs by init date
  headline_numbers.md         compact text summary for papers/talks
  pub_scorecard_pcc.png       weekly PCC scorecard
  pub_dual_basis_gap.png      ERA5-basis minus model-own-basis PCC gap
  pub_skill_horizon.png       last useful week at PCC >= 0.5
  pub_crpss_ssr.png           probabilistic skill/calibration panel

The script is deliberately season-agnostic. It works for JFM2026 and JJAS result
folders as long as they follow the final_analysis CSV schema.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["FuXi", "SPIRE", "ECMWF", "MME", "Persistence"]
VAR_ORDER = ["TP", "Z500", "T2M"]
WEEKS = [1, 2, 3, 4, 5, 6]
MODEL_STYLE = {
    "SPIRE": dict(color="#0072B2", marker="o", lw=2.2, label="SPIRE"),
    "FuXi": dict(color="#D55E00", marker="s", lw=2.2, label="FuXi"),
    "ECMWF": dict(color="#009E73", marker="^", lw=2.2, label="ECMWF"),
    "MME": dict(color="#000000", marker="D", lw=2.0, label="MME"),
    "Persistence": dict(color="#999999", marker="x", lw=1.6, label="Persistence"),
}


def apply_theme():
    matplotlib.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def style_for(model):
    return MODEL_STYLE.get(model, dict(color="#444444", marker=".", lw=1.8, label=model))


def _model_rank(model):
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def _var_rank(var):
    return VAR_ORDER.index(var) if var in VAR_ORDER else len(VAR_ORDER)


def _read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_results(results):
    root = Path(results)
    det = _read_csv(root / "skill_deterministic.csv")
    prob = _read_csv(root / "skill_probabilistic.csv")
    brier = _read_csv(root / "skill_brier.csv")
    return det, prob, brier


def bootstrap_mean(values, nboot=2000, seed=1234):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1 or nboot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    boots = np.nanmean(x[idx], axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def summarize_metric(df, metric, *, source, region, nboot):
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    d = df[(df["scale"] == "weekly") & (df["region"] == region)].copy()
    if d.empty:
        return pd.DataFrame()
    if "clim_basis" not in d.columns:
        d["clim_basis"] = "era5"

    rows = []
    keys = ["variable", "model", "clim_basis", "lead"]
    for key, sub in d.groupby(keys, dropna=False):
        var, model, basis, lead = key
        vals = sub.groupby("init_date", dropna=False)[metric].mean().values
        mean, lo, hi = bootstrap_mean(vals, nboot=nboot)
        rows.append(dict(
            source=source, metric=metric, variable=var, model=model,
            clim_basis=basis, lead=int(lead), region=region,
            n_init=int(np.isfinite(vals).sum()), mean=mean,
            ci_low=lo, ci_high=hi,
        ))
    return pd.DataFrame(rows)


def build_summary(det, prob, region, nboot):
    parts = [
        summarize_metric(det, "pcc", source="deterministic", region=region, nboot=nboot),
        summarize_metric(det, "rmse", source="deterministic", region=region, nboot=nboot),
        summarize_metric(det, "bias", source="deterministic", region=region, nboot=nboot),
        summarize_metric(det, "std_ratio", source="deterministic", region=region, nboot=nboot),
        summarize_metric(prob, "crpss_clim", source="probabilistic", region=region, nboot=nboot),
        summarize_metric(prob, "ssr", source="probabilistic", region=region, nboot=nboot),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(
        ["source", "metric", "variable", "model", "clim_basis", "lead"],
        key=lambda s: s.map(_model_rank) if s.name == "model"
        else s.map(_var_rank) if s.name == "variable"
        else s,
    ).reset_index(drop=True)
    return out


def _weekly_matrix(summary, metric="pcc", basis="era5"):
    s = summary[(summary["metric"] == metric) & (summary["clim_basis"] == basis)].copy()
    if s.empty:
        return [], np.empty((0, len(WEEKS)))
    rows = []
    info = []
    for (var, model), sub in s.groupby(["variable", "model"], sort=False):
        vals = sub.set_index("lead")["mean"]
        n_init = int(sub["n_init"].max())
        rows.append([vals.get(w, np.nan) for w in WEEKS])
        info.append((var, model, f"{var} | {model} (n={n_init})"))
    order = sorted(range(len(info)), key=lambda i: (_var_rank(info[i][0]), _model_rank(info[i][1])))
    return [info[i][2] for i in order], np.asarray([rows[i] for i in order], float)


def fig_scorecard(summary, out, basis="era5"):
    labels, mat = _weekly_matrix(summary, metric="pcc", basis=basis)
    if mat.size == 0:
        return
    h = max(4.5, 0.42 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8.6, h))
    im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(WEEKS)), [f"W{w}" for w in WEEKS])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"Weekly pattern correlation scorecard ({basis.upper()} basis)")
    ax.set_xlabel("Forecast lead")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if val < 0.15 or val > 0.72 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("PCC")
    fig.savefig(out / "pub_scorecard_pcc.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def fig_dual_basis_gap(summary, out):
    s = summary[(summary["metric"] == "pcc")].copy()
    if s.empty or "model_own" not in set(s["clim_basis"]):
        return
    era = s[s["clim_basis"] == "era5"]
    own = s[s["clim_basis"] == "model_own"]
    merged = era.merge(
        own,
        on=["variable", "model", "lead", "region", "metric", "source"],
        suffixes=("_era5", "_own"),
    )
    if merged.empty:
        return
    merged["gap"] = merged["mean_era5"] - merged["mean_own"]

    fig, axes = plt.subplots(1, len(sorted(merged.variable.unique(), key=_var_rank)),
                             figsize=(6.2 * merged.variable.nunique(), 4.8),
                             squeeze=False)
    for ax, var in zip(axes[0], sorted(merged.variable.unique(), key=_var_rank)):
        d = merged[merged.variable == var]
        for model in sorted(d.model.unique(), key=_model_rank):
            dm = d[d.model == model].sort_values("lead")
            st = style_for(model)
            ax.plot(dm["lead"], dm["gap"], marker=st["marker"], color=st["color"],
                    lw=2.4, label=model)
        ax.axhline(0, color="0.35", lw=1.0)
        ax.set_xticks(WEEKS, [f"W{w}" for w in WEEKS])
        ax.set_title(var)
        ax.set_xlabel("Forecast lead")
        ax.set_ylabel("PCC(ERA5 basis) - PCC(model-own basis)")
        ax.legend()
    fig.suptitle("Climatology-basis sensitivity: apparent vs model-own anomaly skill",
                 fontsize=14, fontweight="bold")
    fig.savefig(out / "pub_dual_basis_gap.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def fig_skill_horizon(summary, out, threshold=0.5):
    s = summary[(summary["metric"] == "pcc") & (summary["clim_basis"] == "era5")]
    if s.empty:
        return
    rows = []
    for (var, model), sub in s.groupby(["variable", "model"]):
        vals = sub.set_index("lead")["mean"]
        horizon = 0
        for week in WEEKS:
            val = vals.get(week, np.nan)
            if np.isfinite(val) and val >= threshold:
                horizon = week
            else:
                break
        rows.append(dict(variable=var, model=model, horizon_days=7 * horizon))
    h = pd.DataFrame(rows)
    if h.empty:
        return

    vars_ = sorted(h.variable.unique(), key=_var_rank)
    fig, axes = plt.subplots(1, len(vars_), figsize=(5.4 * len(vars_), 4.6), squeeze=False)
    for ax, var in zip(axes[0], vars_):
        d = h[h.variable == var].sort_values("model", key=lambda c: c.map(_model_rank))
        colors = [style_for(m)["color"] for m in d.model]
        ax.bar(d.model, d.horizon_days, color=colors, edgecolor="white")
        ax.set_ylim(0, 42)
        ax.set_yticks(range(0, 43, 7))
        ax.set_title(var)
        ax.set_ylabel("Useful horizon (days)")
        ax.axhline(14, color="0.65", lw=1.0, ls=":")
        ax.axhline(28, color="0.65", lw=1.0, ls=":")
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")
    fig.suptitle(f"Last continuous lead with PCC >= {threshold:.1f}",
                 fontsize=14, fontweight="bold")
    fig.savefig(out / "pub_skill_horizon.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def fig_prob(summary, out):
    s = summary[summary["metric"].isin(["crpss_clim", "ssr"])].copy()
    if s.empty:
        return
    vars_ = sorted(s.variable.unique(), key=_var_rank)
    fig, axes = plt.subplots(2, len(vars_), figsize=(5.8 * len(vars_), 8.2),
                             squeeze=False, sharex=True)
    for j, var in enumerate(vars_):
        for model in sorted(s.model.unique(), key=_model_rank):
            st = style_for(model)
            for i, metric in enumerate(["crpss_clim", "ssr"]):
                ax = axes[i][j]
                d = s[(s.variable == var) & (s.model == model) &
                      (s.metric == metric)].sort_values("lead")
                if d.empty:
                    continue
                ax.plot(d.lead, d["mean"], marker=st["marker"], color=st["color"],
                        lw=2.3, label=model)
                ax.fill_between(d.lead, d.ci_low, d.ci_high, color=st["color"],
                                alpha=0.14, linewidth=0)
        axes[0][j].axhline(0, color="0.35", lw=1.0)
        axes[1][j].axhline(1, color="0.35", lw=1.0)
        axes[0][j].set_title(var)
        axes[0][j].set_ylabel("CRPSS vs climatology")
        axes[1][j].set_ylabel("Spread-skill ratio")
        axes[1][j].set_xlabel("Forecast lead")
        for ax in axes[:, j]:
            ax.set_xticks(WEEKS, [f"W{w}" for w in WEEKS])
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Probabilistic skill and calibration", fontsize=14, fontweight="bold")
    fig.savefig(out / "pub_crpss_ssr.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def write_headlines(summary, out):
    lines = ["# Headline Numbers", ""]
    pcc = summary[(summary.metric == "pcc") & (summary.clim_basis == "era5")]
    if not pcc.empty:
        lines.append("## Weekly All-India PCC")
        for var in sorted(pcc.variable.unique(), key=_var_rank):
            lines.append("")
            lines.append(f"### {var}")
            d = pcc[pcc.variable == var]
            for model in sorted(d.model.unique(), key=_model_rank):
                dm = d[d.model == model].set_index("lead")
                vals = []
                for w in WEEKS:
                    if w in dm.index:
                        row = dm.loc[w]
                        vals.append(f"W{w} {row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]")
                if vals:
                    n_init = int(dm["n_init"].max())
                    lines.append(f"- {model} (n={n_init} inits): " + "; ".join(vals))

    gap = summary[summary.metric == "pcc"]
    if "model_own" in set(gap.clim_basis):
        lines.extend(["", "## Dual-Basis Interpretation", ""])
        lines.append("Positive gap means the shared ERA5-climatology score is higher "
                     "than the model-own-climatology score.")
    (out / "headline_numbers.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Directory containing final_analysis skill CSVs")
    ap.add_argument("--out", default=None,
                    help="Output directory; default: <results>/publication_diagnostics")
    ap.add_argument("--region", default="All India")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()

    apply_theme()
    det, prob, brier = load_results(args.results)
    if det.empty:
        raise SystemExit(f"No deterministic rows found in {args.results}")

    out = Path(args.out) if args.out else Path(args.results) / "publication_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary(det, prob, args.region, args.nboot)
    if summary.empty:
        raise SystemExit("No weekly rows found to summarize")
    summary.to_csv(out / "summary_bootstrap.csv", index=False)

    fig_scorecard(summary, out)
    fig_dual_basis_gap(summary, out)
    fig_skill_horizon(summary, out)
    fig_prob(summary, out)
    write_headlines(summary, out)

    print(f"WROTE publication diagnostics -> {out}")
    print(f"  rows: {len(summary)}")
    print("  figures: pub_scorecard_pcc.png, pub_dual_basis_gap.png, "
          "pub_skill_horizon.png, pub_crpss_ssr.png")


if __name__ == "__main__":
    main()
