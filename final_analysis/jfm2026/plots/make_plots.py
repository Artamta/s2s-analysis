#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plots/make_plots.py  —  Publication figure suite for the JFM2026 verification.
================================================================================
Reads the three skill CSVs (+ reliability.npz) from a results directory and
writes a consistent set of figures. Everything is averaged over the 13 init
dates (mean skill), clim_basis='era5', and split by region/variable/lead.

Figures (per variable unless noted)
-----------------------------------
  01 pcc_daily            PCC vs daily lead (1..42), All-India, all models, week bands
  02 deterministic_weekly 2x2: PCC, RMSE, MSSS-vs-clim, bias  by week
  03 prob_weekly          1x2: CRPSS-vs-clim, Spread-Skill-Ratio (ideal=1) by week
  04 bss_events           Brier Skill Score by week, one line per event
  05 regional_pcc         PCC by week, small-multiples over the 4 IMD regions + All India
  06 reliability          reliability diagrams for the tercile/threshold events
  07 skill_horizon        bar summary of PCC>0.5 and CRPSS>0 horizons (days)
  R  resolution_compare   (only with --compare) common vs native PCC & CRPSS overlay

Run
---
  python make_plots.py                                   # uses ../results_1.5deg
  python make_plots.py --results ../results_0.5deg
  python make_plots.py --results ../results_1.5deg --compare ../results_0.5deg
  python make_plots.py --region central_india
================================================================================
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(HERE)))     # final_analysis/ on path
from core.plotting import (apply_theme, style_for, shade_weeks, ref_line,
                           skill_horizon, VAR_UNITS, MODEL_STYLE)

REGION_TITLE = {
    "All India": "All India", "northwest_india": "Northwest India",
    "central_india": "Central India", "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & NE India",
}
DET_ORDER  = ["SPIRE", "FuXi", "ECMWF", "MME", "Persistence"]
PROB_ORDER = ["SPIRE", "FuXi", "ECMWF"]


# ------------------------------------------------------------------ data load --
def load(results):
    det = pd.read_csv(os.path.join(results, "skill_deterministic.csv"))
    prob = pd.read_csv(os.path.join(results, "skill_probabilistic.csv"))
    brier = pd.read_csv(os.path.join(results, "skill_brier.csv"))
    det = det[det.clim_basis == "era5"]
    rel_path = os.path.join(results, "reliability.npz")
    rel = np.load(rel_path) if os.path.exists(rel_path) else None
    return det, prob, brier, rel


def agg(df, value, *, scale, region, variable, models):
    """Mean over init dates -> {model: (leads, values)} sorted by lead."""
    sub = df[(df.scale == scale) & (df.region == region) & (df.variable == variable)]
    g = sub.groupby(["model", "lead"])[value].mean().reset_index()
    out = {}
    for m in models:
        s = g[g.model == m].sort_values("lead")
        if len(s):
            out[m] = (s.lead.values, s[value].values)
    return out


def _line_by_week(ax, series, ylabel, title, *, hline=None, hl_txt=None):
    for m, (leads, vals) in series.items():
        st = style_for(m)
        ax.plot(leads, vals, color=st["color"], marker=st["marker"],
                lw=st["lw"], ms=6, label=st["label"])
    if hline is not None:
        ref_line(ax, hline, hl_txt)
    ax.set_xlabel("Lead (week)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])


# ===================================================================== figures ==
def fig_pcc_daily(det, region, var, out):
    series = agg(det, "pcc", scale="daily", region=region, variable=var, models=DET_ORDER)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    shade_weeks(ax)
    for m, (leads, vals) in series.items():
        st = style_for(m)
        ax.plot(leads, vals, color=st["color"], lw=st["lw"], label=st["label"])
    ref_line(ax, 0.5, "PCC=0.5")
    ax.set_ylim(-0.2, 1.02); ax.set_xlim(1, 42)
    ax.set_xlabel("Lead day"); ax.set_ylabel("Pattern correlation (PCC)")
    ax.set_title(f"{var} daily PCC — {REGION_TITLE[region]}")
    ax.legend(ncol=3, loc="upper right", fontsize=9)
    _save(fig, out, f"01_pcc_daily_{var}")


def fig_deterministic_weekly(det, region, var, out):
    panels = [("pcc", "PCC", "Pattern correlation", 0.5, "0.5"),
              ("rmse", "RMSE", f"RMSE ({VAR_UNITS[var]})", None, None),
              ("msss_clim", "MSSS", "MSSS vs climatology", 0.0, "0"),
              ("bias", "Bias", f"Bias ({VAR_UNITS[var]})", 0.0, None)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, short, ylab, hl, hltxt) in zip(axes.ravel(), panels):
        series = agg(det, col, scale="weekly", region=region, variable=var, models=DET_ORDER)
        _line_by_week(ax, series, ylab, f"{var} {short}", hline=hl, hl_txt=hltxt)
    axes[0, 0].legend(ncol=3, loc="upper right", fontsize=8)
    fig.suptitle(f"{var} deterministic skill by week — {REGION_TITLE[region]}",
                 fontsize=13, fontweight="bold")
    _save(fig, out, f"02_deterministic_weekly_{var}")


def fig_prob_weekly(prob, region, var, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    s1 = agg(prob, "crpss_clim", scale="weekly", region=region, variable=var, models=PROB_ORDER)
    _line_by_week(axes[0], s1, "CRPSS vs climatology", f"{var} CRPSS", hline=0.0, hl_txt="0")
    s2 = agg(prob, "ssr", scale="weekly", region=region, variable=var, models=PROB_ORDER)
    _line_by_week(axes[1], s2, "Spread-skill ratio", f"{var} SSR (ideal=1)", hline=1.0, hl_txt="1 (calibrated)")
    axes[0].legend(ncol=3, loc="upper right", fontsize=9)
    fig.suptitle(f"{var} probabilistic skill — {REGION_TITLE[region]}",
                 fontsize=13, fontweight="bold")
    _save(fig, out, f"03_prob_weekly_{var}")


def fig_bss_events(brier, region, var, out):
    evs = sorted(brier[(brier.variable == var)].event.unique())
    if not evs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(evs) * len(PROB_ORDER), 1)))
    sub = brier[(brier.region == region) & (brier.variable == var)]
    handles = []
    for j, m in enumerate(PROB_ORDER):
        for k, ev in enumerate(evs):
            s = (sub[(sub.model == m) & (sub.event == ev)]
                 .groupby("lead")["briss_clim"].mean().reset_index().sort_values("lead"))
            if not len(s):
                continue
            st = style_for(m)
            ls = ["-", "--", ":", "-."][k % 4]
            ax.plot(s.lead, s.briss_clim, color=st["color"], ls=ls, lw=2, marker=st["marker"], ms=4,
                    label=f"{m} · {ev}")
    ref_line(ax, 0.0, "no skill")
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
    ax.set_xlabel("Lead (week)"); ax.set_ylabel("Brier skill score")
    ax.set_title(f"{var} BSS by event — {REGION_TITLE[region]}")
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    _save(fig, out, f"04_bss_events_{var}")


def fig_regional_pcc(det, var, out):
    regions = list(REGION_TITLE)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    for ax, rg in zip(axes.ravel(), regions):
        series = agg(det, "pcc", scale="weekly", region=rg, variable=var, models=DET_ORDER)
        _line_by_week(ax, series, "PCC", REGION_TITLE[rg], hline=0.5, hl_txt="")
        ax.set_ylim(-0.3, 1.02)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(ncol=2, fontsize=8, loc="lower left")
    fig.suptitle(f"{var} PCC by IMD region — weekly", fontsize=13, fontweight="bold")
    _save(fig, out, f"05_regional_pcc_{var}")


def fig_reliability(rel, out):
    if rel is None:
        return
    events = sorted({k.split("__")[0] for k in rel.files if k.endswith("obs_freq")})
    if not events:
        return
    n = len(events); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for i, ev in enumerate(events):
        ax = axes[i // cols][i % cols]
        ax.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--")
        for m in PROB_ORDER:
            of, fp = f"{ev}__{m}__obs_freq", f"{ev}__{m}__fcst_p"
            if of in rel.files:
                st = style_for(m)
                ax.plot(rel[fp], rel[of], color=st["color"], marker=st["marker"], ms=5, lw=1.8, label=m)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(ev, fontsize=10); ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Reliability diagrams (All India)", fontsize=13, fontweight="bold")
    _save(fig, out, "06_reliability")


def fig_skill_horizon(det, prob, region, out):
    """Bar chart: lead-day where PCC>0.5 (det) and CRPSS>0 (prob), per var/model."""
    rows = []
    for var in sorted(det.variable.unique()):
        for m in DET_ORDER:
            s = agg(det, "pcc", scale="daily", region=region, variable=var, models=[m]).get(m)
            if s:
                rows.append(dict(var=var, model=m, kind="PCC>0.5",
                                 horizon=skill_horizon(s[0], s[1], 0.5, above=True)))
    for var in sorted(prob.variable.unique()):
        for m in PROB_ORDER:
            s = agg(prob, "crpss_clim", scale="daily", region=region, variable=var, models=[m]).get(m)
            if s:
                rows.append(dict(var=var, model=m, kind="CRPSS>0",
                                 horizon=skill_horizon(s[0], s[1], 0.0, above=True)))
    if not rows:
        return None
    hz = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, kind in zip(axes, ["PCC>0.5", "CRPSS>0"]):
        sub = hz[hz.kind == kind]
        vars_ = sorted(sub["var"].unique()); models = [m for m in DET_ORDER if m in set(sub.model)]
        x = np.arange(len(vars_)); width = 0.8 / max(len(models), 1)
        for j, m in enumerate(models):
            vals = [sub[(sub["var"] == v) & (sub.model == m)]["horizon"].mean() for v in vars_]
            ax.bar(x + j * width, vals, width, color=style_for(m)["color"], label=m)
        ax.set_xticks(x + width * (len(models) - 1) / 2); ax.set_xticklabels(vars_)
        ax.set_ylabel("Skill horizon (lead days)"); ax.set_title(kind)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Skill horizon — {REGION_TITLE[region]}", fontsize=13, fontweight="bold")
    _save(fig, out, "07_skill_horizon")
    return hz


def fig_resolution_compare(results_a, results_b, region, out):
    """Overlay PCC (daily) and CRPSS (weekly) at common vs native resolution."""
    da, pa, _, _ = load(results_a); db, pb, _, _ = load(results_b)
    ra = _res_label(results_a); rb = _res_label(results_b)
    for var in sorted(set(da.variable) & set(db.variable)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, (frame_a, frame_b, col, scale, ylab, models, hl) in zip(axes, [
            (da, db, "pcc", "daily", "PCC", DET_ORDER, 0.5),
            (pa, pb, "crpss_clim", "weekly", "CRPSS vs clim", PROB_ORDER, 0.0)]):
            sa = agg(frame_a, col, scale=scale, region=region, variable=var, models=models)
            sb = agg(frame_b, col, scale=scale, region=region, variable=var, models=models)
            for m in sa:
                st = style_for(m)
                ax.plot(*sa[m], color=st["color"], lw=2.0, ls="-", label=f"{m} {ra}")
                if m in sb:
                    ax.plot(*sb[m], color=st["color"], lw=1.6, ls="--", label=f"{m} {rb}")
            ref_line(ax, hl)
            ax.set_xlabel("Lead day" if scale == "daily" else "Lead (week)")
            ax.set_ylabel(ylab); ax.set_title(f"{var} {ylab}")
            if scale == "weekly":
                ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        axes[0].legend(ncol=2, fontsize=7)
        fig.suptitle(f"{var}: common ({ra}) vs native ({rb}) — {REGION_TITLE[region]}  "
                     f"[solid={ra}, dashed={rb}]", fontsize=12, fontweight="bold")
        _save(fig, out, f"R_resolution_{var}")


# --------------------------------------------------------------------- helpers --
def _res_label(path):
    base = os.path.basename(path.rstrip("/"))
    return base.replace("results_", "").replace("deg", "°") if "results_" in base else base


def _save(fig, out, name):
    p = os.path.join(out, name + ".png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {os.path.relpath(p)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results_1.5deg"))
    ap.add_argument("--compare", default=None, help="second results dir for resolution overlay")
    ap.add_argument("--region", default="All India")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_theme()
    out = args.out or os.path.join(HERE, "figs", _res_label(args.results).replace("°", "deg"))
    os.makedirs(out, exist_ok=True)
    print(f"Reading {args.results}  ->  writing figures to {out}")

    det, prob, brier, rel = load(args.results)
    for var in sorted(det.variable.unique()):
        fig_pcc_daily(det, args.region, var, out)
        fig_deterministic_weekly(det, args.region, var, out)
        fig_regional_pcc(det, var, out)
    for var in sorted(prob.variable.unique()):
        fig_prob_weekly(prob, args.region, var, out)
    for var in sorted(brier.variable.unique()):
        fig_bss_events(brier, args.region, var, out)
    fig_reliability(rel, out)
    hz = fig_skill_horizon(det, prob, args.region, out)
    if hz is not None:
        hz.to_csv(os.path.join(out, "skill_horizon.csv"), index=False)
        print("\nSkill-horizon summary (lead days):")
        print(hz.pivot_table(index=["var", "model"], columns="kind", values="horizon").to_string())
    if args.compare:
        fig_resolution_compare(args.results, args.compare, args.region, out)
    print("\nDONE.")


if __name__ == "__main__":
    main()
