#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plots/meeting_figs.py  —  Slide-ready "highlights" figures for a meeting.
================================================================================
A small, punchy set (big fonts, takeaway titles, clean layout) drawn from any
results dir's skill CSVs. Not the full publication suite (see make_plots.py) —
these are the 4 figures you put on slides.

  M1  skill_horizon   PCC vs lead day (TP | Z500), all models, week bands
  M2  crpss           probabilistic skill CRPSS vs week (TP | Z500)
  M3  scorecard       model × week PCC heatmap (TP | Z500) — at-a-glance
  M4  heavy_rain_bss  Brier skill for heavy rain (>10 mm/day) by week

Run
  python meeting_figs.py --results ../results_1.5deg
  python meeting_figs.py --results /storage/raj.ayush/s2s_final_data/jjas/results_2019_1.5deg
Outputs -> /storage/raj.ayush/s2s_final_data/figs_meeting/<results-name>/  (data on /storage)
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
sys.path.append(os.path.dirname(os.path.dirname(HERE)))
from core.plotting import style_for, MODEL_STYLE, VAR_UNITS

DET_ORDER = ["SPIRE", "FuXi", "ECMWF", "MME", "Persistence"]
PROB_ORDER = ["SPIRE", "FuXi", "ECMWF"]
VAR_TITLE = {"TP": "Rainfall", "Z500": "500 hPa height", "T2M": "2 m temperature"}


def theme():
    matplotlib.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "font.size": 15,
        "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True, "lines.linewidth": 3,
        "lines.markersize": 7,
    })


def agg(df, val, scale, region, var, models):
    s = df[(df.scale == scale) & (df.region == region) & (df.variable == var)]
    g = s.groupby(["model", "lead"])[val].mean().reset_index()
    return {m: (g[g.model == m].sort_values("lead").lead.values,
                g[g.model == m].sort_values("lead")[val].values)
            for m in models if (g.model == m).any()}


def _save(fig, out, name, msg=None):
    if msg:
        fig.text(0.5, -0.02, msg, ha="center", fontsize=13, style="italic", color="0.3")
    p = os.path.join(out, name + ".png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def m1_skill_horizon(det, region, out):
    vars_ = [v for v in ("TP", "Z500", "T2M") if v in det.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 5.6), squeeze=False)
    for ax, var in zip(axes[0], vars_):
        for i, (a, b) in enumerate([(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]):
            if i % 2 == 0:
                ax.axvspan(a - .5, b + .5, color="0.94", zorder=0)
        for m, (ld, v) in agg(det, "pcc", "daily", region, var, DET_ORDER).items():
            st = style_for(m)
            ax.plot(ld, v, color=st["color"], label=st["label"])
        ax.axhline(0.5, color="0.4", ls="--", lw=1.5)
        ax.set_xlim(1, 42); ax.set_ylim(-0.2, 1.02)
        ax.set_xticks([4, 11, 18, 25, 32, 39]); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        ax.set_xlabel("Forecast lead"); ax.set_ylabel("Pattern correlation")
        ax.set_title(VAR_TITLE.get(var, var))
    axes[0][0].legend(loc="upper right", ncol=2)
    fig.suptitle("How far ahead is the forecast skilful?  (pattern correlation, All-India)",
                 fontsize=20, fontweight="bold")
    _save(fig, out, "M1_skill_horizon", "dashed line = PCC 0.5 (a common skill threshold)")


def m2_crpss(prob, region, out):
    vars_ = [v for v in ("TP", "Z500", "T2M") if v in prob.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 5.6), squeeze=False)
    for ax, var in zip(axes[0], vars_):
        for m, (ld, v) in agg(prob, "crpss_clim", "weekly", region, var, PROB_ORDER).items():
            st = style_for(m)
            ax.plot(ld, v, color=st["color"], marker=st["marker"], label=st["label"])
        ax.axhline(0.0, color="0.4", ls="--", lw=1.5)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        ax.set_xlabel("Forecast lead"); ax.set_ylabel("CRPSS vs climatology")
        ax.set_title(VAR_TITLE.get(var, var))
    axes[0][0].legend(loc="upper right")
    fig.suptitle("Probabilistic skill above climatology  (CRPSS, All-India)",
                 fontsize=20, fontweight="bold")
    _save(fig, out, "M2_crpss", "above 0 = better than just guessing climatology")


def m3_scorecard(det, region, out):
    vars_ = [v for v in ("TP", "Z500", "T2M") if v in det.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(6.2 * len(vars_), 5.4), squeeze=False)
    models = [m for m in DET_ORDER]
    for ax, var in zip(axes[0], vars_):
        d = det[(det.scale == "weekly") & (det.region == region) & (det.variable == var)]
        mat = np.full((len(models), 6), np.nan)
        for i, m in enumerate(models):
            for wk in range(1, 7):
                s = d[(d.model == m) & (d.lead == wk)]["pcc"]
                if len(s):
                    mat[i, wk - 1] = s.mean()
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(6):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=12, fontweight="bold",
                            color="white" if mat[i, j] < 0.3 else "black")
        ax.set_title(VAR_TITLE.get(var, var))
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Pattern correlation")
    fig.suptitle("Skill scorecard — pattern correlation by model and week",
                 fontsize=20, fontweight="bold")
    _save(fig, out, "M3_scorecard", "green = skilful, red = no skill")


def m4_heavy_rain(brier, region, out):
    ev = "tp_gt_10mm"
    sub = brier[(brier.region == region) & (brier.event == ev)]
    if not len(sub):
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for m in PROB_ORDER:
        s = sub[sub.model == m].groupby("lead")["briss_clim"].mean().reset_index().sort_values("lead")
        if len(s):
            st = style_for(m)
            ax.plot(s.lead, s.briss_clim, color=st["color"], marker=st["marker"], label=m)
    ax.axhline(0.0, color="0.4", ls="--", lw=1.5)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
    ax.set_xlabel("Forecast lead"); ax.set_ylabel("Brier skill score")
    ax.set_title("Skill for heavy rain (>10 mm/day)")
    ax.legend()
    fig.suptitle("Can the models flag heavy-rain weeks?  (All-India)",
                 fontsize=20, fontweight="bold")
    _save(fig, out, "M4_heavy_rain_bss", "above 0 = better than climatology at predicting heavy-rain events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--region", default="All India")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    theme()
    name = os.path.basename(args.results.rstrip("/"))
    # figures are for VIEWING -> keep them in home (gitignored), findable in the IDE.
    out = args.out or os.path.join(HERE, "figs", "meeting", name)
    os.makedirs(out, exist_ok=True)
    print(f"Reading {args.results} -> {out}")
    det = pd.read_csv(os.path.join(args.results, "skill_deterministic.csv"))
    det = det[det.clim_basis == "era5"]
    prob = pd.read_csv(os.path.join(args.results, "skill_probabilistic.csv"))
    brier_p = os.path.join(args.results, "skill_brier.csv")
    brier = pd.read_csv(brier_p) if os.path.exists(brier_p) else pd.DataFrame()
    m1_skill_horizon(det, args.region, out)
    m2_crpss(prob, args.region, out)
    m3_scorecard(det, args.region, out)
    if len(brier):
        m4_heavy_rain(brier, args.region, out)
    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    main()
