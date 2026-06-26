#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a3_region_profiles.py  —  Skill across the 4 IMD homogeneous regions.
================================================================================
Reads the VERIFIED skill CSVs (results_1.5deg/) and renders, per variable, a
2x2 small-multiple — one IMD homogeneous region per panel — of pattern
correlation vs lead day for every model. This is the direct "how does skill
differ region to region?" view the All-India curves cannot give.

A companion region-x-week scorecard heatmap (one model picked, default the MME)
shows the same information as a compact at-a-glance grid.

  python a3_region_profiles.py --results ../results_1.5deg
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

DET_MODELS = ["SPIRE", "FuXi", "ECMWF", "MME", "Persistence"]
WEEK_BANDS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]


def _series(det, var, region, model, metric="pcc"):
    s = det[(det.scale == "daily") & (det.variable == var) &
            (det.region == region) & (det.model == model)]
    g = s.groupby("lead")[metric].mean().reset_index().sort_values("lead")
    return g.lead.values, g[metric].values


def region_profiles(det, var):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, region in zip(axes.ravel(), C.REGION_ORDER):
        for i, (a, b) in enumerate(WEEK_BANDS):
            if i % 2 == 0:
                ax.axvspan(a - .5, b + .5, color="0.95", zorder=0)
        for m in DET_MODELS:
            ld, v = _series(det, var, region, m)
            if not len(ld):
                continue
            col = C.MODEL_COLOR.get(m, "0.5")
            ls = "--" if m == "Persistence" else "-"
            lw = 1.6 if m in ("Persistence", "MME") else 2.4
            ax.plot(ld, v, color=col, ls=ls, lw=lw, label=m)
        ax.axhline(0.5, color="0.4", ls=":", lw=1.2)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xlim(1, 42); ax.set_ylim(-0.3, 1.02)
        ax.set_xticks([4, 11, 18, 25, 32, 39])
        ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        ax.set_title(C.REGION_NICE[region])
        ax.grid(alpha=0.25, ls=":")
    axes[0, 0].set_ylabel("Pattern correlation")
    axes[1, 0].set_ylabel("Pattern correlation")
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=10)
    fig.suptitle(f"{C.VAR_LONG[var]} — skill by IMD homogeneous region  "
                 f"(pattern correlation vs lead, JFM2026)",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, -0.01, "dotted line = PCC 0.5 skill threshold", ha="center",
             fontsize=11, style="italic", color="0.4")
    C.savefig(fig, f"A3_region_profiles_{var}.png")


def region_week_scorecard(det, var, model):
    d = det[(det.scale == "weekly") & (det.variable == var) & (det.model == model)]
    if not len(d):
        return
    regions = ["All India"] + C.REGION_ORDER
    mat = np.full((len(regions), 6), np.nan)
    for i, rg in enumerate(regions):
        for wk in range(1, 7):
            s = d[(d.region == rg) & (d.lead == wk)]["pcc"]
            if len(s):
                mat[i, wk - 1] = s.mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([f"Week {i}" for i in range(1, 7)])
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(["All-India"] + [C.REGION_NICE[r] for r in C.REGION_ORDER])
    for i in range(len(regions)):
        for j in range(6):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if mat[i, j] < 0.3 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pattern correlation")
    ax.set_title(f"{C.VAR_LONG[var]} — {model} skill by region & week  (JFM2026)",
                 fontweight="bold")
    C.savefig(fig, f"A3_region_scorecard_{var}_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(C.JFM, "results_1.5deg"))
    ap.add_argument("--vars", nargs="+", default=["TP", "T2M", "Z500"])
    ap.add_argument("--scorecard-model", default="MME")
    args = ap.parse_args()
    C.theme()
    det = pd.read_csv(os.path.join(args.results, "skill_deterministic.csv"))
    det = det[det.clim_basis == "era5"]
    print(f"A3 region profiles  (from {os.path.relpath(args.results, C.JFM)})")
    for var in args.vars:
        if var not in det.variable.unique():
            continue
        region_profiles(det, var)
        region_week_scorecard(det, var, args.scorecard_model)


if __name__ == "__main__":
    main()
