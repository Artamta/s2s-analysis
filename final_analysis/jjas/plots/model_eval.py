#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/plots/model_eval.py — PRESENTATION model-evaluation figures (JJAS 2019).
================================================================================
Reads the JJAS results CSVs (skill_deterministic / skill_probabilistic) and
builds crisp, big-font slide figures for the ECMWF reforecast vs ERA5 story
(plus FuXi where it was scored). Output PNGs -> jjas/plots/figs/model_eval/.

Figures:
  A  ecmwf_skill_vs_lead   PCC + RMSE vs week (W1-6), TP & Z500, All-India.
  B  dual_basis_pcc        ERA5 vs MODEL-OWN climatology PCC (paired bars),
                           the headline: ECMWF rain skill drops on its own
                           (wet-biased) climatology; circulation barely moves.
  C1 crpss_by_week         CRPSS vs climatology by week (TP, Z500).
  C2 spread_skill          spread-skill ratio (SSR) vs week — calibration.
  E  fuxi_vs_ecmwf         small FuXi-vs-ECMWF panel for the FuXi init (n=1).

Run (in s2s-hind env):
  python plots/model_eval.py \
    --results /storage/raj.ayush/s2s_final_data/jjas/results_2019_1.5deg
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
from core.plotting import style_for                          # noqa: E402

VAR_TITLE = {"TP": "Rainfall (TP)", "Z500": "500 hPa height (Z500)"}
VAR_UNITS = {"TP": "mm/day", "Z500": "gpm"}
WEEKS = list(range(1, 7))
WEEK_LBL = [f"W{i}" for i in range(1, 7)]


def theme():
    matplotlib.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "font.size": 16,
        "axes.titlesize": 19, "axes.titleweight": "bold", "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True, "lines.linewidth": 3.2,
        "lines.markersize": 9,
    })


def _save(fig, out, name, msg=None):
    if msg:
        fig.text(0.5, -0.03, msg, ha="center", fontsize=13, style="italic", color="0.3")
    p = os.path.join(out, name + ".png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def _weekly_mean(df, val, var, model, basis=None, region="All India"):
    """Per-week mean of `val` over inits -> arrays aligned to WEEKS (NaN if absent)."""
    s = df[(df.scale == "weekly") & (df.region == region) &
           (df.variable == var) & (df.model == model)]
    if basis is not None and "clim_basis" in s.columns:
        s = s[s.clim_basis == basis]
    g = s.groupby("lead")[val].mean()
    return np.array([g.get(wk, np.nan) for wk in WEEKS], float)


# ── A. ECMWF skill (PCC + RMSE) vs lead ──────────────────────────────────────
def figA_skill_vs_lead(det, out, model="ECMWF"):
    vars_ = [v for v in ("TP", "Z500") if v in det.variable.unique()]
    fig, axes = plt.subplots(2, len(vars_), figsize=(7.0 * len(vars_), 9.6), squeeze=False)
    st = style_for(model)
    for j, var in enumerate(vars_):
        pcc = _weekly_mean(det, "pcc", var, model, basis="era5")
        rmse = _weekly_mean(det, "rmse", var, model, basis="era5")
        # PCC (top)
        ax = axes[0][j]
        ax.plot(WEEKS, pcc, color=st["color"], marker=st["marker"])
        ax.axhline(0.5, color="0.4", ls="--", lw=1.6)
        ax.set_ylim(min(-0.1, np.nanmin(pcc) - 0.05), 1.03)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WEEK_LBL)
        ax.set_ylabel("Pattern correlation"); ax.set_title(VAR_TITLE.get(var, var))
        # RMSE (bottom)
        ax = axes[1][j]
        ax.plot(WEEKS, rmse, color=st["color"], marker=st["marker"])
        ax.set_xticks(WEEKS); ax.set_xticklabels(WEEK_LBL)
        ax.set_xlabel("Forecast lead")
        ax.set_ylabel(f"RMSE ({VAR_UNITS.get(var, '')})")
    fig.suptitle(f"{model} forecast skill vs lead — JJAS 2019, All-India",
                 fontsize=22, fontweight="bold")
    _save(fig, out, "A_ecmwf_skill_vs_lead",
          "top: pattern correlation (dashed = 0.5 skill line);  bottom: RMSE.  Skill falls with lead.")


# ── B. Dual-basis PCC: ERA5 vs model-own climatology ─────────────────────────
def figB_dual_basis(det, out, model="ECMWF"):
    vars_ = [v for v in ("TP", "Z500") if v in det.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 6.0), squeeze=False)
    x = np.arange(len(WEEKS)); bw = 0.38
    for j, var in enumerate(vars_):
        ax = axes[0][j]
        era = _weekly_mean(det, "pcc", var, model, basis="era5")
        own = _weekly_mean(det, "pcc", var, model, basis="model_own")
        ax.bar(x - bw / 2, era, bw, label="vs ERA5 climatology",
               color="#009E73", edgecolor="white")
        ax.bar(x + bw / 2, own, bw, label="vs model-own climatology",
               color="#D55E00", edgecolor="white")
        for xi, (a, b) in enumerate(zip(era, own)):
            if np.isfinite(a):
                ax.text(xi - bw / 2, a + 0.012, f"{a:.2f}", ha="center", va="bottom", fontsize=11)
            if np.isfinite(b):
                ax.text(xi + bw / 2, b + 0.012, f"{b:.2f}", ha="center", va="bottom", fontsize=11)
        ax.axhline(0.5, color="0.4", ls="--", lw=1.4)
        ax.set_xticks(x); ax.set_xticklabels(WEEK_LBL)
        ax.set_ylim(0, 1.08); ax.set_ylabel("Pattern correlation")
        ax.set_title(VAR_TITLE.get(var, var))
        ax.legend(loc="lower left", fontsize=13)
    fig.suptitle(f"{model}: skill collapses on its OWN climatology for rainfall, not circulation",
                 fontsize=20, fontweight="bold")
    _save(fig, out, "B_dual_basis_pcc",
          "ECMWF is wet-biased: scored on its own climatology, rainfall PCC drops sharply; Z500 barely changes.")


# ── C1. CRPSS by week ────────────────────────────────────────────────────────
def figC1_crpss(prob, out, models=("ECMWF", "FuXi")):
    vars_ = [v for v in ("TP", "Z500") if v in prob.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 6.0), squeeze=False)
    for j, var in enumerate(vars_):
        ax = axes[0][j]
        for m in models:
            v = _weekly_mean(prob, "crpss_clim", var, m)
            if np.isfinite(v).any():
                st = style_for(m)
                ax.plot(WEEKS, v, color=st["color"], marker=st["marker"], label=m)
        ax.axhline(0.0, color="0.4", ls="--", lw=1.6)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WEEK_LBL)
        ax.set_xlabel("Forecast lead"); ax.set_ylabel("CRPSS vs climatology")
        ax.set_title(VAR_TITLE.get(var, var)); ax.legend(loc="best")
    fig.suptitle("Probabilistic skill above climatology — CRPSS, JJAS 2019, All-India",
                 fontsize=21, fontweight="bold")
    _save(fig, out, "C1_crpss_by_week", "above 0 = beats a climatology forecast.")


# ── C2. Spread-skill ratio ───────────────────────────────────────────────────
def figC2_spread_skill(prob, out, models=("ECMWF", "FuXi")):
    vars_ = [v for v in ("TP", "Z500") if v in prob.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 6.0), squeeze=False)
    for j, var in enumerate(vars_):
        ax = axes[0][j]
        for m in models:
            v = _weekly_mean(prob, "ssr", var, m)
            if np.isfinite(v).any():
                st = style_for(m)
                ax.plot(WEEKS, v, color=st["color"], marker=st["marker"], label=m)
        ax.axhline(1.0, color="0.4", ls="--", lw=1.6)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WEEK_LBL)
        ax.set_xlabel("Forecast lead"); ax.set_ylabel("Spread / RMSE")
        ax.set_title(VAR_TITLE.get(var, var)); ax.legend(loc="best")
    fig.suptitle("Ensemble calibration — spread-skill ratio, JJAS 2019, All-India",
                 fontsize=21, fontweight="bold")
    _save(fig, out, "C2_spread_skill_ratio",
          "ratio ~1 = well-calibrated;  <1 = under-dispersive (over-confident),  >1 = over-dispersive.")


# ── E. FuXi (single init) vs ECMWF (multi-init mean) — n=1 caveat ─────────────
def figE_fuxi_vs_ecmwf(det, prob, out, fuxi_init="2019-06-20"):
    """The one extracted FuXi init has no co-located ECMWF reforecast file, so we
    plot FuXi (this single init) against the ECMWF multi-init mean for reference.
    Clearly flagged as illustrative (n=1 FuXi init)."""
    fux = det[(det.model == "FuXi") & (det.scale == "weekly") &
              (det.init_date == fuxi_init)]
    if not len(fux):
        print("  [E] FuXi not scored — skipping FuXi panel.")
        return
    vars_ = [v for v in ("TP", "Z500") if v in det.variable.unique()]
    fig, axes = plt.subplots(1, len(vars_), figsize=(7.5 * len(vars_), 6.0), squeeze=False)
    for j, var in enumerate(vars_):
        ax = axes[0][j]
        # ECMWF reference = mean over all scored ECMWF inits (era5 basis)
        ec = _weekly_mean(det, "pcc", var, "ECMWF", basis="era5")
        st = style_for("ECMWF")
        ax.plot(WEEKS, ec, color=st["color"], marker=st["marker"], lw=2.6,
                ls="--", label="ECMWF (mean of all inits)")
        # FuXi single init
        s = fux[(fux.region == "All India") & (fux.variable == var) &
                (fux.clim_basis == "era5")]
        g = s.groupby("lead")["pcc"].mean()
        v = np.array([g.get(wk, np.nan) for wk in WEEKS], float)
        st = style_for("FuXi")
        ax.plot(WEEKS, v, color=st["color"], marker=st["marker"],
                label=f"FuXi (init {fuxi_init})")
        ax.axhline(0.5, color="0.4", ls=":", lw=1.4)
        ax.set_xticks(WEEKS); ax.set_xticklabels(WEEK_LBL)
        ax.set_ylim(-0.2, 1.03)
        ax.set_xlabel("Forecast lead"); ax.set_ylabel("Pattern correlation")
        ax.set_title(VAR_TITLE.get(var, var)); ax.legend(loc="best", fontsize=12)
    fig.suptitle(f"FuXi (single init {fuxi_init}) vs ECMWF (multi-init mean)  —  illustrative, n=1 FuXi init",
                 fontsize=18, fontweight="bold")
    _save(fig, out, "E_fuxi_vs_ecmwf",
          "FuXi has ONLY this one extracted init; ECMWF line is its multi-init mean. Not a robust head-to-head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "model_eval"))
    ap.add_argument("--region", default="All India")
    args = ap.parse_args()
    theme()
    os.makedirs(args.out, exist_ok=True)
    print(f"Reading {args.results} -> {args.out}")

    det = pd.read_csv(os.path.join(args.results, "skill_deterministic.csv"))
    prob = pd.read_csv(os.path.join(args.results, "skill_probabilistic.csv"))
    if not len(det):
        sys.exit("ERROR: skill_deterministic.csv is empty — run the verification first.")

    figA_skill_vs_lead(det, args.out)
    figB_dual_basis(det, args.out)
    if len(prob):
        figC1_crpss(prob, args.out)
        figC2_spread_skill(prob, args.out)
    figE_fuxi_vs_ecmwf(det, prob, args.out)
    print(f"\nDONE -> {args.out}")


if __name__ == "__main__":
    main()
