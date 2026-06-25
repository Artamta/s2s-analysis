#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis/monthwise.py  —  Stratify S2S forecast skill by VALID MONTH + IMD region.
================================================================================
Season-agnostic post-processor for the verification CSVs written by the driver
(`skill_deterministic.csv`, `skill_probabilistic.csv`, `skill_brier.csv`). It
derives the calendar month of the *valid* date from `init_date + lead` — so no
re-run is needed — then summarises and plots skill stratified by that month and
by IMD region. For JFM2026 inits this yields Jan/Feb/Mar (+ spill into Apr/May
at long leads); for a JJAS run it yields Jun/Jul/Aug/Sep. Nothing here is
hard-coded to a season: it uses whatever months actually appear in the data.

Valid-date derivation
---------------------
  scale='daily'  : valid = init_date + lead days        (lead is the lead DAY 1..42)
  scale='weekly' : valid = init_date + ((lead-1)*7 + 4)  (window CENTRE day; lead=week 1..6)

Lead buckets (W1-2 / W3-4 / W5-6)
---------------------------------
  weekly: lead (the week number) -> ceil(lead/2)
  daily : the calendar week ceil(lead/7) -> ceil(week/2)

Outputs (to --out)
------------------
  monthwise_summary.csv               tidy mean skill by [variable, model, valid_month, region, lead_bucket]
  heatmap_pcc_<VAR>.png               month x region weekly-mean PCC, one panel per model
  bars_pcc_<VAR>.png                  PCC by valid_month per model, faceted by region
  heatmap_crpss_<VAR>.png             month x region weekly-mean CRPSS, one panel per model (if prob present)
  + a compact month x region PCC table printed to stdout

Run
---
  python monthwise.py --results ../jfm2026/results_1.5deg --out ../jfm2026/results_1.5deg/monthwise
  python monthwise.py --results ... --basis model_own
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
sys.path.append(os.path.dirname(HERE))                      # final_analysis/ on path
from core.plotting import apply_theme, style_for, VAR_UNITS, MODEL_STYLE  # noqa: E402

# ------------------------------------------------------------------- constants --
REGION_TITLE = {
    "All India": "All India", "northwest_india": "Northwest India",
    "central_india": "Central India", "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & NE India",
}
REGION_ORDER = ["All India", "northwest_india", "central_india",
                "south_peninsula", "east_northeast_india"]
DET_ORDER  = ["SPIRE", "FuXi", "ECMWF", "MME", "Persistence"]
PROB_ORDER = ["SPIRE", "FuXi", "ECMWF"]
BUCKETS    = ["W1-2", "W3-4", "W5-6"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------- valid-month + buckets --
def add_valid_month(df):
    """Tag each row with valid_date / valid_month (number) / valid_month_name
       and a W1-2/W3-4/W5-6 lead bucket. Season-agnostic."""
    df = df.copy()
    init = pd.to_datetime(df["init_date"])
    lead = df["lead"].astype(int)

    daily = df["scale"] == "daily"
    # daily: + lead days ; weekly: + window-centre offset of a week-number lead
    offset = np.where(daily, lead, (lead - 1) * 7 + 4)
    valid = init + pd.to_timedelta(offset, unit="D")

    df["valid_date"] = valid
    df["valid_month"] = valid.dt.month
    df["valid_month_name"] = valid.dt.strftime("%b")

    # lead -> calendar week (1..6) -> bucket index (1..3)
    week = np.where(daily, np.ceil(lead / 7.0), lead).astype(int)
    week = np.clip(week, 1, 6)
    bucket_idx = np.ceil(week / 2.0).astype(int)            # 1,2 -> 1 ; 3,4 -> 2 ; 5,6 -> 3
    df["lead_bucket"] = [BUCKETS[i - 1] for i in bucket_idx]
    return df


def month_order(df):
    """Months present, in calendar order (handles JFM, JJAS, anything)."""
    present = sorted(df["valid_month"].dropna().unique().astype(int))
    return present, [MONTH_ABBR[m - 1] for m in present]


# ------------------------------------------------------------------- load CSVs --
def load(results, basis):
    det = pd.read_csv(os.path.join(results, "skill_deterministic.csv"))
    prob_path = os.path.join(results, "skill_probabilistic.csv")
    brier_path = os.path.join(results, "skill_brier.csv")
    prob = pd.read_csv(prob_path) if os.path.exists(prob_path) else None
    brier = pd.read_csv(brier_path) if os.path.exists(brier_path) else None

    def _basis(df):
        if df is None or "clim_basis" not in df.columns:
            return df
        avail = set(df["clim_basis"].dropna().unique())
        if basis not in avail:
            print(f"  [warn] clim_basis='{basis}' absent (have {sorted(avail)}); "
                  f"keeping all rows.")
            return df
        return df[df["clim_basis"] == basis]

    det, prob, brier = _basis(det), _basis(prob), _basis(brier)
    det = add_valid_month(det)
    prob = add_valid_month(prob) if prob is not None else None
    brier = add_valid_month(brier) if brier is not None else None
    return det, prob, brier


# --------------------------------------------------------------- tidy summary --
def build_summary(det, prob, brier):
    """Mean skill grouped by [variable, model, valid_month, region, lead_bucket].
       Uses the WEEKLY scale (one clean value per week-window); brier is weekly
       and averaged over its events into a single BSS column."""
    keys = ["variable", "model", "valid_month", "valid_month_name",
            "region", "lead_bucket"]

    det_w = det[det["scale"] == "weekly"]
    det_g = (det_w.groupby(keys, observed=True)
             .agg(pcc=("pcc", "mean"), rmse=("rmse", "mean"),
                  bias=("bias", "mean"), msss_clim=("msss_clim", "mean"),
                  std_ratio=("std_ratio", "mean"), n_init=("init_date", "nunique"))
             .reset_index())
    summary = det_g

    if prob is not None and len(prob):
        prob_w = prob[prob["scale"] == "weekly"]
        prob_g = (prob_w.groupby(keys, observed=True)
                  .agg(crpss_clim=("crpss_clim", "mean"),
                       crps=("crps", "mean"), ssr=("ssr", "mean"),
                       spread=("spread", "mean"))
                  .reset_index())
        summary = summary.merge(prob_g, on=keys, how="outer")

    if brier is not None and len(brier):
        # average BSS over all events -> one value per (var, model, month, region, bucket)
        bss_g = (brier.groupby(keys, observed=True)
                 .agg(briss_clim=("briss_clim", "mean"))
                 .reset_index())
        summary = summary.merge(bss_g, on=keys, how="outer")

    # stable, readable ordering
    months, _ = month_order(summary)
    summary["valid_month"] = summary["valid_month"].astype(int)
    summary["_morder"] = summary["valid_month"].map({m: i for i, m in enumerate(months)})
    summary["_rorder"] = summary["region"].map(
        {r: i for i, r in enumerate(REGION_ORDER)}).fillna(99)
    summary["_border"] = summary["lead_bucket"].map(
        {b: i for i, b in enumerate(BUCKETS)})
    summary = (summary.sort_values(
        ["variable", "model", "_morder", "_rorder", "_border"])
        .drop(columns=["_morder", "_rorder", "_border"])
        .reset_index(drop=True))
    return summary


# ----------------------------------------------------------- month x region PCC --
def month_region_table(det, variable, model, value="pcc"):
    """Weekly-mean `value` averaged over all leads+inits -> DataFrame
       indexed by month (rows) x region (cols)."""
    sub = det[(det["scale"] == "weekly") & (det["variable"] == variable)
              & (det["model"] == model)]
    if not len(sub):
        return None
    piv = (sub.groupby(["valid_month", "region"], observed=True)[value]
           .mean().reset_index()
           .pivot(index="valid_month", columns="region", values=value))
    cols = [r for r in REGION_ORDER if r in piv.columns]
    piv = piv.reindex(columns=cols)
    piv.index = [MONTH_ABBR[m - 1] for m in piv.index]
    return piv


# ------------------------------------------------------------------- figures ----
def _models_in(df, order):
    have = set(df["model"].unique())
    return [m for m in order if m in have]


def fig_heatmap(det, variable, value, out, *, order, cbar_label, fname,
                vmin=None, vmax=None, cmap="viridis"):
    """month x region heatmap of weekly-mean `value`, one panel per model."""
    models = _models_in(det[det["scale"] == "weekly"], order)
    tables = {m: month_region_table(det, variable, m, value) for m in models}
    tables = {m: t for m, t in tables.items() if t is not None and t.notna().any().any()}
    if not tables:
        return None

    months_present = sorted(
        {mi for t in tables.values() for mi in t.index},
        key=lambda x: MONTH_ABBR.index(x))
    regions = REGION_ORDER

    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 1.4, 0.7 * len(months_present) + 2.2),
                             squeeze=False)
    axes = axes[0]
    if vmin is None or vmax is None:
        allv = np.concatenate([t.values[np.isfinite(t.values)] for t in tables.values()])
        vmin = np.nanpercentile(allv, 2) if vmin is None else vmin
        vmax = np.nanpercentile(allv, 98) if vmax is None else vmax
    im = None
    for ax, (m, t) in zip(axes, tables.items()):
        t = t.reindex(index=months_present, columns=regions)
        im = ax.imshow(t.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([REGION_TITLE.get(r, r) for r in regions],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(months_present)))
        ax.set_yticklabels(months_present, fontsize=9)
        ax.set_title(MODEL_STYLE.get(m, {}).get("label", m))
        ax.grid(False)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label=cbar_label)
    unit = VAR_UNITS.get(variable, "")
    fig.suptitle(f"{variable} ({unit}) — {cbar_label} by valid-month x region",
                 fontsize=13, fontweight="bold")
    path = os.path.join(out, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_bars_pcc(det, variable, out):
    """Grouped bars: weekly-mean PCC by valid_month for each model, faceted by region."""
    sub = det[(det["scale"] == "weekly") & (det["variable"] == variable)]
    if not len(sub):
        return None
    models = _models_in(sub, DET_ORDER)
    months, month_lbls = month_order(sub)
    g = (sub.groupby(["region", "model", "valid_month"], observed=True)["pcc"]
         .mean().reset_index())

    regions = [r for r in REGION_ORDER if r in sub["region"].unique()]
    ncol = 3
    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                             squeeze=False, sharey=True)
    axes = axes.ravel()
    x = np.arange(len(months))
    width = 0.8 / max(len(models), 1)
    for ax, region in zip(axes, regions):
        rg = g[g["region"] == region]
        for k, m in enumerate(models):
            vals = [rg[(rg.model == m) & (rg.valid_month == mo)]["pcc"].mean()
                    for mo in months]
            st = style_for(m)
            ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width,
                   color=st["color"], label=st["label"], edgecolor="white", linewidth=0.4)
        ax.axhline(0.5, color="0.4", lw=1.0, ls="--")
        ax.set_title(REGION_TITLE.get(region, region))
        ax.set_xticks(x)
        ax.set_xticklabels(month_lbls)
        ax.set_ylabel("PCC")
    for ax in axes[len(regions):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.06),
               ncol=1, fontsize=9)
    unit = VAR_UNITS.get(variable, "")
    fig.suptitle(f"{variable} ({unit}) — weekly PCC by valid-month, per region",
                 fontsize=13, fontweight="bold")
    path = os.path.join(out, f"bars_pcc_{variable}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- driver --
def main():
    ap = argparse.ArgumentParser(
        description="Stratify S2S skill by VALID-MONTH and IMD region.")
    ap.add_argument("--results", required=True, help="dir with the 3 skill CSVs")
    ap.add_argument("--out", required=True, help="dir for summary CSV + figures")
    ap.add_argument("--basis", default="era5",
                    help="clim_basis to keep (era5 | model_own); default era5")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    apply_theme()

    det, prob, brier = load(args.results, args.basis)
    variables = sorted(det["variable"].unique())
    months, month_lbls = month_order(det)
    print(f"\nResults : {args.results}")
    print(f"Basis   : {args.basis}")
    print(f"Variables: {variables}")
    print(f"Valid months present: {month_lbls}\n")

    # 1) tidy summary CSV ------------------------------------------------------
    summary = build_summary(det, prob, brier)
    summ_path = os.path.join(args.out, "monthwise_summary.csv")
    summary.to_csv(summ_path, index=False)
    print(f"[csv]  {summ_path}  ({len(summary)} rows)")

    # 2) figures ---------------------------------------------------------------
    made = []
    for var in variables:
        p = fig_heatmap(det, var, "pcc", args.out, order=DET_ORDER,
                        cbar_label="PCC", fname=f"heatmap_pcc_{var}.png",
                        vmin=-0.2, vmax=1.0, cmap="RdYlGn")
        if p:
            made.append(p)
        p = fig_bars_pcc(det, var, args.out)
        if p:
            made.append(p)
        if prob is not None and len(prob):
            p = fig_heatmap(prob, var, "crpss_clim", args.out, order=PROB_ORDER,
                            cbar_label="CRPSS", fname=f"heatmap_crpss_{var}.png",
                            vmin=-0.5, vmax=0.5, cmap="RdBu_r")
            if p:
                made.append(p)
    for p in made:
        print(f"[fig]  {p}")

    # 3) stdout month x region PCC tables (weekly-mean, all leads) -------------
    print("\n" + "=" * 72)
    print("MONTH x REGION  weekly-mean PCC  (averaged over all leads + inits)")
    print("=" * 72)
    for var in variables:
        models = _models_in(det[det["scale"] == "weekly"], DET_ORDER)
        for m in models:
            tab = month_region_table(det, var, m, "pcc")
            if tab is None:
                continue
            print(f"\n--- {var} / {m} ---")
            with pd.option_context("display.width", 140,
                                   "display.float_format", lambda v: f"{v:6.3f}"):
                print(tab.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
