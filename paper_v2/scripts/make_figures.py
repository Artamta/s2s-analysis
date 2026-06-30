#!/usr/bin/env python3
"""Generate figures for the two-season India S2S benchmark paper.

Reads the verification-pipeline summary CSVs and writes publication PDFs into
paper_v2/figs/. Single-column arXiv-style figures, colour-blind-safe palette.

    python paper_v2/scripts/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
os.makedirs(OUT, exist_ok=True)

DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/probabilistic_summary.csv",
}

# Okabe-Ito colour-blind-safe palette, fixed per model.
COLOR = {
    "spire": "#0072B2",   # blue
    "fuxi": "#D55E00",    # vermillion
    "delysm": "#009E73",  # green
    "ecmwf": "#000000",   # black
    "ukmo": "#CC79A7",    # purple-pink
    "ncep": "#E69F00",    # orange
    "mme": "#999999",     # grey (dashed)
}
LABEL = {
    "spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "delysm": "DLESyM",
    "ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP", "mme": "MME",
}
ORDER = ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep", "mme"]
WEEKS = [1, 2, 3, 4, 5, 6]
REGIONS = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East/NE India",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})


def _allindia(path):
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _curve(ax, df, variable, value):
    sub = df[df["variable"] == variable]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")
    for m in ORDER:
        if m not in piv.index:
            continue
        y = [piv.loc[m, w] if w in piv.columns else None for w in WEEKS]
        ls = "--" if m == "mme" else "-"
        lw = 1.4 if m == "mme" else 1.8
        ax.plot(WEEKS, y, ls, color=COLOR[m], lw=lw, marker="o", ms=3.5,
                label=LABEL[m])
    ax.set_xticks(WEEKS)
    ax.set_xlabel("Lead week")


# ----------------------------------------------------------------------
def fig_acc_lead():
    """2x3 grid: ACC vs lead for TP / Z500 / T2M, JFM (top) & JJAS (bottom)."""
    djfm, djjas = _allindia(DET["jfm"]), _allindia(DET["jjas"])
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), sharex=True)
    panels = [
        (axes[0, 0], djfm, "tp", "JFM 2026 — Precipitation"),
        (axes[0, 1], djfm, "z500", "JFM 2026 — Z500"),
        (axes[0, 2], djfm, "t2m", "JFM 2026 — T2M"),
        (axes[1, 0], djjas, "tp", "JJAS 2019 — Precipitation"),
        (axes[1, 1], djjas, "z500", "JJAS 2019 — Z500"),
        (axes[1, 2], djjas, "t2m", "JJAS 2019 — T2M"),
    ]
    for ax, df, var, title in panels:
        _curve(ax, df, var, "acc")
        ax.set_title(title)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_ylim(-0.7, 1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("ACC")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/fig_acc_lead.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead.pdf")


def fig_crpss():
    """CRPSS vs lead, TP & Z500, JFM season (probabilistic story)."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharex=True)
    for ax, var, title in [(axes[0], "tp", "Precipitation"),
                           (axes[1], "z500", "Z500")]:
        _curve(ax, d, var, "crpss_clim")
        ax.set_title(f"JFM 2026 CRPSS — {title}")
        ax.axhline(0.0, color="grey", lw=0.7)
        ax.set_ylim(-0.6, 1.0)
    axes[0].set_ylabel("CRPSS vs climatology")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig_crpss.pdf")
    plt.close(fig)
    print("wrote fig_crpss.pdf")


def fig_regional_scorecard():
    """Heatmap-style scorecard: model x region, week-1 precip ACC, JFM."""
    df = pd.read_csv(DET["jfm"])
    df = df[(df["variable"] == "tp") & (df["week"] == 1)]
    df = df[df["region"].isin(REGIONS)]
    piv = df.pivot_table(index="model", columns="region", values="acc", aggfunc="mean")
    models = [m for m in ORDER if m in piv.index]
    cols = list(REGIONS.keys())
    piv = piv.reindex(models)[cols]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    im = ax.imshow(piv.values, cmap="RdYlBu", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([REGIONS[c] for c in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black")
    ax.set_title("JFM 2026 Week-1 precipitation ACC by region")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ACC")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_regional_scorecard.pdf")
    plt.close(fig)
    print("wrote fig_regional_scorecard.pdf")


def fig_spread_skill():
    """Spread-skill ratio vs lead (calibration), JFM precip & Z500."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharex=True)
    for ax, var, title in [(axes[0], "tp", "Precipitation"),
                           (axes[1], "z500", "Z500")]:
        _curve(ax, d, var, "spread_skill_ratio")
        ax.set_title(f"JFM 2026 spread-skill — {title}")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_ylim(0, 1.6)
    axes[0].set_ylabel("Spread-skill ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig_spread_skill.pdf")
    plt.close(fig)
    print("wrote fig_spread_skill.pdf")


def main():
    fig_acc_lead()
    fig_crpss()
    fig_regional_scorecard()
    fig_spread_skill()
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
