#!/usr/bin/env python3
"""Generate figures for the two-season India S2S benchmark paper.

Reads the verification-pipeline summary CSVs and writes publication PDFs into
paper_v2/figs/. Single-column arXiv-style figures, colour-blind-safe palette,
serif typography matching the paper body, panel labels, and a restrained
visual hierarchy (Spire as the narrative anchor, MME muted/dashed).

    python paper_v2/scripts/make_figures.py
"""
from __future__ import annotations

import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
os.makedirs(OUT, exist_ok=True)

# Paired-bootstrap 95% CIs (produced by make_bootstrap.py). Loaded lazily so
# the figures degrade gracefully to no-band mode if the file is absent.
_BOOT_CI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tables", "bootstrap_ci.csv"))


def _load_boot_ci():
    if not os.path.exists(_BOOT_CI_PATH):
        return None
    return pd.read_csv(_BOOT_CI_PATH)


# Maps the (det/prob summary) source key to the season key used in bootstrap_ci.
_BOOT_SEASON = {
    "jfm_tp": "jfm", "jfm_z500": "jfm",
    "jjas_tp_tp": "jjas_tp", "jjas_z500_z500": "jjas_z500",
    "jjas17_t2m": "jjas17",
}

DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/deterministic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/deterministic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/probabilistic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/probabilistic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/probabilistic_summary.csv",
}

# Okabe-Ito colour-blind-safe palette, fixed per model. Spire gets the
# heaviest visual weight since it anchors the paper's main narrative.
COLOR = {
    "spire": "#0072B2",   # blue
    "fuxi": "#D55E00",    # vermillion
    "delysm": "#009E73",  # green
    "ecmwf": "#4D4D4D",   # dark grey (was black; softer against text)
    "ukmo": "#CC79A7",    # purple-pink
    "ncep": "#E69F00",    # orange
    "mme": "#B0B0B0",     # light grey (dashed, background reference)
}
LABEL = {
    "spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "delysm": "DLESyM",
    "ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP", "mme": "MME",
}
ORDER = ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep", "mme"]
WEEKS = [1, 2, 3, 4, 5, 6]
REGIONS = {
    "northwest_india": "Northwest\nIndia",
    "central_india": "Central\nIndia",
    "south_peninsula": "South\nPeninsula",
    "east_northeast_india": "East/NE\nIndia",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})


def _allindia(path):
    df = pd.read_csv(path)
    return df[df["region"] == "All India"].copy()


def _panel_label(ax, letter):
    ax.text(-0.02, 1.10, f"({letter})", transform=ax.transAxes,
             fontsize=10, fontweight="bold", va="top", ha="left")


def _curve(ax, df, variable, value, boot_ci=None, boot_season=None, boot_metric=None):
    """Plot metric-vs-lead curves. When ``boot_ci`` and ``boot_season`` are
    given, overlay the paired-bootstrap 95% CI as a shaded band per model
    (drawn only for the narrative-anchor systems to avoid clutter)."""
    sub = df[df["variable"] == variable]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")

    # Systems whose CI band we actually shade: keep the plot readable by banding
    # only the anchor (Spire) plus the leading dynamical reference (ECMWF).
    band_models = {"spire", "ecmwf"}

    for m in ORDER:
        if m not in piv.index:
            continue
        y = [piv.loc[m, w] if w in piv.columns else None for w in WEEKS]
        if m == "mme":
            ls, lw, alpha, zorder = "--", 1.3, 0.85, 1
        elif m == "spire":
            ls, lw, alpha, zorder = "-", 2.4, 1.0, 5
        else:
            ls, lw, alpha, zorder = "-", 1.5, 0.95, 2

        if boot_ci is not None and boot_season is not None and m in band_models:
            bm = boot_metric if boot_metric is not None else value
            bsub = boot_ci[(boot_ci["season"] == boot_season)
                           & (boot_ci["variable"] == variable)
                           & (boot_ci["metric"] == bm)
                           & (boot_ci["model"] == m)].set_index("week")
            if not bsub.empty:
                lo = [bsub.loc[w, "ci_lo"] if w in bsub.index else None for w in WEEKS]
                hi = [bsub.loc[w, "ci_hi"] if w in bsub.index else None for w in WEEKS]
                ax.fill_between(WEEKS, lo, hi, color=COLOR[m], alpha=0.14,
                                lw=0, zorder=zorder - 1)

        ax.plot(WEEKS, y, ls, color=COLOR[m], lw=lw, alpha=alpha,
                 marker="o", ms=3.2 if m != "spire" else 4.0,
                 markeredgewidth=0, label=LABEL[m], zorder=zorder)
    ax.set_xticks(WEEKS)
    ax.set_xlabel("Lead week")
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _style_axis(ax):
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


# ----------------------------------------------------------------------
def fig_acc_lead():
    """2x2 grid: ACC vs lead for TP / Z500, JFM (top) & JJAS (bottom).
    T2M is excluded here (3 systems, no JJAS cross-model comparison) and
    shown separately in the appendix (fig_acc_lead_t2m)."""
    djfm = _allindia(DET["jfm"])
    djjas = pd.concat([_allindia(DET["jjas_tp"]), _allindia(DET["jjas_z500"])],
                      ignore_index=True)
    boot = _load_boot_ci()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharex=True)
    panels = [
        (axes[0, 0], djfm, "tp", "JFM 2026 — Precipitation", "jfm"),
        (axes[0, 1], djfm, "z500", "JFM 2026 — Z500", "jfm"),
        (axes[1, 0], djjas, "tp", "JJAS 2019 — Precipitation", "jjas_tp"),
        (axes[1, 1], djjas, "z500", "JJAS 2019 — Z500", "jjas_z500"),
    ]
    letters = string.ascii_lowercase
    for k, (ax, df, var, title, bseason) in enumerate(panels):
        _curve(ax, df, var, "acc", boot_ci=boot, boot_season=bseason, boot_metric="acc")
        ax.set_title(title, pad=14)
        _panel_label(ax, letters[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("ACC")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.8, w_pad=2.4)
    fig.savefig(f"{OUT}/fig_acc_lead.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead.pdf")


def fig_acc_lead_t2m():
    """Appendix figure: JJAS 2019 T2M ACC vs lead (DLESyM only).
    JFM 2026 T2M is intentionally omitted: the full_jfm2026_daily_spire run
    no longer scores t2m while its verification truth is being rebuilt."""
    djjas = _allindia(DET["jjas17"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    _curve(ax, djjas, "t2m", "acc")
    ax.set_title("JJAS 2019 — T2M", pad=14)
    ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
    ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlim(0.6, 6.4)
    _style_axis(ax)
    ax.set_ylabel("ACC")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.18), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(f"{OUT}/fig_acc_lead_t2m.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead_t2m.pdf")


def fig_crpss():
    """CRPSS vs lead, TP & Z500, JFM season (probabilistic story)."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    letters = string.ascii_lowercase
    for k, (ax, var, title) in enumerate([(axes[0], "tp", "Precipitation"),
                                          (axes[1], "z500", "Z500")]):
        _curve(ax, d, var, "crpss_clim")
        ax.set_title(f"JFM 2026 CRPSS — {title}", pad=14)
        _panel_label(ax, letters[k])
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.6, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    axes[0].set_ylabel("CRPSS vs. climatology")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.1), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.2)
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

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(piv.values, cmap="RdYlBu", vmin=0.15, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([REGIONS[c] for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models])
    for i, m in enumerate(models):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if pd.notna(v):
                txt_color = "white" if (v < 0.35 or v > 0.72) else "#222"
                weight = "bold" if m == "spire" else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color=txt_color, fontweight=weight)
    ax.set_title("JFM 2026 week-1 precipitation ACC by region", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ACC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.grid(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_regional_scorecard.pdf")
    plt.close(fig)
    print("wrote fig_regional_scorecard.pdf")


def fig_spread_skill():
    """Spread-skill ratio vs lead (calibration), JFM precip & Z500."""
    d = _allindia(PROB["jfm"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    letters = string.ascii_lowercase
    for k, (ax, var, title) in enumerate([(axes[0], "tp", "Precipitation"),
                                          (axes[1], "z500", "Z500")]):
        _curve(ax, d, var, "spread_skill_ratio")
        ax.set_title(f"JFM 2026 spread–skill — {title}", pad=14)
        _panel_label(ax, letters[k])
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", zorder=0)
        ax.text(6.3, 1.0, "ideal", fontsize=7.5, color="grey", va="bottom", ha="right")
        ax.set_ylim(0, 2.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    axes[0].set_ylabel("Spread–skill ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.1), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.2)
    fig.savefig(f"{OUT}/fig_spread_skill.pdf")
    plt.close(fig)
    print("wrote fig_spread_skill.pdf")


def main():
    fig_acc_lead()
    fig_acc_lead_t2m()
    fig_crpss()
    fig_regional_scorecard()
    fig_spread_skill()
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
