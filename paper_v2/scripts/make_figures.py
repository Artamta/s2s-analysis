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
import numpy as np
import pandas as pd
import xarray as xr

from paper_paths import IMD_MASK_025, PAPER_OUTPUT_ROOT

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:  # pragma: no cover - spatial figures degrade gracefully
    _HAS_CARTOPY = False

ROOT = str(PAPER_OUTPUT_ROOT)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

IMD_CLIM_SEASONAL = (
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/"
    "climatology/imd_rain_1991_2020_seasonal_climatology.nc"
)
WB2_ERA5_025_ZARR = (
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
ERA5_JJAS_TP_CLIM_CACHE = os.path.join(CACHE, "era5_jjas_1991_2020_tp_clim_025.nc")

DET = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/deterministic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/deterministic_summary.csv",
    "jjas_tp_era5": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_era5truth/deterministic_summary.csv",
    "jjas_z500": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_z500/deterministic_summary.csv",
    "jjas17": f"{ROOT}/jjas2019/05_tables/full_jjas2019_common17_fuxi_imd/deterministic_summary.csv",
}
PROB = {
    "jfm": f"{ROOT}/jfm2026/05_tables/full_jfm2026_daily_spire/probabilistic_summary.csv",
    "jjas_tp": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/probabilistic_summary.csv",
    "jjas_tp_era5": f"{ROOT}/jjas2019/05_tables/full_jjas2019_operational35_plus_fuxi_tp_era5truth/probabilistic_summary.csv",
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


def _curve(ax, df, variable, value):
    """Plot metric-vs-lead curves."""
    sub = df[df["variable"] == variable]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")

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


def _legend_union(axes):
    """Handles/labels from every axis, deduped and in canonical model ORDER,
    so a model plotted in only one panel (e.g. DLESyM, Z500-only) still gets
    a legend entry."""
    by_label = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(l, h)
    order = [LABEL[m] for m in ORDER]
    labels = sorted(by_label, key=lambda l: order.index(l) if l in order else 99)
    return [by_label[l] for l in labels], labels


# ----------------------------------------------------------------------
def fig_acc_lead():
    """Main Figure 2: JFM and JJAS precipitation ACC side by side."""
    jfm_all = _allindia(DET["jfm"])
    jfm_regional = pd.read_csv(DET["jfm"])
    jjas_all = _allindia(DET["jjas_tp_era5"])
    jjas_regional = pd.read_csv(DET["jjas_tp_era5"])

    rows = [("All India", None)]
    rows.extend((REGION_PANEL[r], r) for r in REGION_PANEL)
    seasons = [
        ("JJAS 2019", jjas_all, jjas_regional),
        ("JFM 2026", jfm_all, jfm_regional),
    ]

    fig, axes = plt.subplots(
        len(rows), 2, figsize=(7.35, 8.15), sharex=True, sharey=True
    )
    letters = string.ascii_lowercase

    for i, (_row_label, region) in enumerate(rows):
        for j, (season_label, all_india, regional) in enumerate(seasons):
            ax = axes[i, j]
            if region is None:
                _curve(ax, all_india, "tp", "acc")
            else:
                _regional_curve(ax, regional, "tp", region, "acc")
            ax.text(0.02, 0.92, f"({letters[i * 2 + j]})",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="top", ha="left")
            ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
            ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
            ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
            ax.set_ylim(-0.05, 0.90)
            ax.set_xlim(0.6, 6.4)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            _style_axis(ax)
            if j == 0:
                ax.set_ylabel("ACC")
            if i == len(rows) - 1:
                ax.set_xlabel("Lead week")
            else:
                ax.set_xlabel("")
            if i == 0:
                ax.set_title(f"{season_label} (ERA5 truth)", pad=10)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.94, bottom=0.115,
                        hspace=0.28, wspace=0.18)

    for i, (row_label, _region) in enumerate(rows):
        bbox = axes[i, 0].get_position()
        y = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.030, y, row_label, va="center", ha="center",
                 rotation=90, fontsize=9.5, fontweight="bold")

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.55, -0.005), frameon=False,
               columnspacing=0.95, handletextpad=0.4)
    fig.savefig(f"{OUT}/fig_acc_lead.pdf")
    plt.close(fig)
    print("wrote fig_acc_lead.pdf")


def fig_z500_acc_appendix():
    """Appendix-only all-India Z500 ACC context for both case studies."""
    jfm = _allindia(DET["jfm"])
    jjas = _allindia(DET["jjas_z500"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharex=True, sharey=True)
    panels = [
        (axes[0], jfm, "JFM 2026"),
        (axes[1], jjas, "JJAS 2019"),
    ]
    for k, (ax, df, title) in enumerate(panels):
        _curve(ax, df, "z500", "acc")
        ax.set_title(f"{title} Z500 ACC", pad=12)
        _panel_label(ax, string.ascii_lowercase[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_xlim(0.6, 6.4)
        ax.set_ylabel("ACC")
        _style_axis(ax)
    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.11), frameon=False,
               columnspacing=1.0, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2.2)
    fig.savefig(f"{OUT}/fig_z500_acc_appendix.pdf")
    plt.close(fig)
    print("wrote fig_z500_acc_appendix.pdf")


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


def fig_jjas_reference_sensitivity():
    """Appendix sensitivity: JJAS TP scores under gauge vs ERA5 references."""
    det_imd = _allindia(DET["jjas_tp"])
    det_era5 = _allindia(DET["jjas_tp_era5"])
    prob_imd = _allindia(PROB["jjas_tp"])
    prob_era5 = _allindia(PROB["jjas_tp_era5"])

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    panels = [
        (axes[0, 0], det_era5, "acc", "ACC (ERA5 truth)"),
        (axes[0, 1], det_imd, "acc", "ACC (IMD truth)"),
        (axes[1, 0], prob_era5, "crpss_clim", "CRPSS (ERA5 truth)"),
        (axes[1, 1], prob_imd, "crpss_clim", "CRPSS (IMD truth)"),
    ]
    letters = string.ascii_lowercase
    for k, (ax, df, metric, title) in enumerate(panels):
        _curve(ax, df, "tp", metric)
        ax.set_title(title, pad=14)
        _panel_label(ax, letters[k])
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
        if metric == "acc":
            ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
            ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
            ax.set_ylim(-0.2, 0.9)
        else:
            ax.set_ylim(-0.15, 0.6)

    axes[0, 0].set_ylabel("ACC")
    axes[0, 1].set_ylabel("ACC")
    axes[1, 0].set_ylabel("CRPSS vs. climatology")
    axes[1, 1].set_ylabel("CRPSS vs. climatology")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[1, :]:
        ax.set_xlabel("Lead week")

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), frameon=False,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=2.5, w_pad=2.2)
    fig.savefig(f"{OUT}/fig_jjas_reference_sensitivity.pdf")
    plt.close(fig)
    print("wrote fig_jjas_reference_sensitivity.pdf")


def fig_jjas_era5_tp():
    """Main-text JJAS precipitation skill under ERA5 verification."""
    det_era5 = _allindia(DET["jjas_tp_era5"])
    prob_era5 = _allindia(PROB["jjas_tp_era5"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    panels = [
        (axes[0], det_era5, "acc", "ACC (ERA5 truth)", "ACC", (-0.05, 0.9)),
        (axes[1], prob_era5, "crpss_clim", "CRPSS (ERA5 truth)",
         "CRPSS vs. climatology", (-0.15, 0.6)),
    ]
    for k, (ax, df, metric, title, ylabel, ylim) in enumerate(panels):
        _curve(ax, df, "tp", metric)
        ax.set_title(title, pad=12)
        _panel_label(ax, string.ascii_lowercase[k])
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        if metric == "acc":
            ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
            ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.set_ylim(*ylim)
        ax.set_xlim(0.6, 6.4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Lead week")
        _style_axis(ax)

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.10), frameon=False,
               columnspacing=1.1, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.2)
    fig.savefig(f"{OUT}/fig_jjas_era5_tp.pdf")
    plt.close(fig)
    print("wrote fig_jjas_era5_tp.pdf")


def _set_positive_ylim(ax):
    vals = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        vals.extend(y[np.isfinite(y)])
    if not vals:
        return
    ymax = max(vals) * 1.12
    ax.set_ylim(0, ymax)


def fig_error_scores():
    """Main-text all-India RMSE and CRPS, with JJAS first."""
    det_jjas = _allindia(DET["jjas_tp_era5"])
    det_jfm = _allindia(DET["jfm"])
    prob_jjas = _allindia(PROB["jjas_tp_era5"])
    prob_jfm = _allindia(PROB["jfm"])

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.95), sharex=True)
    panels = [
        (axes[0, 0], det_jjas, "rmse", "JJAS 2019", "RMSE (mm day$^{-1}$)"),
        (axes[0, 1], det_jfm, "rmse", "JFM 2026", "RMSE (mm day$^{-1}$)"),
        (axes[1, 0], prob_jjas, "crps", "JJAS 2019", "CRPS (mm day$^{-1}$)"),
        (axes[1, 1], prob_jfm, "crps", "JFM 2026", "CRPS (mm day$^{-1}$)"),
    ]
    for k, (ax, df, metric, title, ylabel) in enumerate(panels):
        _curve(ax, df, "tp", metric)
        ax.set_title(title if k < 2 else "", pad=10)
        ax.text(0.02, 0.92, f"({string.ascii_lowercase[k]})",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.6, 6.4)
        _set_positive_ylim(ax)
        _style_axis(ax)
        if k < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Lead week")

    axes[0, 0].text(-0.18, 0.5, "Deterministic error", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9.5,
                    fontweight="bold")
    axes[1, 0].text(-0.18, 0.5, "Probabilistic error", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9.5,
                    fontweight="bold")

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               columnspacing=0.95, handletextpad=0.4)
    fig.tight_layout(rect=(0.04, 0.09, 1, 1), h_pad=1.7, w_pad=1.5)
    fig.savefig(f"{OUT}/fig_error_scores.pdf")
    plt.close(fig)
    print("wrote fig_error_scores.pdf")


def fig_jjas17_z500_sensitivity():
    """Appendix sensitivity: DLESyM-inclusive JJAS Z500 on 17 common inits."""
    d = _allindia(DET["jjas17"])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    _curve(ax, d, "z500", "acc")
    ax.set_title("JJAS 2019 Z500 — 17-init DLESyM-inclusive subset", pad=14)
    ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
    ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlim(0.6, 6.4)
    ax.set_ylabel("ACC")
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), frameon=False,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"{OUT}/fig_jjas17_z500_sensitivity.pdf")
    plt.close(fig)
    print("wrote fig_jjas17_z500_sensitivity.pdf")


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


REGION_PANEL = {
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East/NE India",
}


def _regional_curve(ax, df, variable, region, value="acc"):
    """One region panel: metric vs lead for every model, filtered to a region."""
    sub = df[(df["variable"] == variable) & (df["region"] == region)]
    piv = sub.pivot_table(index="model", columns="week", values=value, aggfunc="mean")
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
        ax.plot(WEEKS, y, ls, color=COLOR[m], lw=lw, alpha=alpha,
                marker="o", ms=3.2 if m != "spire" else 4.0,
                markeredgewidth=0, label=LABEL[m], zorder=zorder)
    ax.set_xticks(WEEKS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def fig_jjas_era5_acc_lead():
    """Main-text JJAS 2019 ERA5 precipitation ACC, matching the JFM layout."""
    all_india = _allindia(DET["jjas_tp_era5"])
    regional = pd.read_csv(DET["jjas_tp_era5"])

    fig = plt.figure(figsize=(7.25, 7.25))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.52, wspace=0.58)
    axes = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
    ]

    panels = [("All India", axes[0], None)]
    panels.extend((REGION_PANEL[r], axes[i + 1], r)
                  for i, r in enumerate(REGION_PANEL))

    for k, (title, ax, region) in enumerate(panels):
        if region is None:
            _curve(ax, all_india, "tp", "acc")
        else:
            _regional_curve(ax, regional, "tp", region, "acc")
        ax.set_title(title, pad=12)
        _panel_label(ax, string.ascii_lowercase[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.05, 0.90)
        ax.set_xlim(0.6, 6.4)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.set_ylabel("ACC")
        _style_axis(ax)

    axes[0].set_xlabel("")
    for ax in axes[1:3]:
        ax.set_xlabel("")
    for ax in axes[3:]:
        ax.set_xlabel("Lead week")

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.015), frameon=False,
               columnspacing=1.05, handletextpad=0.45)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.115,
                        hspace=0.60, wspace=0.62)
    fig.savefig(f"{OUT}/fig_jjas_era5_acc_lead.pdf")
    plt.close(fig)
    print("wrote fig_jjas_era5_acc_lead.pdf")


def fig_regional_acc_lead(variable="tp", season="jfm", tag="tp"):
    """2x2 grid of the four IMD homogeneous regions: ACC vs lead for every
    system, one season/variable. Shows how the domain-mean skill of
    fig_acc_lead is distributed across India's rainfall regions."""
    df = pd.read_csv(DET[season])
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True, sharey=True)
    letters = string.ascii_lowercase
    regions = list(REGION_PANEL.keys())
    for k, region in enumerate(regions):
        ax = axes[k // 2, k % 2]
        _regional_curve(ax, df, variable, region, "acc")
        ax.set_title(REGION_PANEL[region], pad=14)
        _panel_label(ax, letters[k])
        ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
        ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_xlim(0.6, 6.4)
        _style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("ACC")
    for ax in axes[1, :]:
        ax.set_xlabel("Lead week")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.8, w_pad=2.0)
    out = f"{OUT}/fig_regional_acc_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote fig_regional_acc_{tag}.pdf")


def fig_jjas_regional_reference_acc():
    """JJAS precipitation regional ACC under ERA5 and IMD verification.

    This is the regional counterpart to the main-text JJAS reference-sensitivity
    figure: same 35 common starts, same model set, same regions; only the
    precipitation verification reference and matching climatology change.
    """
    refs = [
        ("ERA5 truth", pd.read_csv(DET["jjas_tp_era5"])),
        ("IMD truth", pd.read_csv(DET["jjas_tp"])),
    ]
    regions = list(REGION_PANEL.keys())
    fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.9), sharex=True, sharey=True)
    letters = string.ascii_lowercase

    for i, (ref_label, df) in enumerate(refs):
        for j, region in enumerate(regions):
            ax = axes[i, j]
            _regional_curve(ax, df, "tp", region, "acc")
            ax.axhspan(0.5, 1.05, color="#0072B2", alpha=0.04, zorder=0)
            ax.axhline(0.5, color="grey", lw=0.6, ls=":", zorder=0)
            ax.axhline(0.0, color="grey", lw=0.7, zorder=0)
            ax.set_ylim(-0.22, 0.9)
            ax.set_xlim(0.6, 6.4)
            ax.text(-0.14, 1.04, f"({letters[i * len(regions) + j]})",
                    transform=ax.transAxes, fontsize=8.8, fontweight="bold",
                    va="bottom", ha="left", clip_on=False)
            if i == 0:
                ax.set_title(REGION_PANEL[region], pad=10)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Lead week")
            if j == 0:
                ax.set_ylabel(f"{ref_label}\nACC")
            _style_axis(ax)

    handles, labels = _legend_union(axes)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.055), frameon=False,
               columnspacing=1.1, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.7, w_pad=1.5)
    fig.savefig(f"{OUT}/fig_jjas_regional_reference_acc.pdf")
    plt.close(fig)
    print("wrote fig_jjas_regional_reference_acc.pdf")


# ----------------------------------------------------------------------
# Spatial maps (India, 1.5-degree common grid, land points only).
# Reads the compact per-cell cache written by make_spatial_cache.py.
# ----------------------------------------------------------------------
def _load_spatial(season):
    path = os.path.join(CACHE, f"spatial_cells_{season}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# India land outline + IMD homogeneous-region boundaries, drawn from the same
# Survey-of-India-derived region masks used for every score in the paper (more
# correct for an India-focused study than a generic global coastline product).
_MASK_PATH = str(IMD_MASK_025)
_MASK_CACHE = {}


def _india_overlay(ax, regions=True):
    """Overlay the India land boundary (and optionally IMD region borders) on a
    cartopy axis, using the SOI-derived region masks. Degrades silently if the
    mask or xarray is unavailable."""
    try:
        import xarray as xr
    except Exception:
        return
    if "ds" not in _MASK_CACHE:
        if not os.path.exists(_MASK_PATH):
            _MASK_CACHE["ds"] = None
        else:
            _MASK_CACHE["ds"] = xr.open_dataset(_MASK_PATH)
    ds = _MASK_CACHE["ds"]
    if ds is None:
        return
    region_vars = list(ds.data_vars)
    lat = ds["lat"].values
    lon = ds["lon"].values
    # integer region id (0 = ocean/non-India, 1..N = regions), and a land mask
    region_id = np.zeros((len(lat), len(lon)))
    for k, rv in enumerate(region_vars, start=1):
        m = np.nan_to_num(ds[rv].values) > 0
        region_id[m] = k
    land = (region_id > 0).astype(float)
    # India land outline
    ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.8,
               transform=ccrs.PlateCarree(), zorder=6)
    if regions:
        # boundaries between adjacent regions
        ax.contour(lon, lat, region_id, levels=np.arange(1.5, len(region_vars)),
                   colors="k", linewidths=0.35, alpha=0.5,
                   transform=ccrs.PlateCarree(), zorder=6)


def _era5_jjas_tp_clim_025():
    """ERA5 JJAS 1991-2020 daily precipitation climatology in mm/day.

    Cached locally because the WeatherBench2 zarr read is the expensive part.
    """
    if os.path.exists(ERA5_JJAS_TP_CLIM_CACHE):
        return xr.open_dataset(ERA5_JJAS_TP_CLIM_CACHE)["tp"]

    ds = xr.open_zarr(WB2_ERA5_025_ZARR)
    tp = ds["total_precipitation_24hr"].sel(
        latitude=slice(40, 5),
        longitude=slice(65, 101),
        time=slice("1991-01-01", "2020-12-31"),
    )
    tp = tp.sel(time=tp["time.month"].isin([6, 7, 8, 9]))
    tp = tp.sel(time=tp["time.hour"] == 6)
    clim = (tp.mean("time") * 1000.0).rename({"latitude": "lat", "longitude": "lon"})
    clim = clim.sortby("lat").sortby("lon").transpose("lat", "lon").astype("float32")
    clim.name = "tp"
    clim.attrs.update({
        "long_name": "ERA5 JJAS mean daily precipitation climatology",
        "baseline": "1991-2020",
        "units": "mm/day",
    })
    clim.to_dataset().to_netcdf(ERA5_JJAS_TP_CLIM_CACHE)
    return clim


def _imd_jjas_tp_clim_025():
    ds = xr.open_dataset(IMD_CLIM_SEASONAL)
    da = ds["season_daily_mean"].sel(season="JJAS")
    da = da.sortby("lat").sortby("lon").transpose("lat", "lon").astype("float32")
    da.name = "tp"
    da.attrs.update({
        "long_name": "IMD JJAS mean daily rainfall climatology",
        "baseline": "1991-2020",
        "units": "mm/day",
    })
    return da


def _imd_land_mask_on(da):
    ds = xr.open_dataset(_MASK_PATH)
    mask = None
    for name in ds.data_vars:
        part = ds[name] > 0
        mask = part if mask is None else (mask | part)
    mask = mask.sortby("lat").sortby("lon").astype(float)
    mask = mask.interp(lat=da["lat"], lon=da["lon"], method="nearest").fillna(0) > 0
    return mask


def fig_imd_era5_climatology_tp():
    """Appendix map: IMD vs ERA5 JJAS precipitation climatology."""
    if not _HAS_CARTOPY:
        print("[skip] fig_imd_era5_climatology_tp: cartopy unavailable")
        return

    imd = _imd_jjas_tp_clim_025()
    era5 = _era5_jjas_tp_clim_025().interp(lat=imd["lat"], lon=imd["lon"])
    mask = _imd_land_mask_on(imd)
    imd = imd.where(mask)
    era5 = era5.where(mask)
    diff = era5 - imd

    abs_vmax = max(10.0, float(np.ceil(np.nanpercentile(np.r_[imd.values.ravel(), era5.values.ravel()], 98))))
    diff_vmax = max(4.0, float(np.ceil(np.nanpercentile(np.abs(diff.values), 98))))

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.05), subplot_kw={"projection": proj})
    fields = [
        (imd, "IMD rainfall", "YlGnBu", 0, abs_vmax, None),
        (era5, "ERA5 precipitation", "YlGnBu", 0, abs_vmax, None),
        (diff, "ERA5 minus IMD", "RdBu", -diff_vmax, diff_vmax, 0.0),
    ]
    ims = []
    for k, (ax, (da, title, cmap, vmin, vmax, center)) in enumerate(zip(axes, fields)):
        if center is None:
            im = ax.pcolormesh(da["lon"], da["lat"], da.transpose("lat", "lon").values,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               transform=proj, shading="auto")
        else:
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
            im = ax.pcolormesh(da["lon"], da["lat"], da.transpose("lat", "lon").values,
                               cmap=cmap, norm=norm, transform=proj, shading="auto")
        ims.append(im)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.35, edgecolor="#777")
        _india_overlay(ax, regions=True)
        ax.set_extent([66, 99.8, 6, 38.5], crs=proj)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=5)
        ax.set_facecolor("#f4f4f4")
        ax.text(0.02, 0.98, f"({string.ascii_lowercase[k]})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))

    cbar0 = fig.colorbar(ims[1], ax=axes[:2].tolist(), orientation="horizontal",
                         fraction=0.055, pad=0.045, aspect=28, extend="max")
    cbar0.set_label("JJAS mean precipitation (mm day$^{-1}$)", fontsize=8.5)
    cbar0.ax.tick_params(labelsize=7.8)
    cbar1 = fig.colorbar(ims[2], ax=[axes[2]], orientation="horizontal",
                         fraction=0.055, pad=0.045, aspect=18, extend="both")
    cbar1.set_label("Difference (mm day$^{-1}$)", fontsize=8.5)
    cbar1.ax.tick_params(labelsize=7.8)
    fig.suptitle("JJAS precipitation climatology reference comparison (1991-2020)",
                 y=0.995, fontsize=10.5, fontweight="bold")
    fig.savefig(f"{OUT}/fig_imd_era5_climatology_tp.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_imd_era5_climatology_tp.pdf")


def _grid_from_cells(sub, value):
    """Turn a (lat, lon, value) long frame into a full 2-D array on the
    regular 1.5-degree grid, NaN where there is no land cell, plus the cell-edge
    coordinates for pcolormesh."""
    piv = sub.pivot_table(index="lat", columns="lon", values=value, aggfunc="mean")
    lats = np.sort(sub["lat"].unique())
    lons = np.sort(sub["lon"].unique())
    piv = piv.reindex(index=lats, columns=lons)
    d = 1.5
    lat_edges = np.concatenate([lats - d / 2, [lats[-1] + d / 2]])
    lon_edges = np.concatenate([lons - d / 2, [lons[-1] + d / 2]])
    return piv.values, lon_edges, lat_edges


def _spatial_panel(ax, sub, value, cmap, vmin, vmax, center=None):
    grid, lon_e, lat_e = _grid_from_cells(sub, value)
    if center is not None:
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), shading="flat")
    else:
        im = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="flat")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888")
    _india_overlay(ax, regions=True)
    ax.set_extent([66, 99, 6, 38], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f2f2f2")
    return im


def _spatial_grid_figure(season, variable, week, value, models, cmap, vmin, vmax,
                         center, cbar_label, title, outname, agg_weeks=None):
    """Grid of per-model spatial maps for one metric. If agg_weeks is given,
    the metric is averaged over those weeks (used for the weeks-1--6 mean bias);
    otherwise a single lead week is shown."""
    if not _HAS_CARTOPY:
        print(f"[skip] {outname}: cartopy unavailable")
        return
    df = _load_spatial(season)
    if df is None:
        print(f"[skip] {outname}: spatial cache for {season} missing")
        return
    df = df[df["variable"] == variable]
    models = [m for m in models if m in df["model"].unique()]
    if not models:
        print(f"[skip] {outname}: no models for {season}/{variable}")
        return

    ncol = 3 if len(models) > 4 else len(models)
    nrow = int(np.ceil(len(models) / ncol))
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.9 * nrow),
                             subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes).ravel()
    im = None
    for k, m in enumerate(models):
        ax = axes[k]
        if agg_weeks is not None:
            sub = df[(df["model"] == m) & (df["week"].isin(agg_weeks))]
            sub = sub.groupby(["lat", "lon"], as_index=False)[value].mean()
        else:
            sub = df[(df["model"] == m) & (df["week"] == week)]
        im = _spatial_panel(ax, sub, value, cmap, vmin, vmax, center)
        ax.set_title(LABEL[m], fontsize=9, fontweight="bold", pad=3)
    for j in range(len(models), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.99)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02,
                        aspect=30)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"{OUT}/{outname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}.pdf")


def fig_spatial_localacc_z500():
    """Per-cell temporal anomaly correlation (local ACC), JFM Z500, week 3 -
    where each system keeps circulation skill into the subseasonal range."""
    _spatial_grid_figure(
        "jfm", "z500", 3, "local_acc",
        ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep"],
        cmap="RdBu_r", vmin=-0.8, vmax=0.8, center=0.0,
        cbar_label="Local anomaly correlation",
        title="JFM 2026 Z500 week-3 local anomaly correlation",
        outname="fig_spatial_localacc_z500")


def fig_spatial_bias_tp():
    """Per-cell mean precipitation bias (weeks 1-6 mean), JFM - spatial wet/dry
    structure behind the domain-mean bias table."""
    _spatial_grid_figure(
        "jfm", "tp", None, "bias",
        ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        cmap="BrBG", vmin=-3.0, vmax=3.0, center=0.0,
        cbar_label="Bias (mm day$^{-1}$)",
        title="JFM 2026 precipitation bias (weeks 1–6 mean, forecast $-$ ERA5)",
        outname="fig_spatial_bias_tp", agg_weeks=[1, 2, 3, 4, 5, 6])


def main():
    fig_acc_lead()
    fig_z500_acc_appendix()
    fig_jjas_era5_tp()
    fig_error_scores()
    fig_jjas_era5_acc_lead()
    fig_jjas_reference_sensitivity()
    fig_jjas17_z500_sensitivity()
    fig_regional_scorecard()
    fig_regional_acc_lead(variable="tp", season="jfm", tag="tp")
    fig_regional_acc_lead(variable="z500", season="jfm", tag="z500")
    fig_jjas_regional_reference_acc()
    fig_imd_era5_climatology_tp()
    fig_spatial_localacc_z500()
    fig_spatial_bias_tp()
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
