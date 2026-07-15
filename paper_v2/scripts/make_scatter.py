#!/usr/bin/env python3
r"""Forecast-versus-reference anomaly density scatter plots for the appendix.

For each system, pooling all Indian land grid cells and all initializations
within a lead week, we bin the forecast anomaly against the reference
anomaly. A perfectly skillful forecast lies on the 1:1 line; the spread
around it and the pooled Pearson correlation $r$ summarize how tightly the
forecast tracks the reference and how skill degrades with lead. This is the
grid-level companion to the domain-mean, per-initialization ACC in the tables,
and reproduces the density diagnostic of the earlier project draft.

Reads the full grid-scatter CSVs and writes:
  paper_v2/figs/fig_scatter_tp.pdf            (JFM 2026 precipitation anomaly)
  paper_v2/figs/fig_scatter_jjas_tp_imd.pdf   (JJAS 2019 precipitation, IMD reference)
  paper_v2/figs/fig_scatter_jjas_tp_era5.pdf  (JJAS 2019 precipitation, ERA5 reference)

Run:  python paper_v2/scripts/make_scatter.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd

from paper_paths import PAPER_OUTPUT_ROOT

GRID = {
    "jfm": str(
        PAPER_OUTPUT_ROOT
        / "jfm2026"
        / "03_metrics"
        / "full_jfm2026_gridscatter"
        / "scatter_grid_weekly.csv"
    ),
    "jjas_tp_imd": str(
        PAPER_OUTPUT_ROOT
        / "jjas2019"
        / "03_metrics"
        / "full_jjas2019_operational35_plus_fuxi_tp_imdtruth"
        / "scatter_grid_weekly.csv"
    ),
    "jjas_tp_era5": str(
        PAPER_OUTPUT_ROOT
        / "jjas2019"
        / "03_metrics"
        / "full_jjas2019_operational35_plus_fuxi_tp_era5truth"
        / "scatter_grid_weekly.csv"
    ),
}
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))

LABEL = {"spire": "Spire AI-S2S", "fuxi": "FuXi-S2S",
         "ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP"}
WEEKS = [1, 3, 6]
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "s2s_density",
    ["#f8fbff", "#d7ecf6", "#9bd4e4", "#49a6c8", "#126aa6", "#08306b", "#3f007d"],
)
DENSITY_CMAP.set_bad((1, 1, 1, 0))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _ticks_for_limit(lim: float) -> list[float]:
    return [-5, 0, 5] if lim <= 10 else [-20, 0, 20]


def _scatter_figure(df, variable, models, unit, lim, outname, title):
    models = [m for m in models if m in df["model"].unique()]
    nrow, ncol = len(WEEKS), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.05 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    panel = {}
    vmax = 1
    for r, wk in enumerate(WEEKS):
        for c, m in enumerate(models):
            sub = df[(df.variable == variable) & (df.model == m) & (df.week == wk)]
            x = sub["truth_anomaly"].to_numpy()
            y = sub["forecast_anomaly"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            hist, xedges, yedges = np.histogram2d(
                x, y, bins=58, range=[[-lim, lim], [-lim, lim]]
            )
            r_p = np.corrcoef(x, y)[0, 1] if x.size > 5 and x.std() and y.std() else np.nan
            panel[(r, c)] = (hist.T, xedges, yedges, r_p)
            if hist.size:
                vmax = max(vmax, int(hist.max()))

    norm = LogNorm(vmin=1, vmax=vmax)
    ticks = _ticks_for_limit(lim)

    for r, wk in enumerate(WEEKS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            hist, xedges, yedges, r_p = panel[(r, c)]
            masked = np.ma.masked_where(hist <= 0, hist)
            ax.pcolormesh(
                xedges, yedges, masked, cmap=DENSITY_CMAP, norm=norm,
                shading="auto", rasterized=True
            )
            if np.isfinite(r_p):
                ax.text(
                    0.055, 0.93, f"$r$={r_p:.2f}", transform=ax.transAxes,
                    fontsize=7.2, va="top", ha="left", color="#111111",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d7dce2", lw=0.45, alpha=0.88),
                )
            ax.plot([-lim, lim], [-lim, lim], color="#0072B2", lw=0.8, ls=(0, (4, 2)))
            ax.axhline(0, color="#7b8794", lw=0.45, alpha=0.7)
            ax.axvline(0, color="#7b8794", lw=0.45, alpha=0.7)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_facecolor("#fbfcfd")
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if r == 0:
                ax.set_title(LABEL[m], fontsize=8.2, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(f"Week {wk}\nForecast", fontsize=7.8)
            if r == nrow - 1:
                ax.set_xlabel("Reference", fontsize=7.8)
    fig.suptitle(title, fontsize=10.2, fontweight="bold", y=0.985)
    fig.text(0.5, 0.02, f"Anomaly ({unit}); dashed line is 1:1; darker cells contain more samples",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.095, top=0.90,
                        wspace=0.08, hspace=0.11)
    fig.savefig(f"{OUT}/{outname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}.pdf")


def main():
    usecols = ["variable", "model", "week", "forecast_anomaly", "truth_anomaly"]

    df = pd.read_csv(GRID["jfm"], usecols=usecols)
    _scatter_figure(
        df, "tp", ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        "mm day$^{-1}$", 8.0, "fig_scatter_tp",
        "JFM 2026 precipitation anomalies vs ERA5 reference")
    jjas_models = ["fuxi", "ecmwf", "ukmo", "ncep"]
    df = pd.read_csv(GRID["jjas_tp_imd"], usecols=usecols)
    _scatter_figure(
        df, "tp", jjas_models,
        "mm day$^{-1}$", 20.0, "fig_scatter_jjas_tp_imd",
        "JJAS 2019 precipitation anomalies vs IMD reference")

    df = pd.read_csv(GRID["jjas_tp_era5"], usecols=usecols)
    _scatter_figure(
        df, "tp", jjas_models,
        "mm day$^{-1}$", 20.0, "fig_scatter_jjas_tp_era5",
        "JJAS 2019 precipitation anomalies vs ERA5 reference")


if __name__ == "__main__":
    main()
