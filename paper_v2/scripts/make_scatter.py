#!/usr/bin/env python3
r"""Forecast-versus-truth anomaly density scatter plots for the appendix.

For each system, pooling all Indian land grid cells and all initializations
within a lead week, we hexbin the forecast anomaly against the observed
(ERA5) anomaly. A perfectly skilful forecast lies on the 1:1 line; the spread
around it and the pooled Pearson correlation $r$ summarise how tightly the
forecast tracks the truth and how skill degrades with lead. This is the
grid-level companion to the domain-mean, per-initialization ACC in the tables,
and reproduces the density-scatter diagnostic of the earlier project draft.

Reads the full grid-scatter CSVs and writes:
  paper_v2/figs/fig_scatter_tp.pdf            (JFM 2026 precipitation anomaly)
  paper_v2/figs/fig_scatter_z500.pdf          (JFM 2026 Z500 anomaly)
  paper_v2/figs/fig_scatter_jjas_tp_imd.pdf   (JJAS 2019 precipitation, IMD truth)
  paper_v2/figs/fig_scatter_jjas_tp_era5.pdf  (JJAS 2019 precipitation, ERA5 truth)
  paper_v2/figs/fig_scatter_jjas_z500.pdf     (JJAS 2019 Z500, ERA5 truth)

Run:  python paper_v2/scripts/make_scatter.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "jjas_z500": str(
        PAPER_OUTPUT_ROOT
        / "jjas2019"
        / "03_metrics"
        / "full_jjas2019_operational35_plus_fuxi_z500"
        / "scatter_grid_weekly.csv"
    ),
}
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))

LABEL = {"spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "delysm": "DLESyM",
         "ecmwf": "ECMWF", "ukmo": "UKMO", "ncep": "NCEP"}
WEEKS = [1, 3, 6]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _scatter_figure(df, variable, models, unit, lim, outname, title):
    models = [m for m in models if m in df["model"].unique()]
    nrow, ncol = len(WEEKS), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.0 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for r, wk in enumerate(WEEKS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            sub = df[(df.variable == variable) & (df.model == m) & (df.week == wk)]
            x = sub["truth_anomaly"].to_numpy()
            y = sub["forecast_anomaly"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size > 5:
                ax.hexbin(x, y, gridsize=34, cmap="magma_r", bins="log",
                          extent=(-lim, lim, -lim, lim), mincnt=1, linewidths=0)
                r_p = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
                ax.text(0.05, 0.92, f"$r$={r_p:.2f}", transform=ax.transAxes,
                        fontsize=7.5, va="top", color="#111")
            ax.plot([-lim, lim], [-lim, lim], color="#0072B2", lw=0.8, ls="--")
            ax.axhline(0, color="grey", lw=0.4)
            ax.axvline(0, color="grey", lw=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(LABEL[m], fontsize=8.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Week {wk}\nforecast", fontsize=8)
            if r == nrow - 1:
                ax.set_xlabel("observed", fontsize=8)
    fig.suptitle(title, fontsize=10.5, fontweight="bold", y=1.005)
    fig.text(0.5, -0.01, f"Anomaly ({unit}); dashed line is 1:1",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{outname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname}.pdf")


def main():
    usecols = ["variable", "model", "week", "forecast_anomaly", "truth_anomaly"]

    df = pd.read_csv(GRID["jfm"], usecols=usecols)
    _scatter_figure(
        df, "tp", ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        "mm day$^{-1}$", 8.0, "fig_scatter_tp",
        "JFM 2026 precipitation-anomaly forecast vs. observed (grid points pooled)")
    _scatter_figure(
        df, "z500", ["spire", "fuxi", "delysm", "ecmwf", "ukmo", "ncep"],
        "m", 120.0, "fig_scatter_z500",
        "JFM 2026 Z500-anomaly forecast vs. observed (grid points pooled)")

    jjas_models = ["fuxi", "ecmwf", "ukmo", "ncep"]
    df = pd.read_csv(GRID["jjas_tp_imd"], usecols=usecols)
    _scatter_figure(
        df, "tp", jjas_models,
        "mm day$^{-1}$", 20.0, "fig_scatter_jjas_tp_imd",
        "JJAS 2019 precipitation-anomaly forecast vs. IMD observed (grid points pooled)")

    df = pd.read_csv(GRID["jjas_tp_era5"], usecols=usecols)
    _scatter_figure(
        df, "tp", jjas_models,
        "mm day$^{-1}$", 20.0, "fig_scatter_jjas_tp_era5",
        "JJAS 2019 precipitation-anomaly forecast vs. ERA5 observed (grid points pooled)")

    df = pd.read_csv(GRID["jjas_z500"], usecols=usecols)
    _scatter_figure(
        df, "z500", jjas_models,
        "m", 120.0, "fig_scatter_jjas_z500",
        "JJAS 2019 Z500-anomaly forecast vs. ERA5 observed (grid points pooled)")


if __name__ == "__main__":
    main()
