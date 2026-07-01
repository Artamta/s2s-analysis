#!/usr/bin/env python3
r"""Case-study spatial maps: a specific India rainfall event, truth versus each
system, at two forecast leads.

Unlike the season-aggregate diagnostics, a case study needs the per-cell
forecast/truth *fields* for one initialization and lead window, which the
pipeline stores in scatter_grid_weekly.csv. This script extracts one event and
renders it, so a reader can see how the systems captured (or missed) a real
event rather than only aggregate scores.

Featured event (data-selected as the strongest positive domain-mean
precipitation anomaly of the JFM 2026 season): the wet spell verifying
15--21 March 2026 over India. We show it at week-1 lead (initialized 14 March)
and week-2 lead (initialized 7 March) so the lead-time degradation of a single
real event is visible alongside the truth.

Output: paper_v2/figs/fig_case_study_jfm.pdf
Run:    python paper_v2/scripts/make_case_study.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

GRID = ("/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs/"
        "jfm2026/03_metrics/full_jfm2026_gridscatter/scatter_grid_weekly.csv")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))

LABEL = {"spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "ecmwf": "ECMWF",
         "ukmo": "UKMO", "ncep": "NCEP", "truth": "ERA5 (truth)"}
# Precip systems, in the paper's fixed order; truth is drawn first.
MODELS = ["spire", "fuxi", "ecmwf", "ukmo", "ncep"]

EVENT = {
    "valid": "15--21 March 2026",
    "leads": [(20260314, 1, "Week-1 lead (init 14 Mar)"),
              (20260307, 2, "Week-2 lead (init 7 Mar)")],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _grid(sub, value):
    piv = sub.pivot_table(index="lat", columns="lon", values=value, aggfunc="mean")
    lats = np.sort(sub["lat"].unique())
    lons = np.sort(sub["lon"].unique())
    piv = piv.reindex(index=lats, columns=lons)
    d = 1.5
    lat_e = np.concatenate([lats - d / 2, [lats[-1] + d / 2]])
    lon_e = np.concatenate([lons - d / 2, [lons[-1] + d / 2]])
    return piv.values, lon_e, lat_e


def _panel(ax, sub, value, vmax):
    grid, lon_e, lat_e = _grid(sub, value)
    im = ax.pcolormesh(lon_e, lat_e, grid, cmap="YlGnBu", vmin=0, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="flat")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#555")
    ax.set_extent([66, 99, 6, 38], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f2f2f2")
    return im


def main():
    if not _HAS_CARTOPY:
        print("[skip] make_case_study: cartopy unavailable")
        return
    df = pd.read_csv(
        GRID,
        usecols=["init_date", "week", "variable", "model", "lat", "lon",
                 "forecast_value", "truth_value"],
    )
    df = df[df["variable"] == "tp"]

    proj = ccrs.PlateCarree()
    ncol = len(MODELS) + 1  # truth + each model
    nrow = len(EVENT["leads"])
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.7 * nrow),
                             subplot_kw={"projection": proj})
    axes = np.atleast_2d(axes)

    # Shared precip colour scale across all panels for honest comparison.
    ev = df[((df.init_date == 20260314) & (df.week == 1))
            | ((df.init_date == 20260307) & (df.week == 2))]
    vmax = float(np.nanpercentile(
        pd.concat([ev["truth_value"], ev["forecast_value"]]), 98))
    vmax = max(4.0, round(vmax))

    im = None
    for r, (init, wk, lead_label) in enumerate(EVENT["leads"]):
        sel = df[(df.init_date == init) & (df.week == wk)]
        # column 0 = truth (identical across models; take one)
        truth = sel[sel.model == "spire"]
        im = _panel(axes[r, 0], truth, "truth_value", vmax)
        if r == 0:
            axes[r, 0].set_title(LABEL["truth"], fontsize=9, fontweight="bold", pad=3)
        axes[r, 0].text(-0.08, 0.5, lead_label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8.5,
                        fontweight="bold")
        for c, m in enumerate(MODELS, start=1):
            ms = sel[sel.model == m]
            _panel(axes[r, c], ms, "forecast_value", vmax)
            if r == 0:
                axes[r, c].set_title(LABEL[m], fontsize=9, fontweight="bold", pad=3)

    fig.suptitle(f"JFM 2026 case study: precipitation valid {EVENT['valid']}",
                 fontsize=11, fontweight="bold", y=1.005)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                        aspect=32)
    cbar.set_label("Precipitation (mm day$^{-1}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"{OUT}/fig_case_study_jfm.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_case_study_jfm.pdf")


if __name__ == "__main__":
    main()
