#!/usr/bin/env python3
r"""Case-study spatial maps: specific India rainfall events, truth versus each
system, at two forecast leads.

Unlike the season-aggregate diagnostics, a case study needs the per-cell
forecast/truth *fields* for one initialization and lead window, which the
pipeline stores in scatter_grid_weekly.csv. This script extracts one event per
season and renders it, so a reader can see how the systems captured (or missed)
a real event rather than only aggregate scores.

Featured events (each the strongest positive domain-mean precipitation anomaly
of its season that is reachable at both week-1 and week-2 lead):
  * JFM 2026 winter: wet spell verifying 15--21 March 2026 (western-disturbance
    rainfall over northwest India and the Himalayan foothills), ERA5 truth.
  * JJAS 2019 monsoon: active spell verifying 6--12 August 2019, IMD truth.

Outputs: paper_v2/figs/fig_case_study_jfm.pdf, fig_case_study_jjas.pdf
Run:     python paper_v2/scripts/make_case_study.py
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

ROOT = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/outputs/s2s_paper_outputs"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
MASK_PATH = "/home/raj.ayush/s2s/s2s_anlysis/final_paper/masks/imd_region_masks_0.25deg.nc"
_MASK = {}

LABEL = {"spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "ecmwf": "ECMWF",
         "ukmo": "UKMO", "ncep": "NCEP"}

# One config per season. `truth_model` is any model row (truth is identical
# across models); `leads` pairs the two initializations that verify the same
# window at week-1 and week-2 lead.
CASES = {
    "jfm": {
        "grid": f"{ROOT}/jfm2026/03_metrics/full_jfm2026_gridscatter/scatter_grid_weekly.csv",
        "models": ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        "truth_label": "ERA5 (truth)",
        "valid": "15-21 March 2026",
        "leads": [(20260314, 1, "Week-1 lead (init 14 Mar)"),
                  (20260307, 2, "Week-2 lead (init 7 Mar)")],
        "title": "JFM 2026 winter case study: precipitation valid 15-21 March 2026",
        "out": "fig_case_study_jfm",
    },
    "jjas": {
        "grid": f"{ROOT}/jjas2019/03_metrics/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/scatter_grid_weekly.csv",
        "models": ["fuxi", "ecmwf", "ukmo", "ncep"],
        "truth_label": "IMD (truth)",
        "valid": "6-12 August 2019",
        "leads": [(20190805, 1, "Week-1 lead (init 5 Aug)"),
                  (20190729, 2, "Week-2 lead (init 29 Jul)")],
        "title": "JJAS 2019 monsoon case study: precipitation valid 6-12 August 2019",
        "out": "fig_case_study_jjas",
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _india_overlay(ax):
    try:
        import xarray as xr
    except Exception:
        return
    if "ds" not in _MASK:
        _MASK["ds"] = xr.open_dataset(MASK_PATH) if os.path.exists(MASK_PATH) else None
    ds = _MASK["ds"]
    if ds is None:
        return
    rvars = list(ds.data_vars)
    lat, lon = ds["lat"].values, ds["lon"].values
    rid = np.zeros((len(lat), len(lon)))
    for k, rv in enumerate(rvars, start=1):
        rid[np.nan_to_num(ds[rv].values) > 0] = k
    ax.contour(lon, lat, (rid > 0).astype(float), levels=[0.5], colors="k",
               linewidths=0.8, transform=ccrs.PlateCarree(), zorder=6)
    ax.contour(lon, lat, rid, levels=np.arange(1.5, len(rvars)), colors="k",
               linewidths=0.35, alpha=0.5, transform=ccrs.PlateCarree(), zorder=6)


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
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888")
    _india_overlay(ax)
    ax.set_extent([66, 99, 6, 38], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f2f2f2")
    return im


def render_case(cfg):
    if not _HAS_CARTOPY:
        print(f"[skip] {cfg['out']}: cartopy unavailable")
        return
    df = pd.read_csv(
        cfg["grid"],
        usecols=["init_date", "week", "variable", "model", "lat", "lon",
                 "forecast_value", "truth_value"],
    )
    df = df[df["variable"] == "tp"]
    models = [m for m in cfg["models"] if m in df["model"].unique()]

    proj = ccrs.PlateCarree()
    ncol = len(models) + 1
    nrow = len(cfg["leads"])
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.7 * nrow),
                             subplot_kw={"projection": proj})
    axes = np.atleast_2d(axes)

    ev = df[((df.init_date == cfg["leads"][0][0]) & (df.week == cfg["leads"][0][1]))
            | ((df.init_date == cfg["leads"][1][0]) & (df.week == cfg["leads"][1][1]))]
    vmax = float(np.nanpercentile(
        pd.concat([ev["truth_value"], ev["forecast_value"]]), 98))
    vmax = max(4.0, round(vmax))

    im = None
    for r, (init, wk, lead_label) in enumerate(cfg["leads"]):
        sel = df[(df.init_date == init) & (df.week == wk)]
        truth = sel[sel.model == models[0]]
        im = _panel(axes[r, 0], truth, "truth_value", vmax)
        if r == 0:
            axes[r, 0].set_title(cfg["truth_label"], fontsize=9, fontweight="bold", pad=3)
        axes[r, 0].text(-0.08, 0.5, lead_label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8.5,
                        fontweight="bold")
        for c, m in enumerate(models, start=1):
            _panel(axes[r, c], sel[sel.model == m], "forecast_value", vmax)
            if r == 0:
                axes[r, c].set_title(LABEL[m], fontsize=9, fontweight="bold", pad=3)

    fig.suptitle(cfg["title"], fontsize=11, fontweight="bold", y=1.005)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                        aspect=32)
    cbar.set_label("Precipitation (mm day$^{-1}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(f"{OUT}/{cfg['out']}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cfg['out']}.pdf")


def main():
    for cfg in CASES.values():
        render_case(cfg)


if __name__ == "__main__":
    main()
