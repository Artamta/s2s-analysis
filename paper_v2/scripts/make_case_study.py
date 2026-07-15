#!/usr/bin/env python3
r"""Case-study spatial maps: specific India rainfall events, reference versus
each system, across lead weeks 1--6.

Unlike the season-aggregate diagnostics, a case study needs the per-cell
forecast/reference *fields* for one initialization and lead window, which the
pipeline stores in scatter_grid_weekly.csv. This script extracts one event per
season and renders it, so a reader can see how the systems captured (or missed)
a real event rather than only aggregate scores.

Featured events (each the strongest positive domain-mean precipitation anomaly
of its season that is reachable at lead weeks 1--6):
  * JFM 2026 winter: wet spell verifying 15--21 March 2026 (western-disturbance
    rainfall over northwest India and the Himalayan foothills), ERA5 reference.
  * JJAS 2019 monsoon: active spell verifying 6--12 August 2019, IMD reference.

Outputs:
  paper_v2/figs/fig_case_study_jfm_w1w3.pdf
  paper_v2/figs/fig_case_study_jfm_w4w6.pdf
  paper_v2/figs/fig_case_study_jjas_w1w3.pdf
  paper_v2/figs/fig_case_study_jjas_w4w6.pdf
Run:     python paper_v2/scripts/make_case_study.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_paths import IMD_MASK_025, PAPER_OUTPUT_ROOT

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

ROOT = str(PAPER_OUTPUT_ROOT)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
MASK_PATH = str(IMD_MASK_025)
_MASK = {}

LABEL = {"spire": "Spire AI-S2S", "fuxi": "FuXi-S2S", "ecmwf": "ECMWF",
         "ukmo": "UKMO", "ncep": "NCEP"}
MONTH = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _lead(init: int, week: int) -> tuple[int, int, str]:
    s = str(init)
    label = f"W{week} lead\ninit {int(s[6:8])} {MONTH[int(s[4:6])]}"
    return init, week, label

# One config per season. `truth_model` is any model row (truth is identical
# across models); `leads` lists the initializations that verify the same event
# window at lead weeks 1--6.
CASES = {
    "jfm": {
        "grid": f"{ROOT}/jfm2026/03_metrics/full_jfm2026_gridscatter/scatter_grid_weekly.csv",
        "models": ["spire", "fuxi", "ecmwf", "ukmo", "ncep"],
        "truth_label": "ERA5",
        "valid": "15-21 March 2026",
        "leads": [_lead(20260314, 1), _lead(20260307, 2), _lead(20260228, 3),
                  _lead(20260221, 4), _lead(20260214, 5), _lead(20260207, 6)],
        "title": "JFM 2026 winter precipitation event, valid 15-21 March 2026",
        "plot_title": "JFM 2026 winter event",
        "out": "fig_case_study_jfm",
    },
    "jjas": {
        "grid": f"{ROOT}/jjas2019/03_metrics/full_jjas2019_operational35_plus_fuxi_tp_imdtruth/scatter_grid_weekly.csv",
        "models": ["fuxi", "ecmwf", "ukmo", "ncep"],
        "truth_label": "IMD",
        "valid": "6-12 August 2019",
        "leads": [_lead(20190805, 1), _lead(20190729, 2), _lead(20190722, 3),
                  _lead(20190715, 4), _lead(20190708, 5), _lead(20190701, 6)],
        "title": "JJAS 2019 active-monsoon precipitation event, valid 6-12 August 2019",
        "plot_title": "JJAS 2019 active-monsoon event",
        "out": "fig_case_study_jjas",
    },
}

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
               linewidths=0.7, transform=ccrs.PlateCarree(), zorder=6)
    ax.contour(lon, lat, rid, levels=np.arange(1.5, len(rvars)), colors="k",
               linewidths=0.3, alpha=0.45, transform=ccrs.PlateCarree(), zorder=6)


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
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#8a8a8a")
    _india_overlay(ax)
    ax.set_extent([66, 99, 6, 38], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f4f6f7")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    return im


def _case_vmax(df, cfg):
    ev = pd.concat([
        df[(df.init_date == init) & (df.week == wk)]
        for init, wk, _label in cfg["leads"]
    ])
    vmax = float(np.nanpercentile(
        pd.concat([ev["truth_value"], ev["forecast_value"]]), 98))
    return max(4.0, round(vmax))


def render_case(cfg, leads, suffix, lead_title, vmax):
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
    row_items = [(cfg["truth_label"], None)] + [(LABEL[m], m) for m in models]
    ncol = len(leads)
    nrow = len(row_items)
    is_short_grid = nrow <= 5
    if is_short_grid:
        figsize = (2.05 * ncol, 1.72 * nrow)
    else:
        figsize = (2.25 * ncol, 1.45 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             subplot_kw={"projection": proj})
    axes = np.atleast_2d(axes)

    im = None
    for c, (init, wk, lead_label) in enumerate(leads):
        sel = df[(df.init_date == init) & (df.week == wk)]
        truth = sel[sel.model == models[0]]
        axes[0, c].set_title(lead_label, fontsize=8.8, fontweight="bold", pad=3)
        for r, (row_label, model) in enumerate(row_items):
            if model is None:
                im = _panel(axes[r, c], truth, "truth_value", vmax)
            else:
                _panel(axes[r, c], sel[sel.model == model], "forecast_value", vmax)
            if c == 0:
                axes[r, c].text(-0.09, 0.5, row_label, transform=axes[r, c].transAxes,
                                va="center", ha="right", fontsize=8.6,
                                fontweight="bold")

    fig.suptitle(f"{cfg.get('plot_title', cfg['title'])} ({lead_title})",
                 fontsize=11.2, fontweight="bold", y=0.985)
    if is_short_grid:
        fig.subplots_adjust(left=0.16, right=0.965, top=0.90, bottom=0.12,
                            wspace=0.055, hspace=0.06)
        cax = fig.add_axes([0.28, 0.065, 0.48, 0.018])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Precipitation (mm day$^{-1}$)", fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8, pad=1)
    else:
        fig.subplots_adjust(left=0.155, right=0.875, top=0.91, bottom=0.045,
                            wspace=0.055, hspace=0.075)
        cax = fig.add_axes([0.90, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Precipitation (mm day$^{-1}$)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    out = f"{cfg['out']}_{suffix}"
    fig.savefig(f"{OUT}/{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf")


def main():
    for cfg in CASES.values():
        df = pd.read_csv(
            cfg["grid"],
            usecols=["init_date", "week", "variable", "model", "lat", "lon",
                     "forecast_value", "truth_value"],
        )
        df = df[df["variable"] == "tp"]
        vmax = _case_vmax(df, cfg)
        render_case(cfg, cfg["leads"][:3], "w1w3", "weeks 1-3", vmax)
        render_case(cfg, cfg["leads"][3:], "w4w6", "weeks 4-6", vmax)


if __name__ == "__main__":
    main()
