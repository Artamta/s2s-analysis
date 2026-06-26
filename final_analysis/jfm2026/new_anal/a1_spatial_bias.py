#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a1_spatial_bias.py  —  WHERE are the models wet/dry, warm/cold? (spatial bias)
================================================================================
The skill CSVs give ONE number per region. This figure opens that number up in
space: for each variable it composites the Week-1 (and optionally Week-3) forecast
over ALL 13 JFM2026 inits and shows

        row 1 : ERA5 truth |  SPIRE  |  FuXi  |  ECMWF      (absolute mean state)
        row 2 :   --        | SPIRE-ERA5 | FuXi-ERA5 | ECMWF-ERA5   (BIAS map)

so you can say e.g. "FuXi runs ~3 K cold over the Indo-Gangetic plain in week 1"
or "ECMWF is systematically wet over the Western Ghats" — statements the regional
averages hide.

Composite = mean over inits of the per-init weekly-mean field (the field every
model is actually scored on). Maps are drawn on the 0.5° grid for clarity but the
analysis is identical at 1.5°.

  python a1_spatial_bias.py                 # TP, T2M, Z500 ; Week 1
  python a1_spatial_bias.py --weeks 1 3     # also a 3-week-ahead panel
  python a1_spatial_bias.py --dgrid 1.5     # on the fair verification grid
================================================================================
"""
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import common as C


# TP uses a perceptually-uniform sequential map; bias maps are diverging.
ABS_CMAP = {"TP": "YlGnBu", "T2M": "RdYlBu_r", "Z500": "viridis"}
BIAS_CMAP = "RdBu_r"
# sensible fixed ranges so panels are comparable across models
ABS_RANGE = {"TP": (0, 12), "T2M": (270, 305), "Z500": (5500, 5900)}
BIAS_RANGE = {"TP": 6.0, "T2M": 5.0, "Z500": 60.0}


def composite(cfg, GC, truth, var, week_idx):
    """Mean over inits of (forecast weekly-mean) and (ERA5 weekly-mean)."""
    fc_stacks, tr_stack = {}, []
    for init in cfg.init_dates:
        o = C.truth_weekly_mean(cfg, truth, GC, init, var, week_idx)
        fields = C.model_weekly_mean(cfg, GC, init, var, week_idx)
        if o is None or not fields:
            continue
        tr_stack.append(o)
        for m, f in fields.items():
            fc_stacks.setdefault(m, []).append(f)
    if not tr_stack:
        return None, {}
    truth_comp = xr.concat(tr_stack, "i").mean("i")
    fc_comp = {m: xr.concat(v, "i").mean("i") for m, v in fc_stacks.items()}
    return truth_comp, fc_comp


def _pcolor(ax, da, cmap, vmin, vmax):
    return ax.pcolormesh(da.lon, da.lat, da.values, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto",
                         transform=ccrs.PlateCarree())


def _label_regions(ax, GC):
    """Annotate each IMD homogeneous region at its centroid (the truth-column key)."""
    short = {"northwest_india": "NW", "central_india": "Central",
             "south_peninsula": "S. Pen.", "east_northeast_india": "E/NE"}
    lon, lat = GC["lon"], GC["lat"]
    LON, LAT = np.meshgrid(lon, lat)
    for key, m in GC["region_masks"].items():
        if not m.any():
            continue
        cx, cy = LON[m].mean(), LAT[m].mean()
        ax.text(cx, cy, short.get(key, key), ha="center", va="center",
                fontsize=10, fontweight="bold", color="0.15",
                transform=ccrs.PlateCarree(),
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.0))


def figure_for_var(cfg, GC, truth, var, week_idx):
    wn = C.WEEKS[week_idx][0]
    truth_comp, fc_comp = composite(cfg, GC, truth, var, week_idx)
    if truth_comp is None:
        print(f"  {var} {wn}: no data"); return
    models = [m for m in C.MODEL_ORDER if m in fc_comp]
    if not models:
        print(f"  {var} {wn}: no model fields"); return

    # Layout: ONE column per model. Top row = absolute mean state (truth shown
    # once in its own dedicated column on the far left); bottom row = bias map.
    # Every cell holds a real map -> no empty / placeholder panels.
    ncol = 1 + len(models)
    amin, amax = ABS_RANGE[var]
    blim = BIAS_RANGE[var]

    fig = plt.figure(figsize=(3.4 * ncol + 1.0, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(2, ncol, width_ratios=[1] * ncol, hspace=0.22, wspace=0.06,
                          left=0.05, right=0.98, top=0.90, bottom=0.06)

    # --- column 0: ERA5 truth (mean state on top, region key below) ----------
    ax = C.india_ax(fig, gs[0, 0])
    im_abs = _pcolor(ax, truth_comp, ABS_CMAP[var], amin, amax)
    C.add_region_outlines(ax, GC, lw=0.7, color="k")
    ax.set_title("ERA5 truth", fontsize=12)
    cb0 = fig.colorbar(im_abs, ax=ax, location="bottom", shrink=0.92, pad=0.02, aspect=22)
    cb0.ax.tick_params(labelsize=9)
    cb0.set_label(f"mean state [{C.VAR_UNIT[var]}]", fontsize=9, labelpad=1)

    axkey = C.india_ax(fig, gs[1, 0])
    C.add_region_outlines(axkey, GC, lw=0.9, color="k")
    _label_regions(axkey, GC)
    axkey.set_title("IMD homogeneous regions", fontsize=10, color="0.3", pad=14)

    # --- columns 1..N: model mean state (top) + bias (bottom) ----------------
    im_bias = None
    for j, m in enumerate(models, 1):
        axt = C.india_ax(fig, gs[0, j])
        _pcolor(axt, fc_comp[m], ABS_CMAP[var], amin, amax)
        C.add_region_outlines(axt, GC, lw=0.6, color="k")
        axt.set_title(m, color=C.MODEL_COLOR[m], fontsize=13)

        axb = C.india_ax(fig, gs[1, j])
        bias = fc_comp[m] - truth_comp
        im_bias = _pcolor(axb, bias, BIAS_CMAP, -blim, blim)
        C.add_region_outlines(axb, GC, lw=0.6, color="k")
        wmean = float(bias.weighted(np.cos(np.deg2rad(bias.lat))).mean(skipna=True))
        axb.set_title(f"bias {wmean:+.1f} {C.VAR_UNIT[var]}",
                      color=C.MODEL_COLOR[m], fontsize=11)

    if im_bias is not None:
        bias_axes = [ax for ax in fig.axes
                     if ax.get_subplotspec() and ax.get_subplotspec().rowspan.start == 1
                     and ax.get_subplotspec().colspan.start >= 1]
        cb2 = fig.colorbar(im_bias, ax=bias_axes, location="bottom",
                           shrink=0.6, pad=0.04, aspect=40)
        cb2.set_label(f"model − ERA5  [{C.VAR_UNIT[var]}]")

    # row labels on the far left
    fig.text(0.012, 0.74, "MEAN STATE", rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold", color="0.35")
    fig.text(0.012, 0.30, "BIAS", rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold", color="0.35")

    fig.suptitle(f"{C.VAR_LONG[var]} — {wn} forecast composite vs ERA5  "
                 f"(JFM2026, mean of {len(cfg.init_dates)} inits)",
                 fontsize=15, fontweight="bold")
    C.savefig(fig, f"A1_spatial_bias_{var}_W{week_idx+1}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", nargs="+", default=["TP", "T2M", "Z500"])
    ap.add_argument("--weeks", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--dgrid", type=float, default=0.5)
    args = ap.parse_args()
    C.theme()
    cfg = C.get_cfg(args.dgrid)
    GC = C.grid_ctx(cfg)
    truth = C.open_truth(cfg)
    print(f"A1 spatial bias  (grid={args.dgrid}°)")
    for var in args.vars:
        for wk in args.weeks:
            figure_for_var(cfg, GC, truth, var, wk - 1)


if __name__ == "__main__":
    main()
