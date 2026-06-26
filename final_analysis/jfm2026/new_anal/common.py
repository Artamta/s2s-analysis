#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_anal/common.py  —  Shared helpers for the JFM2026 "extra" analysis.
================================================================================
The jfm2026 verification pipeline already produces scalar SKILL CSVs (PCC, CRPSS,
Brier ...). This sibling package adds the things those scalars cannot show:

  * SPATIAL maps  — where each model is wet/dry, warm/cold, and where it is
                    actually skilful over India (not just an All-India average).
  * IMD-region    — clean per-homogeneous-region skill profiles.
  * SST           — FuXi sea-surface-temperature forecast vs ERA5 over the
                    north Indian Ocean (Arabian Sea / Bay of Bengal / equator).

Everything here reuses the EXISTING adapters + grid + truth machinery so the
numbers are identical to the verified pipeline — we only re-render them as maps.

Outputs go to  new_anal/figs/  (figures are for viewing; kept in the repo tree).
================================================================================
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
JFM = os.path.dirname(HERE)                       # jfm2026/
sys.path.append(JFM)
sys.path.append(os.path.dirname(JFM))             # final_analysis/

import adapters_jfm          # noqa: F401  registers SPIRE/FuXi/ECMWF adapters
from config import build_config
from core import grid as G
from core import truth as T
from core.adapters import get_adapter
from core.config import WEEKS

FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---- identity / cosmetics ---------------------------------------------------
MODEL_COLOR = {"SPIRE": "#0072B2", "FuXi": "#D55E00", "ECMWF": "#009E73",
               "MME": "#000000", "ERA5": "#444444"}
MODEL_ORDER = ["SPIRE", "FuXi", "ECMWF"]
REGION_NICE = {
    "All India": "All-India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & NE India",
}
REGION_ORDER = ["northwest_india", "central_india",
                "south_peninsula", "east_northeast_india"]
VAR_LONG = {"TP": "Rainfall", "T2M": "2 m temperature", "Z500": "500 hPa height",
            "SST": "Sea-surface temperature"}
VAR_UNIT = {"TP": "mm/day", "T2M": "K", "Z500": "gpm", "SST": "K"}


def theme():
    import matplotlib
    matplotlib.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 12,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
        "figure.constrained_layout.use": True,
    })


# ---- experiment config at a chosen map resolution ---------------------------
def get_cfg(dgrid):
    """Build the JFM2026 config at the requested map grid (0.5 = pretty maps,
       1.5 = the fair verification grid used by the skill CSVs)."""
    return build_config(dgrid)


def grid_ctx(cfg):
    return G.build_grid_context(cfg.grid, cfg.paths.region_mask_nc)


def open_truth(cfg):
    return T.open_truth(cfg.paths, cfg.physics)


# ---- model field extraction (reuses the verified adapters) ------------------
def model_weekly_mean(cfg, GC, init, var, week_idx):
    """Forecast ensemble-MEAN field for one (init, var, week) -> dict[model]=DataArray.
       week_idx is 0-based (0 = Week-1)."""
    wn, ds, de = WEEKS[week_idx]
    out = {}
    for spec in cfg.models:
        adapt = get_adapter(spec.adapter)
        cube = adapt(init, var, spec, cfg.physics)
        if cube is None or not cube.has_week(de):
            continue
        mu, _ = cube.weekly(ds, de, GC)
        out[spec.name] = mu
    return out


def truth_weekly_mean(cfg, truth, GC, init, var, week_idx):
    """ERA5 truth averaged over the valid days of one forecast week."""
    from core.aggregate import valid_dates_for
    wn, ds, de = WEEKS[week_idx]
    valid = valid_dates_for(init, ds, de, cfg.valid_end)
    if not valid:
        return None
    return T.truth_period_mean(var, truth, valid, GC)


def region_outline_da(GC):
    """All-India boolean mask as a DataArray for contour outlines on maps."""
    return G.region_da("All India", GC)


# ---- pretty India map axis --------------------------------------------------
def india_ax(fig, gs_or_rect, extent=(65, 100, 5, 38)):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_subplot(gs_or_rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, color="0.25")
    ax.add_feature(cfeature.BORDERS, lw=0.5, color="0.45")
    ax.add_feature(cfeature.LAND, facecolor="none")
    gl = ax.gridlines(draw_labels=False, lw=0.3, color="0.8", alpha=0.6)
    return ax


def add_region_outlines(ax, GC, lw=0.8, color="k"):
    """Draw the 4 IMD homogeneous-region boundaries as contour outlines."""
    import cartopy.crs as ccrs
    lon, lat = GC["lon"], GC["lat"]
    LON, LAT = np.meshgrid(lon, lat)
    for key, m in GC["region_masks"].items():
        ax.contour(LON, LAT, m.astype(float), levels=[0.5],
                   colors=color, linewidths=lw, transform=ccrs.PlateCarree())


def savefig(fig, name):
    import matplotlib.pyplot as plt
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, JFM)}")
    return p
