#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication-quality model-vs-observation monsoon maps (one init).
   3 panels: ERA5 observed | model forecast | forecast-minus-observed.
   Works for TP (mm/day) and Z500 (gpm), and ECMWF or FuXi.
   Run: python model_vs_obs_maps.py --model ECMWF --var TP --init 2019-07-04 --week 1 4
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)]
from core import grid as G, truth as T
from core.adapters import get_adapter
import adapters_jjas, adapters_fuxi  # noqa  (register ECMWF + FuXi)
from config import build_config

# var -> (unit, abs-cmap, abs vmin/vmax, diff-cmap, diff-limit)
VAR = {"TP":   ("mm/day", "YlGnBu", 0, 25, "RdBu_r", 12),
       "Z500": ("gpm",    "viridis", 5780, 5960, "RdBu_r", 40)}


def _avail_inits(CFG, model):
    if model == "ECMWF":
        return list(CFG.init_dates)
    d = "/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined"
    return [f"{f[:4]}-{f[4:6]}-{f[6:8]}" for f in sorted(os.listdir(d)) if f.endswith(".nc")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ECMWF", choices=["ECMWF", "FuXi"])
    ap.add_argument("--var", default="TP", choices=["TP", "Z500"])
    ap.add_argument("--init", default="2019-07-04")
    ap.add_argument("--week", nargs=2, type=int, default=[1, 4])
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "model_eval"))
    a = ap.parse_args()
    CFG = build_config(int(a.init[:4])); phys = CFG.physics
    avail = _avail_inits(CFG, a.model)
    a.init = min(avail, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(a.init)))
    GC = G.build_grid_context(CFG.grid, CFG.paths.region_mask_nc)
    ds, de = a.week
    valid = pd.date_range(pd.to_datetime(a.init)+pd.Timedelta(days=ds), periods=de-ds+1).strftime("%Y-%m-%d").tolist()
    truth = T.open_truth_wb2(CFG.paths.wb2_zarr, phys, valid[0],
                             (pd.to_datetime(valid[-1])+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    obs = T.truth_period_mean(a.var, truth, valid, GC)
    spec = CFG.model(a.model)
    cube = get_adapter(spec.adapter)(a.init, a.var, spec, phys)
    if cube is None:
        sys.exit(f"no {a.model} {a.var} for init {a.init}")
    fc, _ = cube.weekly(ds, de, GC)
    obs, fc = obs.transpose("lat", "lon"), fc.transpose("lat", "lon")
    unit, cmap, vmin, vmax, dcmap, dlim = VAR[a.var]
    lon, lat = GC["lon"], GC["lat"]
    panels = [(obs, "Observed (ERA5)", cmap, vmin, vmax),
              (fc,  f"{a.model} forecast", cmap, vmin, vmax),
              (fc-obs, "Forecast − Observed", dcmap, -dlim, dlim)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (d, title, cm, lo, hi) in zip(axes, panels):
        m = ax.pcolormesh(lon, lat, d.values, cmap=cm, vmin=lo, vmax=hi, shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.6); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([65, 100, 5, 38]); ax.set_title(title, fontsize=15, fontweight="bold")
        plt.colorbar(m, ax=ax, shrink=0.7, label=unit)
    name = {"TP": "rainfall", "Z500": "500 hPa height (circulation)"}[a.var]
    fig.suptitle(f"Monsoon {name}: {a.model} forecast vs reality  (init {a.init}, lead day {ds}-{de})",
                 fontsize=17, fontweight="bold")
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"mapcmp_{a.model}_{a.var}_{a.init}.png")
    fig.savefig(p, bbox_inches="tight", dpi=200); print("wrote", p)


if __name__ == "__main__":
    main()
