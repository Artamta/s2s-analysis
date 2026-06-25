#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple, self-explanatory model-vs-observation monsoon maps (one init).
   ERA5 observed | ECMWF forecast | difference, for a chosen JJAS init + lead window.
   Run: python model_vs_obs_maps.py --init 2019-07-04 --week 1 4
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)]  # final_analysis/ + jjas/
import pandas as pd
from core import grid as G, truth as T
from core.adapters import get_adapter
import adapters_jjas  # noqa
from config import build_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="2019-07-04")
    ap.add_argument("--week", nargs=2, type=int, default=[1, 4])  # lead-day window
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "model_eval"))
    a = ap.parse_args()
    CFG = build_config(int(a.init[:4])); phys = CFG.physics
    # snap to the nearest ECMWF reforecast init that actually exists
    a.init = min(CFG.init_dates, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(a.init)))
    print("using init", a.init)
    GC = G.build_grid_context(CFG.grid, CFG.paths.region_mask_nc)
    ds, de = a.week
    # valid dates for the window
    valid = pd.date_range(pd.to_datetime(a.init)+pd.Timedelta(days=ds),
                          periods=de-ds+1).strftime("%Y-%m-%d").tolist()
    truth = T.open_truth_wb2(CFG.paths.wb2_zarr, phys,
                             valid[0], (pd.to_datetime(valid[-1])+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    obs = T.truth_period_mean("TP", truth, valid, GC)
    spec = CFG.model("ECMWF")
    cube = get_adapter(spec.adapter)(a.init, "TP", spec, phys)
    fc, _ = cube.weekly(ds, de, GC)
    obs, fc = obs.transpose("lat", "lon"), fc.transpose("lat", "lon")
    diff = fc - obs
    lon, lat = GC["lon"], GC["lat"]
    panels = [(obs, "Observed (ERA5)", "YlGnBu", 0, 25),
              (fc,  "ECMWF forecast",  "YlGnBu", 0, 25),
              (diff,"Forecast − Observed", "RdBu_r", -12, 12)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (d, title, cmap, vmin, vmax) in zip(axes, panels):
        m = ax.pcolormesh(lon, lat, d.values, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="auto", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.6); ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([65, 100, 5, 38]); ax.set_title(title, fontsize=15, fontweight="bold")
        plt.colorbar(m, ax=ax, shrink=0.7, label="mm/day")
    fig.suptitle(f"Monsoon rainfall: ECMWF forecast vs reality  (init {a.init}, lead day {ds}-{de})",
                 fontsize=17, fontweight="bold")
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"model_vs_obs_{a.init}.png")
    fig.savefig(p, bbox_inches="tight", dpi=200); print("wrote", p)

if __name__ == "__main__":
    main()
