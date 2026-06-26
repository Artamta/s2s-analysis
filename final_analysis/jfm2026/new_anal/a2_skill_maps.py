#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a2_skill_maps.py  —  WHERE is each model skilful? (grid-point anomaly correlation)
================================================================================
All-India PCC is one number; it cannot tell you that a model is great over the
peninsula but useless over the northwest. This figure computes, at EVERY land
grid point, the temporal anomaly correlation (ACC) between the model and ERA5
across the 13 JFM2026 inits:

      ACC(x) = corr_over_inits( fcst_anom(x,init) , obs_anom(x,init) )

where anomalies are vs the ERA5 WMO day-of-year climatology (the same baseline
the verified pipeline uses). High ACC = the model captures the init-to-init
swings at that location. One row of maps per model, one column per week band.

  python a2_skill_maps.py --var TP --weeks 1 2 3
  python a2_skill_maps.py --var T2M
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
from core import climatology as CL
import pandas as pd
from core.aggregate import valid_dates_for


def _clim_field(clim_ds, var, valid_dates, GC, phys):
    doys = [pd.to_datetime(d).dayofyear for d in valid_dates]
    return CL.clim_field(clim_ds, var, doys, GC, phys)


def acc_map(cfg, GC, truth, clim_ds, var, week_idx):
    """Per-grid-point ACC across inits, for each model. Returns dict[model]=DataArray."""
    fc_an, ob_an = {}, []
    inits_used = []
    for init in cfg.init_dates:
        wn, ds, de = C.WEEKS[week_idx]
        valid = valid_dates_for(init, ds, de, cfg.valid_end)
        if not valid:
            continue
        o = C.truth_weekly_mean(cfg, truth, GC, init, var, week_idx)
        fields = C.model_weekly_mean(cfg, GC, init, var, week_idx)
        if o is None or not fields:
            continue
        clim = _clim_field(clim_ds, var, valid, GC, cfg.physics)
        ob_an.append(o - clim)
        for m, f in fields.items():
            fc_an.setdefault(m, {})[init] = f - clim
        inits_used.append(init)

    if len(inits_used) < 3:
        return {}, 0
    obs = xr.concat(ob_an, "i")
    out = {}
    for m, dd in fc_an.items():
        # align model anomalies to the inits that have BOTH model and obs
        common_inits = [it for it in inits_used if it in dd]
        if len(common_inits) < 3:
            continue
        idx = [inits_used.index(it) for it in common_inits]
        fc = xr.concat([dd[it] for it in common_inits], "i")
        ob = obs.isel(i=idx)
        fc_c = fc - fc.mean("i")
        ob_c = ob - ob.mean("i")
        num = (fc_c * ob_c).mean("i")
        den = np.sqrt((fc_c**2).mean("i") * (ob_c**2).mean("i"))
        out[m] = (num / den).where(den > 0)
    return out, len(inits_used)


def figure(cfg, GC, truth, clim_ds, var, weeks):
    accs = {}
    nmax = 0
    for wk in weeks:
        a, n = acc_map(cfg, GC, truth, clim_ds, var, wk - 1)
        accs[wk] = a
        nmax = max(nmax, n)
    models = [m for m in C.MODEL_ORDER if any(m in accs[wk] for wk in weeks)]
    if not models:
        print(f"  {var}: no ACC maps"); return

    nrow, ncol = len(models), len(weeks)
    fig = plt.figure(figsize=(3.0 * ncol + 0.6, 3.0 * nrow + 1.0), constrained_layout=False)
    gs = fig.add_gridspec(nrow, ncol, top=0.90, bottom=0.12, left=0.05, right=0.99,
                          hspace=0.10, wspace=0.05)
    im = None
    for i, m in enumerate(models):
        for j, wk in enumerate(weeks):
            ax = C.india_ax(fig, gs[i, j])
            da = accs[wk].get(m)
            if da is not None:
                im = ax.pcolormesh(da.lon, da.lat, da.values, cmap="RdYlGn",
                                   vmin=-1, vmax=1, shading="auto",
                                   transform=ccrs.PlateCarree())
                # area-mean ACC over land-in-region for a quick number
                rda = C.region_outline_da(GC)
                am = float(da.where(rda).weighted(
                    np.cos(np.deg2rad(da.lat))).mean(skipna=True))
                ax.text(0.97, 0.04, f"ⱉ={am:+.2f}".replace("ⱉ", "x̄"),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=10, fontweight="bold",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
            C.add_region_outlines(ax, GC, lw=0.6, color="k")
            if i == 0:
                ax.set_title(f"Week {wk}", fontsize=13)
            if j == 0:
                ax.text(-0.08, 0.5, m, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=14, fontweight="bold",
                        color=C.MODEL_COLOR[m])
    if im is not None:
        cax = fig.add_axes([0.30, 0.055, 0.40, 0.018])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Anomaly correlation across inits (ACC)")
    fig.suptitle(f"{C.VAR_LONG[var]} — where is each model skilful?  "
                 f"(grid-point ACC, JFM2026, {nmax} inits)",
                 fontsize=15, fontweight="bold", y=0.965)
    C.savefig(fig, f"A2_skill_map_{var}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", nargs="+", default=["TP", "T2M", "Z500"])
    ap.add_argument("--weeks", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    ap.add_argument("--dgrid", type=float, default=0.5)
    args = ap.parse_args()
    C.theme()
    cfg = C.get_cfg(args.dgrid)
    GC = C.grid_ctx(cfg)
    truth = C.open_truth(cfg)
    clim_ds = CL.open_clim(cfg.paths.clim_nc)
    print(f"A2 skill maps  (grid={args.dgrid}°, weeks={args.weeks})")
    for var in args.vars:
        figure(cfg, GC, truth, clim_ds, var, args.weeks)


if __name__ == "__main__":
    main()
