#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication-quality model-vs-observation monsoon maps (one init).

   3 panels: ERA5 observed | model forecast | forecast-minus-observed.
   Works for TP (mm/day) and Z500 (gpm), and ECMWF or FuXi.

   Rendering is publication-grade (not the raw blocky pcolormesh):
     * smooth filled contours (contourf) so the coarse 1.5deg grid reads cleanly
     * the two absolute panels SHARE one colorbar; the difference panel has its own
     * India state outline (Survey-of-India shapefile, reprojected LCC -> WGS84)
     * light lat/lon gridlines with labels, ocean greyed, crisp coast/borders

   Run: python model_vs_obs_maps.py --model ECMWF --var TP --init 2019-07-04 --week 1 4
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)]
from core import grid as G, truth as T
from core.adapters import get_adapter
import adapters_jjas, adapters_fuxi  # noqa  (register ECMWF + FuXi)
from config import build_config

SHAPEFILE = "/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp"
EXTENT = [66, 99, 5, 38]

# var -> (long-name, unit, abs-cmap, abs vmin/vmax, n-levels, diff-cmap, diff-step, n-diff-steps)
VAR = {
    "TP":   ("rainfall",                 "mm day$^{-1}$", "YlGnBu",  0,   24, 13, "RdBu_r",  2, 6),
    "Z500": ("500 hPa geopotential height", "gpm",        "viridis", 5780, 5960, 19, "RdBu_r", 6, 6),
}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "figure.dpi": 120,
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
})


def _fix_proj_db():
    """Point PROJ at the env's proj.db (pyproj otherwise fails to reproject)."""
    import pyproj
    cand = [os.path.join(sys.prefix, "share", "proj"),
            os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")]
    for d in cand:
        if os.path.exists(os.path.join(d, "proj.db")):
            os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = d
            try:
                pyproj.datadir.set_data_dir(d)
            except Exception:
                pass
            return


def _india_outline():
    """India state boundaries as a cartopy feature in PlateCarree (lon/lat)."""
    try:
        _fix_proj_db()
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326")
        from cartopy.feature import ShapelyFeature
        return ShapelyFeature(gdf.geometry, ccrs.PlateCarree(),
                              edgecolor="0.25", facecolor="none", lw=0.4)
    except Exception as e:                       # shapefile optional — degrade gracefully
        print("  (india outline skipped:", e, ")")
        return None


def _avail_inits(CFG, model):
    if model == "ECMWF":
        return list(CFG.init_dates)
    d = "/storage/raj.ayush/s2s_final_data/jjas/fuxi_combined"
    return [f"{f[:4]}-{f[4:6]}-{f[6:8]}" for f in sorted(os.listdir(d)) if f.endswith(".nc")]


def _draw(ax, lon, lat, data, cmap, levels, extend, india):
    """One smooth filled-contour panel with map furniture; returns the mappable."""
    ax.set_facecolor("0.92")                                   # ocean / NaN backdrop
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap,
                     extend=extend, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="0.92", zorder=2)
    ax.add_feature(cfeature.COASTLINE, lw=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS, lw=0.5, edgecolor="0.3", zorder=3)
    if india is not None:
        ax.add_feature(india, zorder=4)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="0.6", alpha=0.5, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlocator = ticker.FixedLocator([70, 80, 90])
    gl.ylocator = ticker.FixedLocator([10, 20, 30])
    return cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ECMWF", choices=["ECMWF", "FuXi"])
    ap.add_argument("--var", default="TP", choices=["TP", "Z500"])
    ap.add_argument("--init", default="2019-07-04")
    ap.add_argument("--week", nargs=2, type=int, default=[1, 7])
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "model_eval"))
    a = ap.parse_args()
    CFG = build_config(int(a.init[:4])); phys = CFG.physics
    avail = _avail_inits(CFG, a.model)
    a.init = min(avail, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(a.init)))
    GC = G.build_grid_context(CFG.grid, CFG.paths.region_mask_nc)
    ds, de = a.week
    valid = pd.date_range(pd.to_datetime(a.init) + pd.Timedelta(days=ds),
                          periods=de - ds + 1).strftime("%Y-%m-%d").tolist()
    truth = T.open_truth_wb2(CFG.paths.wb2_zarr, phys, valid[0],
                             (pd.to_datetime(valid[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    obs = T.truth_period_mean(a.var, truth, valid, GC)
    spec = CFG.model(a.model)
    cube = get_adapter(spec.adapter)(a.init, a.var, spec, phys)
    if cube is None:
        sys.exit(f"no {a.model} {a.var} for init {a.init}")
    fc, _ = cube.weekly(ds, de, GC)
    obs, fc = obs.transpose("lat", "lon"), fc.transpose("lat", "lon")

    longname, unit, cmap, vmin, vmax, nlev, dcmap, dstep, dn = VAR[a.var]
    lon, lat = GC["lon"], GC["lat"]
    dlim = dstep * dn
    alevels = np.linspace(vmin, vmax, nlev)
    dlevels = np.arange(-dlim, dlim + dstep / 2, dstep)    # round, readable diff ticks
    india = _india_outline()

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.12)
    axes = [fig.add_subplot(gs[i], projection=ccrs.PlateCarree()) for i in range(3)]

    cf0 = _draw(axes[0], lon, lat, obs.values, cmap, alevels, "max" if a.var == "TP" else "both", india)
    _draw(axes[1], lon, lat, fc.values, cmap, alevels, "max" if a.var == "TP" else "both", india)
    cfd = _draw(axes[2], lon, lat, (fc - obs).values, dcmap, dlevels, "both", india)

    axes[0].set_title("Observed  (ERA5)")
    axes[1].set_title(f"{a.model} forecast")
    axes[2].set_title("Forecast − Observed (error)")

    # shared colorbar for the two absolute panels
    cb = fig.colorbar(cf0, ax=axes[:2], orientation="horizontal",
                      fraction=0.05, pad=0.10, aspect=40, shrink=0.85)
    cb.set_label(f"{longname} ({unit})", fontsize=12)
    cbd = fig.colorbar(cfd, ax=axes[2], orientation="horizontal",
                       fraction=0.05, pad=0.10, aspect=22, shrink=0.95)
    cbd.set_label(f"error ({unit})", fontsize=11)

    fig.suptitle(f"Monsoon {longname}: {a.model} forecast vs reality   "
                 f"(init {a.init}, lead days {ds}–{de})",
                 fontsize=16, fontweight="bold", y=1.00)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"mapcmp_{a.model}_{a.var}_{a.init}.png")
    fig.savefig(p, bbox_inches="tight", dpi=200, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
