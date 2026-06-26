#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a4_sst.py  —  Sea-surface-temperature forecast verification (FuXi vs ERA5).
================================================================================
SST is the slowly-varying ocean boundary that gives S2S forecasts much of their
skill (monsoon, IOD, ENSO teleconnections). Of the three systems only FuXi
carries an `sst` channel, so this verifies FuXi SST against ERA5 SST truth over
the north Indian Ocean for the JFM2026 inits.

Truth = ERA5 hourly SST (ARCO-ERA5 public zarr) -> daily mean -> weekly mean,
regridded to a 0.5° ocean grid. Three figures:

  S1  spatial mean SST + bias map (FuXi − ERA5), Week-1 composite of all inits
  S2  basin-mean SST anomaly skill: ACC & RMSE vs lead, for 3 ocean boxes
        Arabian Sea, Bay of Bengal, Equatorial Indian Ocean (IOD east+west)
  S3  per-grid-point SST anomaly correlation map (where is FuXi SST skilful?)

Run (needs internet for ARCO-ERA5; caches the truth to a local .nc):
  python a4_sst.py
  python a4_sst.py --no-cache         # force re-download of ERA5 SST truth
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import common as C
from core.config import WEEKS
from core.adapters import get_adapter
from core.aggregate import valid_dates_for

# ---- Indian-Ocean verification domain + ocean grid --------------------------
BOX = dict(n=30.0, s=-15.0, w=40.0, e=110.0)        # north Indian Ocean
DGRID = 0.5
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CACHE = os.path.join(C.HERE, "era5_sst_jfm2026.nc")

# Ocean basins for the basin-mean skill curves (lat0,lat1,lon0,lon1)
BASINS = {
    "Arabian Sea":            (8, 24, 55, 75),
    "Bay of Bengal":          (8, 22, 82, 98),
    "Equatorial Indian Ocean": (-10, 5, 50, 100),
}


def ocean_grid():
    lat = np.arange(BOX["n"], BOX["s"], -DGRID)
    lon = np.arange(BOX["w"], BOX["e"], DGRID)
    return lat, lon


def to_ocean(da, lat, lon):
    ren = {}
    if "latitude" in da.dims:
        ren["latitude"] = "lat"
    if "longitude" in da.dims:
        ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)
    return da.interp(lat=lat, lon=lon, method="linear").squeeze().transpose(..., "lat", "lon")


# ---- ERA5 SST truth (ARCO) --------------------------------------------------
def _needed_days(cfg):
    """Union of all daily valid dates the forecasts will be scored on
       (every lead day 1..42 of every init, capped at valid_end). Fetching only
       these days from ARCO is MUCH cheaper than the whole continuous window."""
    days = set()
    for init in cfg.init_dates:
        rng = pd.date_range(start=pd.to_datetime(init) + pd.Timedelta(days=1), periods=42)
        for d in rng:
            s = d.strftime("%Y-%m-%d")
            if s <= cfg.valid_end:
                days.add(s)
    return sorted(days)


def load_era5_sst(lat, lon, cfg, no_cache=False):
    """Daily-mean ERA5 SST (time, lat, lon) on the ocean grid, for exactly the
       days the forecasts need. Cached to a local NetCDF after first fetch.

       SST barely changes within a day, so the daily mean is taken from just 4
       synoptic hours (00/06/12/18 UTC) instead of all 24 — a 6x lighter pull
       with no meaningful loss for an SST verification."""
    if os.path.exists(CACHE) and not no_cache:
        print(f"  ERA5 SST: using cache {os.path.basename(CACHE)}")
        return xr.open_dataset(CACHE)["sst"]
    days = _needed_days(cfg)
    print(f"  ERA5 SST: fetching {len(days)} days from ARCO-ERA5 (first run)...")
    ds = xr.open_zarr(ARCO, chunks={"time": 24}, storage_options=dict(token="anon"))
    sst = ds["sea_surface_temperature"].sel(
        latitude=slice(BOX["n"] + 1, BOX["s"] - 1),
        longitude=slice(BOX["w"] - 1, BOX["e"] + 1),
        time=slice(min(days), max(days) + " 23:59"))
    # keep only the needed calendar days, then 4 synoptic hours -> daily mean
    tstamp = pd.to_datetime(sst.time.values)
    keep = np.isin(tstamp.strftime("%Y-%m-%d"), days) & np.isin(tstamp.hour, [0, 6, 12, 18])
    sst = sst.isel(time=np.where(keep)[0])
    daily = sst.resample(time="1D").mean()
    daily = to_ocean(daily, lat, lon).load()
    daily.name = "sst"
    daily.to_netcdf(CACHE)
    print(f"  ERA5 SST: cached -> {os.path.basename(CACHE)}  {tuple(daily.shape)}")
    return daily


def era5_weekly(era5, valid_dates, lat, lon):
    try:
        return era5.sel(time=valid_dates).mean("time")
    except Exception:
        return None


def seasonal_cycle(era5, window=31):
    """Smooth seasonal cycle of ERA5 SST = centred rolling-mean over the window.

    With only ONE season available, a day-of-year climatology has a single sample
    per day, so (SST − DOY-clim) is identically zero and carries no anomaly signal.
    A centred rolling mean instead gives a smooth seasonal baseline; subtracting it
    leaves the REAL sub-seasonal SST anomalies that drive init-to-init skill.
    Returned as a (dayofyear-indexed) field so a forecast valid on day-of-year d
    can be anomalised against the same smooth cycle."""
    sm = era5.rolling(time=window, center=True, min_periods=window // 2).mean()
    doy = pd.to_datetime(sm.time.values).dayofyear
    sm = sm.assign_coords(dayofyear=("time", doy)).swap_dims({"time": "dayofyear"})
    # collapse any duplicate day-of-year (none expected within one season)
    return sm.groupby("dayofyear").mean()


def anom_vs_cycle(field, valid_dates, cycle):
    """field − smooth seasonal cycle, averaged over the valid day-of-years."""
    doys = pd.to_datetime(valid_dates).dayofyear
    base = cycle.sel(dayofyear=[d for d in doys if d in cycle.dayofyear.values]).mean("dayofyear")
    return field - base


# ---- FuXi SST forecast ------------------------------------------------------
def fuxi_sst_weekly(cfg, init, week_idx, lat, lon):
    """FuXi ensemble-mean SST for one (init, week) on the ocean grid."""
    spec = cfg.model("FuXi")
    init_str = init.replace("-", "")
    path = os.path.join(spec.kwargs["root"], "combined", f"{init_str}.nc")
    if not os.path.exists(path):
        return None
    da = xr.open_dataset(path)["forecast"].sel(channel="sst")
    da = da.rename({"lead_time": "step"})
    wn, ds, de = WEEKS[week_idx]
    if da.sizes["step"] < de:
        return None
    wk = da.isel(step=slice(ds - 1, de)).mean("step").mean("member")  # ens+time mean
    return to_ocean(wk, lat, lon)


# ===========================================================================
# S1  spatial SST + bias
# ===========================================================================
def fig_s1(cfg, era5, lat, lon, week_idx):
    wn = WEEKS[week_idx][0]
    fc_stack, tr_stack = [], []
    for init in cfg.init_dates:
        valid = valid_dates_for(init, WEEKS[week_idx][1], WEEKS[week_idx][2], cfg.valid_end)
        if not valid:
            continue
        o = era5_weekly(era5, valid, lat, lon)
        f = fuxi_sst_weekly(cfg, init, week_idx, lat, lon)
        if o is None or f is None:
            continue
        tr_stack.append(o); fc_stack.append(f)
    if not tr_stack:
        print("  S1: no SST data"); return
    truth = xr.concat(tr_stack, "i").mean("i")
    fcst = xr.concat(fc_stack, "i").mean("i")
    bias = fcst - truth

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3)
    ext = (BOX["w"], BOX["e"], BOX["s"], BOX["n"])
    panels = [("ERA5 SST (truth)", truth, "RdYlBu_r", 295, 305, "SST [K]"),
              ("FuXi SST", fcst, "RdYlBu_r", 295, 305, "SST [K]"),
              (f"FuXi − ERA5 (bias)", bias, "RdBu_r", -1.5, 1.5, "Bias [K]")]
    for j, (title, da, cmap, vmin, vmax, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=3)
        im = ax.pcolormesh(da.lon, da.lat, da.values, cmap=cmap, vmin=vmin,
                           vmax=vmax, shading="auto", transform=ccrs.PlateCarree())
        if j == 2:
            wm = float(bias.weighted(np.cos(np.deg2rad(bias.lat))).mean(skipna=True))
            title = f"FuXi − ERA5 (bias, mean {wm:+.2f} K)"
        # draw basin boxes
        for name, (la0, la1, lo0, lo1) in BASINS.items():
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False,
                                       edgecolor="k", lw=1.1, ls="--",
                                       transform=ccrs.PlateCarree(), zorder=4))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label=lab)
    fig.suptitle(f"Sea-surface temperature — {wn} composite, FuXi vs ERA5  "
                 f"(JFM2026, {len(tr_stack)} inits)", fontsize=15, fontweight="bold")
    C.savefig(fig, f"A4_S1_sst_bias_W{week_idx+1}.png")


# ===========================================================================
# S2  basin-mean SST anomaly skill vs lead
# ===========================================================================
def _box_mean(da, box):
    la0, la1, lo0, lo1 = box
    sub = da.sel(lat=slice(max(la0, la1), min(la0, la1)), lon=slice(lo0, lo1))
    return float(sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(skipna=True))


def fig_s2(cfg, era5, lat, lon):
    cycle = seasonal_cycle(era5)

    # collect per (week, basin): forecast-anom and obs-anom across inits
    rec = {b: {wk: {"f": [], "o": []} for wk in range(6)} for b in BASINS}
    for init in cfg.init_dates:
        for wk in range(6):
            valid = valid_dates_for(init, WEEKS[wk][1], WEEKS[wk][2], cfg.valid_end)
            if not valid:
                continue
            o = era5_weekly(era5, valid, lat, lon)
            f = fuxi_sst_weekly(cfg, init, wk, lat, lon)
            if o is None or f is None:
                continue
            oa = anom_vs_cycle(o, valid, cycle)
            fa = anom_vs_cycle(f, valid, cycle)
            for b, box in BASINS.items():
                rec[b][wk]["o"].append(_box_mean(oa, box))
                rec[b][wk]["f"].append(_box_mean(fa, box))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(BASINS)))
    for (b, box), col in zip(BASINS.items(), colors):
        accs, rmses = [], []
        for wk in range(6):
            f = np.array(rec[b][wk]["f"]); o = np.array(rec[b][wk]["o"])
            ok = np.isfinite(f) & np.isfinite(o)
            f, o = f[ok], o[ok]
            if len(f) >= 3 and f.std() > 0 and o.std() > 0:
                accs.append(np.corrcoef(f, o)[0, 1])
                rmses.append(np.sqrt(np.mean((f - o) ** 2)))
            else:
                accs.append(np.nan); rmses.append(np.nan)
        wks = np.arange(1, 7)
        axes[0].plot(wks, accs, marker="o", color=col, lw=2.4, label=b)
        axes[1].plot(wks, rmses, marker="s", color=col, lw=2.4, label=b)
    axes[0].axhline(0.5, color="0.4", ls=":", lw=1.2)
    axes[0].set_ylim(-0.5, 1.02); axes[0].set_ylabel("Anomaly correlation (ACC)")
    axes[0].set_title("FuXi SST anomaly skill")
    axes[1].set_ylabel("RMSE of SST anomaly [K]")
    axes[1].set_title("FuXi SST anomaly error")
    for ax in axes:
        ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"W{i}" for i in range(1, 7)])
        ax.set_xlabel("Forecast lead"); ax.grid(alpha=0.25, ls=":")
    axes[0].legend(loc="lower left", fontsize=11)
    fig.suptitle("FuXi sea-surface-temperature skill by ocean basin  (JFM2026)",
                 fontsize=15, fontweight="bold")
    C.savefig(fig, "A4_S2_sst_basin_skill.png")


# ===========================================================================
# S3  grid-point SST anomaly correlation map
# ===========================================================================
def fig_s3(cfg, era5, lat, lon, weeks=(1, 2, 4, 6)):
    cycle = seasonal_cycle(era5)

    maps = {}
    for wk in weeks:
        fa_stack, oa_stack = [], []
        for init in cfg.init_dates:
            valid = valid_dates_for(init, WEEKS[wk - 1][1], WEEKS[wk - 1][2], cfg.valid_end)
            if not valid:
                continue
            o = era5_weekly(era5, valid, lat, lon)
            f = fuxi_sst_weekly(cfg, init, wk - 1, lat, lon)
            if o is None or f is None:
                continue
            oa_stack.append(anom_vs_cycle(o, valid, cycle))
            fa_stack.append(anom_vs_cycle(f, valid, cycle))
        if len(fa_stack) < 3:
            continue
        fc = xr.concat(fa_stack, "i"); ob = xr.concat(oa_stack, "i")
        fc_c = fc - fc.mean("i"); ob_c = ob - ob.mean("i")
        num = (fc_c * ob_c).mean("i")
        den = np.sqrt((fc_c ** 2).mean("i") * (ob_c ** 2).mean("i"))
        maps[wk] = (num / den).where(den > 0)
    if not maps:
        print("  S3: no SST ACC maps"); return

    ext = (BOX["w"], BOX["e"], BOX["s"], BOX["n"])
    n = len(maps)
    ncol = n if n <= 3 else 2
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.4 * ncol, 3.8 * nrow + 1.0), constrained_layout=False)
    gs = fig.add_gridspec(nrow, ncol, top=0.90, bottom=0.10, left=0.04, right=0.98,
                          hspace=0.18, wspace=0.06)
    im = None
    for j, wk in enumerate(sorted(maps)):
        ax = fig.add_subplot(gs[j // ncol, j % ncol], projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=3)
        da = maps[wk]
        im = ax.pcolormesh(da.lon, da.lat, da.values, cmap="RdYlGn", vmin=-1, vmax=1,
                           shading="auto", transform=ccrs.PlateCarree())
        for name, (la0, la1, lo0, lo1) in BASINS.items():
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False,
                                       edgecolor="k", lw=1.0, ls="--",
                                       transform=ccrs.PlateCarree(), zorder=4))
        ax.set_title(f"Week {wk}")
    cax = fig.add_axes([0.30, 0.045, 0.40, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("SST anomaly correlation across inits (ACC)")
    fig.suptitle("Where is FuXi SST skilful?  (grid-point anomaly correlation, JFM2026)",
                 fontsize=15, fontweight="bold", y=0.965)
    C.savefig(fig, "A4_S3_sst_skill_map.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    C.theme()
    cfg = C.get_cfg(1.5)               # only used for init_dates/valid_end/paths
    lat, lon = ocean_grid()
    era5 = load_era5_sst(lat, lon, cfg, no_cache=args.no_cache)
    print("A4 SST verification (FuXi vs ERA5, north Indian Ocean)")
    fig_s1(cfg, era5, lat, lon, week_idx=0)    # Week-1 bias composite
    fig_s1(cfg, era5, lat, lon, week_idx=3)    # Week-4 bias composite (bias growth)
    fig_s2(cfg, era5, lat, lon)
    fig_s3(cfg, era5, lat, lon)


if __name__ == "__main__":
    main()
