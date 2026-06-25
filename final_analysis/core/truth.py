#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/truth.py  —  ERA5 ground truth (the observation everything is scored against).
================================================================================
Three verification variables, three native sources:
  TP   : ERA5 daily total precipitation  [mm/day]   (NetCDF, true 24-h totals)
  T2M  : ERA5 daily-MEAN 2 m temperature [K]         (NetCDF)
  Z500 : ERA5 500 hPa geopotential       [m^2/s^2]   (GRIB) -> /G -> gpm

The truth series is now CONTINUOUS (starts before the first init), so the
persistence baseline reads straight from it — no special pre-init patch needed.

`open_truth()` returns a dict of raw DataArrays (opened once per worker). The
accessors regrid on demand via core.grid.to_grid.
================================================================================
"""
import numpy as np
import pandas as pd
import xarray as xr

from .grid import to_grid, crop_box

# cfgrib open options (no .idx sidecar files written)
_GRIB = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})


def open_truth(paths, physics):
    """Open ERA5 truth sources once. Returns {tp_daily, t2m_daily, z_raw}."""
    t = {}
    t["tp_daily"] = xr.open_dataset(paths.era5_daily_tp)["tp"]            # mm/day
    t["t2m_daily"] = xr.open_dataset(paths.era5_daily_t2m)["t2m"]         # K daily mean
    t["z_raw"] = crop_box(xr.open_dataset(paths.era5_z500_grib, **_GRIB)["z"] / physics.G)  # gpm
    return t


def open_truth_wb2(zarr_path, physics, start, end, box=(40.0, 3.0, 60.0, 102.0)):
    """ERA5 truth from a WeatherBench2 zarr, daily, for a date window [start, end].

    Returns the SAME {tp_daily, z_raw} dict interface as `open_truth`, so every
    truth accessor (period_mean / day / persistence) works unchanged.

    Conversions:
      TP   : total_precipitation_6hr [m] -> daily SUM -> x1000 -> mm/day
      Z500 : geopotential@500 [m^2/s^2] -> daily MEAN -> /g -> gpm
    The window is pre-cropped to an India box and loaded into memory so the
    per-init .sel calls in the driver are cheap.
    """
    ds = xr.open_zarr(zarr_path)
    n, s, w, e = box
    # WB2 latitude is ascending (-90..90), longitude 0..358.5
    ds = ds.sel(latitude=slice(s, n), longitude=slice(w, e),
                time=slice(start, end))
    tp6 = ds["total_precipitation_6hr"]
    tp_daily = (tp6.resample(time="1D").sum() * 1000.0).load()           # mm/day
    z = ds["geopotential"].sel(level=500)
    z_daily = (z.resample(time="1D").mean() / physics.G).load()          # gpm
    t = {"tp_daily": tp_daily, "z_raw": z_daily, "t2m_daily": None}
    return t


def _src(var, truth):
    if var == "TP":
        return truth["tp_daily"]
    if var == "T2M":
        return truth["t2m_daily"]
    return truth["z_raw"]


def truth_series_on_grid(var, truth, start, end, GC):
    """Full daily truth series on the verification grid for [start, end]
       (used to compute the climatological spread)."""
    src = _src(var, truth).sel(time=slice(start, end))
    return to_grid(src, GC)


def truth_period_mean(var, truth, valid_dates, GC):
    """ERA5 truth averaged over the EXPLICIT list of valid dates (weekly mean).
       Uses .sel(time=list) (not a contiguous slice) so the obs window matches
       the day-of-year set used for the climatology exactly, even if the date
       list is non-contiguous (e.g. capped at valid_end)."""
    try:
        src = _src(var, truth).sel(time=valid_dates).mean("time")
        return to_grid(src, GC)
    except Exception:
        return None


def truth_day(var, truth, date, GC):
    """ERA5 truth for a single calendar day (daily-lead verification)."""
    try:
        o = to_grid(_src(var, truth).sel(time=slice(date, date)).mean("time"), GC)
        return o if not bool(np.isnan(o).all()) else None
    except Exception:
        return None


def persistence_field(var, truth, init, GC):
    """Observed mean over the 7 days immediately BEFORE init (persistence baseline).
       Reads straight from the continuous truth series."""
    pre = pd.date_range(end=pd.to_datetime(init) - pd.Timedelta(days=1), periods=7)
    valid = [d.strftime("%Y-%m-%d") for d in pre]
    pieces = []
    for d in valid:
        try:
            pieces.append(to_grid(_src(var, truth).sel(time=d), GC))
        except Exception:
            pass
    return xr.concat(pieces, "t").mean("t") if pieces else None
