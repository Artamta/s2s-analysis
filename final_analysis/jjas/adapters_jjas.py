#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/adapters_jjas.py  —  Multi-year reforecast loaders for the JJAS study.
================================================================================
ECMWF reforecast files stack 20 hindcast years in one GRIB per calendar date:
    <stem>_<cf|pf>_<MMDD>.grib   dims (number, time=20yr, step=46, lat, lon)
    TP   stem='tp'   var 'tp'  kg m^-2 CUMULATIVE   -> accum='cumulative'
    Z500 stem='z500' var 'gh'  gpm (instant)        -> accum='instant'
cf = control (no 'number'); pf = 10 perturbed members. We merge them to 11.

Two adapters:
  ecmwf_reforecast       forecast for ONE hindcast year (init's year) -> members cube
  ecmwf_reforecast_clim  model-own climatology = mean over ALL 20 years AND members
                         -> a mean-only summary cube (the lead-dependent hindcast clim)

Both share `_open_all`, which builds the (member, time, step, lat, lon) array once.
================================================================================
"""
import os
import sys
from functools import lru_cache

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

from core import register, ForecastCube
from core.grid import crop_box

_OPEN = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
_STEM = {"TP": "tp", "Z500": "z500"}
_VAR  = {"TP": "tp", "Z500": "gh"}
_ACCUM = {"TP": "cumulative", "Z500": "instant"}


def _parse_init(init):
    """'2019-06-03' -> (mmdd='0603', year=2019)."""
    y, m, d = init.split("-")
    return f"{m}{d}", int(y)


@lru_cache(maxsize=8)
def _open_all(root, var, mmdd):
    """(member, time, step, lat, lon) with control+perturbed merged to 11 members.
       None if the files are missing. In verification units already (gpm / mm-cumulative).

       Cached: cfgrib indexing a 20-yr×46-step reforecast grib is slow, and the
       forecast + model-own-clim adapters both need the same file per (var, mmdd)
       within a worker, so open+merge ONCE and reuse (loaded into memory)."""
    stem, vname = _STEM[var], _VAR[var]
    pf_p = f"{root}/{stem}_pf_{mmdd}.grib"
    cf_p = f"{root}/{stem}_cf_{mmdd}.grib"
    if not (os.path.exists(pf_p) and os.path.exists(cf_p)):
        return None
    try:
        pf = xr.open_dataset(pf_p, **_OPEN)[vname]                 # (number,time,step,lat,lon)
        cf = xr.open_dataset(cf_p, **_OPEN)[vname]                 # (time,step,lat,lon)
    except Exception as e:
        print(f"  [ECMWF reforecast] open fail {var} {mmdd}: {e}", flush=True)
        return None
    pf = pf.rename({"number": "member"})
    cf = cf.expand_dims(member=[0])                                # control = member 0
    pf = pf.assign_coords(member=np.arange(1, pf.sizes["member"] + 1))
    allm = xr.concat([cf, pf], "member")
    return crop_box(allm).load()        # load the small India box once -> fast reuse


def _year_index(da, year):
    yrs = da["time"].dt.year.values
    hit = np.where(yrs == year)[0]
    return int(hit[0]) if len(hit) else None


@register("ecmwf_reforecast")
def load_ecmwf_reforecast(init, var, spec, physics):
    """ECMWF 11-member reforecast for the init's hindcast YEAR."""
    if var not in _STEM:
        return None                                                # (e.g. T2M not archived)
    mmdd, year = _parse_init(init)
    allm = _open_all(spec.kwargs["root"], var, mmdd)
    if allm is None:
        return None
    yi = _year_index(allm, spec.kwargs.get("year", year))
    if yi is None:
        return None
    members = allm.isel(time=yi)                                   # (member, step, lat, lon)
    return ForecastCube("ECMWF", var, accum=_ACCUM[var], members=members)


@register("ecmwf_reforecast_clim")
def load_ecmwf_reforecast_clim(init, var, spec, physics):
    """Model-own lead-dependent hindcast climatology = mean over all 20 years AND
       all members. Returned as a mean-only summary cube (collapsed like the forecast)."""
    if var not in _STEM:
        return None
    mmdd, _ = _parse_init(init)
    allm = _open_all(spec.kwargs["root"], var, mmdd)
    if allm is None:
        return None
    clim = allm.mean(["member", "time"])                          # (step, lat, lon)
    return ForecastCube("ECMWF", var, accum=_ACCUM[var], mean=clim)
