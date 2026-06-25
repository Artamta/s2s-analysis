#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jfm2026/adapters_jfm.py  —  Concrete loaders for the JFM2026 data layout.
================================================================================
One @register'd function per system. Each opens the system's native files for a
given (init, var) and returns a core.ForecastCube already in VERIFICATION UNITS:

  Z500 -> geopotential HEIGHT [gpm]      T2M -> [K]      TP -> [mm/day-equiv]

Native quirks handled here (and ONLY here):
  SPIRE  : zarr 'mean_stddev' group (absolute ens mean + spread, NOT the anomaly
           group — its embedded ERA5 clim differs from ours by ~17 gpm / 0.2 mm).
           z500 already gpm; tp is a per-step daily rate (mm/day).            [summary]
  FuXi   : one combined NetCDF/init, dims (member, lead_time, channel, lat, lon).
           z500 is geopotential m^2/s^2 -> /G; tp is mm/h rate -> x24.        [members]
  ECMWF  : per-variable NetCDF, dims (number, step, lat, lon). gh already gpm;
           tp is CUMULATIVE kg/m^2 (=mm) -> differenced downstream.          [members]

Variable -> accumulation type:
  Z500/T2M = 'instant'      FuXi/SPIRE tp = 'rate'      ECMWF tp = 'cumulative'
================================================================================
"""
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

from core import register, ForecastCube
from core.grid import crop_box


# ============================================================== SPIRE =========
@register("spire_mean_stddev")
def load_spire(init, var, spec, physics):
    """SPIRE ensemble SUMMARY (mean + stddev) from the zarr 'mean_stddev' group."""
    try:
        ds = xr.open_zarr(spec.kwargs["zarr"],
                          group=spec.kwargs.get("group", "mean_stddev")).sel(reference_time=init)
    except Exception as e:
        print(f"  [SPIRE] open fail {init}: {e}", flush=True)
        return None

    if var == "TP":
        try:
            return ForecastCube("SPIRE", var, accum="rate",
                                mean=crop_box(ds["precipitation_amount"]),
                                std=crop_box(ds["precipitation_amount_stddev"]))
        except Exception:
            return None
    if var == "Z500":
        try:
            z  = ds["geopotential_height_at_isobaric_levels"].sel(isobar=50000.0)
            zs = ds["geopotential_height_at_isobaric_levels_stddev"].sel(isobar=50000.0)
            return ForecastCube("SPIRE", var, accum="instant",
                                mean=crop_box(z), std=crop_box(zs))   # already gpm
        except Exception:
            return None
    if var == "T2M":
        try:
            return ForecastCube("SPIRE", var, accum="instant",
                                mean=crop_box(ds["air_temperature"]),
                                std=crop_box(ds["air_temperature_stddev"]))
        except Exception:
            return None
    return None


# ============================================================== FuXi ==========
_FUXI_CHANNEL = {"TP": "tp", "Z500": "z500", "T2M": "t2m"}


@register("fuxi_combined")
def load_fuxi(init, var, spec, physics):
    """FuXi ensemble from the single combined NetCDF for this init.

    `spec.kwargs["members"]` optionally selects the first N members. This lets
    the same 50-member source be verified as all-50 or as an 11-member fair-size
    subsample without copying data on disk.
    """
    init_str = init.replace("-", "")
    path = os.path.join(spec.kwargs["root"], "combined", f"{init_str}.nc")
    if not os.path.exists(path):
        return None
    try:
        da = xr.open_dataset(path)["forecast"].sel(channel=_FUXI_CHANNEL[var])
    except Exception as e:
        print(f"  [FuXi] {var} load fail {init_str}: {e}", flush=True)
        return None

    # canonical dims: lead_time -> step ; keep 'member'
    da = da.rename({"lead_time": "step"})
    requested_members = spec.kwargs.get("members")
    if requested_members is not None and "member" in da.dims:
        requested_members = int(requested_members)
        available_members = int(da.sizes["member"])
        if available_members < requested_members:
            print(f"  [FuXi] {init_str}: requested {requested_members} members "
                  f"but file has {available_members}", flush=True)
            return None
        da = da.isel(member=slice(0, requested_members))
    da = crop_box(da)                                   # global -> India box (speed)

    if var == "Z500":
        return ForecastCube("FuXi", var, accum="instant", members=da / physics.G)
    if var == "T2M":
        return ForecastCube("FuXi", var, accum="instant", members=da)
    # TP: mm/h rate -> mm/day rate
    return ForecastCube("FuXi", var, accum="rate", members=da * physics.fuxi_tp_factor)


# ============================================================== ECMWF =========
def _ecmwf_path(root, var, init_str, ens):
    if var == "TP":
        return os.path.join(root, "tp", f"{init_str}_{ens}.nc"), "tp"
    if var == "Z500":
        return os.path.join(root, "z", "500", f"{init_str}_{ens}.nc"), "gh"
    raise ValueError(f"ECMWF adapter has no path for {var}")


@register("ecmwf_byvar")
def load_ecmwf(init, var, spec, physics):
    """ECMWF perturbed ensemble (100 members) from the per-variable NetCDF.
       gh is already gpm; tp is cumulative kg/m^2 (=mm) handled as 'cumulative'."""
    init_str = init.replace("-", "")
    ens = spec.kwargs.get("ens", "pf")
    try:
        path, vname = _ecmwf_path(spec.kwargs["root"], var, init_str, ens)
    except ValueError:
        return None
    if not os.path.exists(path):
        return None
    try:
        da = xr.open_dataset(path)[vname]
    except Exception as e:
        print(f"  [ECMWF] {var} load fail {init_str}: {e}", flush=True)
        return None

    # canonical dims: number -> member ; step kept (positional lead days)
    if "number" in da.dims:
        da = da.rename({"number": "member"})
    elif "member" not in da.dims:
        da = da.expand_dims("member")                   # control-only file
    da = crop_box(da)

    accum = "cumulative" if var == "TP" else "instant"  # gh gpm, no scaling
    return ForecastCube("ECMWF", var, accum=accum, members=da)
