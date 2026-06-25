#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jjas/adapters_fuxi.py  —  FuXi-S2S reforecast loader for the JJAS study.
================================================================================
Mirror of jfm2026/adapters_jfm.py::load_fuxi, but reading the COMPACT NetCDFs
that jjas/preprocess_fuxi.py builds from the multi-year .7z reforecast archive:

    jjas/fuxi_combined/<YYYYMMDD>.nc
        forecast(member, lead_time, channel, lat, lon)   channel in {tp, z500}

The on-disk units are the FuXi raw units (identical to JFM), handled here only:
  Z500 : geopotential m^2/s^2  -> / G                  -> gpm   (accum='instant')
  TP   : mm/h rate             -> x physics.fuxi_tp_factor (=24) -> mm/day (accum='rate')

The compact files carry only tp + z500 (T2M was dropped to keep them small), so
this adapter returns None for T2M — it is auto-excluded from the T2M track, just
like ECMWF reforecast is for variables it doesn't archive.

Canonical dims: lead_time -> step ; ensemble dim already 'member'. Global field is
pre-cropped to the India box (crop_box) for speed before the driver regrids.
================================================================================
"""
import os
import sys

import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

from core import register, ForecastCube
from core.grid import crop_box

# FuXi channel names (compact files hold only tp + z500).
_FUXI_CHANNEL = {"TP": "tp", "Z500": "z500"}


@register("fuxi_reforecast")
def load_fuxi_reforecast(init, var, spec, physics):
    """FuXi-S2S 50-member reforecast for this init from the compact NetCDF.
       Returns None for T2M (not stored in the compact files)."""
    if var not in _FUXI_CHANNEL:
        return None                                    # T2M: not in compact files

    init_str = init.replace("-", "")
    path = os.path.join(spec.kwargs["root"], "fuxi_combined", f"{init_str}.nc")
    if not os.path.exists(path):
        return None
    try:
        da = xr.open_dataset(path)["forecast"].sel(channel=_FUXI_CHANNEL[var])
    except Exception as e:
        print(f"  [FuXi reforecast] {var} load fail {init_str}: {e}", flush=True)
        return None

    # canonical dims: lead_time -> step ; keep 'member'
    da = da.rename({"lead_time": "step"})
    da = crop_box(da)                                  # global -> India box (speed)

    if var == "Z500":
        return ForecastCube("FuXi", var, accum="instant", members=da / physics.G)
    # TP: mm/h rate -> mm/day rate
    return ForecastCube("FuXi", var, accum="rate", members=da * physics.fuxi_tp_factor)
