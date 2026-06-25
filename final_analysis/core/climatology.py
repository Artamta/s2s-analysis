#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/climatology.py  —  ERA5 30-yr WMO day-of-year climatology access.
================================================================================
The climatology is the anomaly baseline for the OBSERVATION (and for any model
scored under clim_basis='era5'). It is a day-of-year field (1..366) at 0.25 deg;
we select the relevant day(s), average, scale to verification units, and regrid.

Two products are exposed:
  clim_field()        the climatological MEAN field for a set of day-of-years
  clim_spread_field() the climatological SPREAD = temporal std of ERA5 ANOMALIES
                      over the verification period. Used (a) as the spread of the
                      climatology REFERENCE forecast in CRPSS and (b) to set the
                      Gaussian tercile boundaries for Brier events.

Why anomaly std for the spread (not raw std)? Over Jan->May the Z500 seasonal
cycle rises ~100 gpm; the raw temporal std (~44 gpm) is dominated by that trend,
not by day-to-day variability. Subtracting the DOY climatology first gives the
true climatological spread (~18 gpm over India). TP/T2M get the same treatment
for consistency.
================================================================================
"""
import numpy as np
import pandas as pd
import xarray as xr

from .config import CLIM_VAR
from .grid import to_grid


def open_clim(path):
    """Open the ERA5 climatology dataset (lazy)."""
    return xr.open_dataset(path)


def _clim_scale(var, physics):
    """Multiplicative factor: clim native units -> verification units."""
    if var == "TP":
        return physics.clim_tp_scale          # m/day -> mm/day
    if var == "Z500":
        return 1.0 / physics.G                # m^2/s^2 -> gpm
    return 1.0                                 # T2M already K


def clim_field(clim_ds, var, doys, GC, physics):
    """Climatological-mean field for a list of day-of-year values, regridded to
       the verification grid and converted to verification units."""
    c = clim_ds[CLIM_VAR[var]].sel(dayofyear=doys).mean("dayofyear") * _clim_scale(var, physics)
    return to_grid(c, GC)


def clim_spread_field(var, truth_on_grid_series, clim_ds, GC, physics):
    """Per-grid-point climatological spread = temporal std of ERA5 ANOMALIES.

    Parameters
    ----------
    truth_on_grid_series : ERA5 truth DataArray (time, lat, lon) ALREADY on the
                           verification grid (caller regrids once and reuses).
    """
    src = truth_on_grid_series
    anoms = []
    for t in range(src.sizes["time"]):
        doy = [pd.to_datetime(str(src["time"].values[t])[:10]).dayofyear]
        c = clim_field(clim_ds, var, doy, GC, physics)
        anoms.append((src.isel(time=t) - c).values)
    # ddof=1 to match the ensemble spread convention (aggregate.ens_mean_std);
    # over a multi-week period N is large so the difference is negligible.
    spread = np.nanstd(np.stack(anoms, axis=0), axis=0, ddof=1)
    return src.isel(time=0).copy(data=spread)
