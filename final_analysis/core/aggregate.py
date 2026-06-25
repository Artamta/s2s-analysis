#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/aggregate.py  —  Lead-time aggregation helpers (rate vs cumulative, ensemble).
================================================================================
A forecast carries a `step` (lead-day) dimension. Verification happens per
week-window (W1..W6) or per single lead day. How a window collapses to one field
depends on the variable's accumulation type:

  RATE / INSTANT (z500, t2m, FuXi tp-rate)  -> plain mean over the window's steps
  CUMULATIVE     (ECMWF/NCEP tp)            -> end-minus-start difference / n_days

Both helpers index `step` by 1-based lead day so a shorter record degrades
gracefully (returns None upstream when a week extends past the available leads).
================================================================================
"""
import pandas as pd

from .grid import to_grid


def weekly_mean_cumulative(cum, ds, de):
    """Weekly-mean RATE from a CUMULATIVE total over lead days ds..de (1-based)."""
    days = de - ds + 1
    if ds == 1:
        return cum.isel(step=de - 1) / days
    return (cum.isel(step=de - 1) - cum.isel(step=ds - 2)) / days


def daily_from_cumulative(cum, di):
    """Single-day RATE from a CUMULATIVE total at lead index di (0-based)."""
    return cum.isel(step=di) - cum.isel(step=di - 1) if di > 0 else cum.isel(step=di)


def ens_mean_std(da, mdim, GC):
    """Gridded ensemble mean & spread (sample std, ddof=1) over member dim `mdim`."""
    mu = to_grid(da.mean(mdim), GC)
    sig = to_grid(da.std(mdim, ddof=1), GC)
    return mu, sig


def valid_dates_for(init, ds, de, end):
    """Calendar dates for lead days ds..de (1-based) from `init`, capped at `end`.
       Lead day k = calendar date init + k days."""
    dates = pd.date_range(start=pd.to_datetime(init) + pd.Timedelta(days=1), periods=42)[ds - 1:de]
    return [d.strftime("%Y-%m-%d") for d in dates if d.strftime("%Y-%m-%d") <= end]
