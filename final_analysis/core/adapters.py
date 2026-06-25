#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/adapters.py  —  Canonical forecast container + the model-adapter registry.
================================================================================
THE abstraction that makes the pipeline pluggable.

Every forecast system is wildly different on disk (SPIRE = zarr summary, FuXi =
one combined NetCDF with a `channel` axis, ECMWF = per-variable NetCDF with a
`number` axis...). An ADAPTER hides all of that and returns ONE object:

    ForecastCube
      .name   model name
      .var    'TP' | 'Z500' | 'T2M'
      .accum  'instant'    z500/t2m              -> window mean
              'rate'       already mm/day/step   -> window mean
              'cumulative' accumulated mm        -> end-minus-start difference
      EITHER  .members  (member, step, <native lat/lon>)   ensemble systems
      OR      .mean,.std (step, <native lat/lon>)           summary systems (SPIRE)

The cube is ALREADY in verification units (z500 gpm, t2m K, tp mm/day-equivalent)
and carries an integer 1-based `step` coordinate. It knows how to collapse itself
to a single (mean, spread) field for a week-window or a single lead day — so the
driver and metrics never branch on model identity.

Adding a model = write one function decorated with @register("key") that returns
a ForecastCube, then reference "key" from a ModelSpec. Nothing else changes.
================================================================================
"""
from dataclasses import dataclass
from typing import Optional, Callable

import xarray as xr

from .grid import to_grid
from .aggregate import weekly_mean_cumulative, daily_from_cumulative, ens_mean_std


# ----------------------------------------------------------- window collapse --
def _collapse_window(g, accum, ds, de):
    """Reduce a step field to ONE field for lead days ds..de (1-based)."""
    if accum == "cumulative":
        return weekly_mean_cumulative(g, ds, de)
    return g.isel(step=slice(ds - 1, de)).mean("step")


def _collapse_day(g, accum, di):
    """Reduce a step field to ONE field for lead index di (0-based)."""
    if accum == "cumulative":
        return daily_from_cumulative(g, di)
    return g.isel(step=di)


# ================================================================ the cube ====
@dataclass
class ForecastCube:
    name: str
    var: str
    accum: str = "instant"                 # 'instant' | 'rate' | 'cumulative'
    members: Optional["xr.DataArray"] = None   # (member, step, lat/lon)
    mean: Optional["xr.DataArray"] = None      # (step, lat/lon)  summary systems
    std: Optional["xr.DataArray"] = None       # (step, lat/lon)  summary systems

    # --- introspection --------------------------------------------------------
    @property
    def is_summary(self) -> bool:
        return self.members is None

    @property
    def n_steps(self) -> int:
        src = self.mean if self.is_summary else self.members
        return int(src.sizes.get("step", 0)) if src is not None else 0

    def has_week(self, de) -> bool:
        return self.n_steps >= de

    def has_day(self, di) -> bool:
        return self.n_steps > di

    # --- collapse to (mean, spread) on the verification grid ------------------
    def weekly(self, ds, de, GC):
        """(mu, sig) on the verification grid for lead-day window ds..de.
           `sig` is None for a mean-only summary cube (e.g. a climatology)."""
        if self.is_summary:
            mu = to_grid(_collapse_window(self.mean, self.accum, ds, de), GC)
            sig = (to_grid(_collapse_window(self.std, self.accum, ds, de), GC)
                   if self.std is not None else None)
            return mu, sig
        field = _collapse_window(self.members, self.accum, ds, de)
        return ens_mean_std(field, "member", GC)

    def daily(self, di, GC):
        """(mu, sig) on the verification grid for lead index di (0-based)."""
        if self.is_summary:
            mu = to_grid(_collapse_day(self.mean, self.accum, di), GC)
            sig = (to_grid(_collapse_day(self.std, self.accum, di), GC)
                   if self.std is not None else None)
            return mu, sig
        field = _collapse_day(self.members, self.accum, di)
        return ens_mean_std(field, "member", GC)


# ============================================================== the registry ==
_REGISTRY: dict = {}


def register(key: str) -> Callable:
    """Decorator: register a model adapter under `key` (used by ModelSpec.adapter).

    The decorated function must have signature
        adapter(init: str, var: str, spec: ModelSpec, physics) -> ForecastCube | None
    and return a ForecastCube in verification units, or None if unavailable."""
    def deco(fn):
        if key in _REGISTRY:
            raise KeyError(f"adapter '{key}' already registered")
        _REGISTRY[key] = fn
        return fn
    return deco


def get_adapter(key: str) -> Callable:
    if key not in _REGISTRY:
        raise KeyError(f"unknown adapter '{key}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def registered_adapters():
    return sorted(_REGISTRY)
