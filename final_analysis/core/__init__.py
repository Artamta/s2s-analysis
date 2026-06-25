#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
final_analysis.core  —  season-agnostic verification base.
================================================================================
Shared building blocks used by EVERY season (JFM2026, JJAS, ...). A season layer
(e.g. final_analysis/jfm2026/) supplies only an ExperimentConfig + model adapters
and reuses everything here unchanged.

    metrics       pure verification math (acc, rmse, msss, crps, brier, ...)
    grid          common-grid regridding, land + IMD region masks, cos weights
    climatology   ERA5 WMO day-of-year climatology mean + spread
    truth         ERA5 ground truth (tp/t2m/z500), period/daily/persistence
    aggregate     week/day collapse, cumulative differencing, ensemble mean/std
    adapters      ForecastCube canonical container + model-adapter registry
    config        typed ExperimentConfig / ModelSpec (the injectable contract)
================================================================================
"""
from . import metrics, grid, climatology, truth, aggregate, adapters, config
from .config import (ExperimentConfig, ModelSpec, GridSpec, Paths, Physics,
                     WEEKS, REGIONS, CLIM_VAR, N_RELIABILITY_BINS)
from .adapters import ForecastCube, register, get_adapter, registered_adapters

__all__ = [
    "metrics", "grid", "climatology", "truth", "aggregate", "adapters", "config",
    "ExperimentConfig", "ModelSpec", "GridSpec", "Paths", "Physics",
    "WEEKS", "REGIONS", "CLIM_VAR", "N_RELIABILITY_BINS",
    "ForecastCube", "register", "get_adapter", "registered_adapters",
]
