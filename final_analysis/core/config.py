#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/config.py  —  Typed configuration objects (the injectable contract).
================================================================================
A whole verification experiment is described by ONE `ExperimentConfig`. To run a
new season (JJAS, a new model, a different grid) you build a new config object —
you never touch core/ or the driver. That is the entire "inject a forecast and
go" mechanism:

    cfg = ExperimentConfig(
        season_label="JFM2026",
        init_dates=(...),
        grid=GridSpec(...),
        paths=Paths(...),
        physics=Physics(),
        models=[ModelSpec(name="FuXi", adapter="fuxi_combined", kwargs={...}), ...],
    )

`ModelSpec.adapter` is a key into the adapter REGISTRY (core/adapters.py). Adding
a model = (1) write one adapter function, (2) add one ModelSpec. Nothing else.

Everything period-AGNOSTIC (week windows, region list, the canonical variable
set and their per-variable constants) lives at the bottom as module constants.
================================================================================
"""
from dataclasses import dataclass, field
from typing import Optional


# ============================================================== grid / paths ==
@dataclass
class GridSpec:
    """The single common verification grid. lat0>lat1 (N->S), lon0<lon1 (W->E)."""
    lat0: float = 38.0
    lat1: float = 5.0
    lon0: float = 65.0
    lon1: float = 100.0
    dgrid: float = 1.5          # set this to change the comparison resolution


@dataclass
class Paths:
    """Filesystem locations for the truth, climatology and region masks.
       Model forecast paths live inside each ModelSpec.kwargs (adapter-specific)."""
    clim_nc: str                # ERA5 30-yr WMO day-of-year climatology
    region_mask_nc: str         # prebuilt IMD 4-region boolean masks (target grid)
    era5_daily_tp: str = ""     # ERA5 daily total precip (mm/day)   [JFM file-based truth]
    era5_daily_t2m: str = ""    # ERA5 daily-mean 2 m temperature (K)
    era5_z500_grib: str = ""    # ERA5 500 hPa geopotential (m^2/s^2, GRIB)
    wb2_zarr: str = ""          # WeatherBench2 ERA5 zarr  [JJAS multi-year truth]
    soi_shapefile: str = ""     # optional: SOI STATE_BOUNDARY.shp (to rebuild masks)


@dataclass
class Physics:
    """Physical constants, unit factors, and event thresholds."""
    G: float = 9.80665                      # gravity: geopotential -> geopotential height
    fuxi_tp_factor: float = 24.0            # FuXi tp mm/h-rate -> mm/day
    clim_tp_scale: float = 1000.0           # clim tp m/day -> mm/day
    sig_floor: dict = field(default_factory=lambda: {"TP": 0.05, "Z500": 1.0, "T2M": 0.1})
    tp_thresholds: tuple = (1.0, 10.0)      # mm/day exceedance events (Brier)
    use_terciles: bool = True               # above-/below-normal tercile events


# ================================================================ model spec ==
@dataclass
class ModelSpec:
    """One forecast system.

    name      : display name used in the CSV ('SPIRE', 'FuXi', 'ECMWF', ...)
    adapter   : registry key selecting the loader in core/adapters.py
    kwargs    : adapter-specific settings (root paths, member count, var->channel)
    has_model_own_clim : True if a lead-dependent hindcast clima FILE exists, so
                this model can ALSO be scored under clim_basis='model_own'.
    clim_kwargs: adapter-specific settings to locate the model-own clima.
    """
    name: str
    adapter: str
    kwargs: dict = field(default_factory=dict)
    has_model_own_clim: bool = False
    clim_adapter: str = ""                  # registry key for the model-own clima loader
    clim_kwargs: dict = field(default_factory=dict)


# ============================================================== experiment ====
@dataclass
class ExperimentConfig:
    season_label: str
    init_dates: tuple
    valid_end: str                          # last ERA5 daily-truth date available
    valid_end_clim: str                     # last date used for clim-anomaly window
    grid: GridSpec
    paths: Paths
    physics: Physics
    models: list                            # list[ModelSpec]
    variables: tuple = ("TP", "Z500")       # canonical verification variables
    out_dir: str = "."                      # where CSVs / npz are written

    # --- convenience ----------------------------------------------------------
    def model(self, name) -> Optional[ModelSpec]:
        for m in self.models:
            if m.name == name:
                return m
        return None

    @property
    def model_names(self):
        return [m.name for m in self.models]

    @property
    def model_own_names(self):
        return [m.name for m in self.models if m.has_model_own_clim]


# ====================================================== period-agnostic consts ==
# Lead-time week windows (1-based inclusive lead days).
WEEKS = [("Week 1", 1, 7), ("Week 2", 8, 14), ("Week 3", 15, 21),
         ("Week 4", 22, 28), ("Week 5", 29, 35), ("Week 6", 36, 42)]

# Verification regions: All India + the 4 IMD homogeneous rainfall regions.
REGIONS = ["All India", "northwest_india", "central_india",
           "south_peninsula", "east_northeast_india"]

# ERA5 climatology: variable name + scale to verification units, per variable.
CLIM_VAR = {"TP": "tp", "Z500": "z500", "T2M": "t2m"}

# Reliability-diagram bin count.
N_RELIABILITY_BINS = 10
