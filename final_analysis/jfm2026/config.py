#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jfm2026/config.py  —  The JFM2026 experiment definition (paths + models).
================================================================================
This is the ONE place JFM2026-specific facts live. To verify a different season,
copy this file, change the dates/paths/models, and run the same driver.

Paths resolved from the current /storage layout (June 2026 reorganisation):
  FuXi    All_Model_Data/fuxi/jfm2026/combined/<init>.nc
  ECMWF   All_Model_Data/ecmwf/jfm2026/{tp,z/500}/<init>_pf.nc
  SPIRE   s2s-forecast-data-prev/spire/spire_hindcast_jfm.zarr   (mean_stddev)
  truth   s2s-forecast-data-prev/era5/daily/era5_daily_{tp,t2m}.nc + z500 GRIB
  clim    benchmark(jfm)/era5_climatology.nc
  masks   s2s-forecast-data-prev/era5/daily/imd_region_masks.nc
================================================================================
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))                 # final_analysis/ on path

from core import ExperimentConfig, ModelSpec, GridSpec, Paths, Physics

STORE = "/storage/raj.ayush"
PREV = f"{STORE}/s2s-forecast-data-prev"               # ERA5 truth + SPIRE live here


def build_config() -> ExperimentConfig:
    paths = Paths(
        clim_nc        = f"{STORE}/benchmark(jfm)/era5_climatology.nc",
        region_mask_nc = f"{PREV}/era5/daily/imd_region_masks.nc",
        era5_daily_tp  = f"{PREV}/era5/daily/era5_daily_tp.nc",
        era5_daily_t2m = f"{PREV}/era5/daily/era5_daily_t2m.nc",
        era5_z500_grib = f"{PREV}/era5/data/era5_pressure_500hpa.grib",
        soi_shapefile  = f"{STORE}/archive/s2s-forecast-/STATE_BOUNDARY.shp",
    )

    models = [
        ModelSpec(name="SPIRE", adapter="spire_mean_stddev",
                  kwargs={"zarr": f"{PREV}/spire/spire_hindcast_jfm.zarr",
                          "group": "mean_stddev"}),
        ModelSpec(name="FuXi", adapter="fuxi_combined",
                  kwargs={"root": f"{STORE}/All_Model_Data/fuxi/jfm2026",
                          "members": 11}),
        ModelSpec(name="ECMWF", adapter="ecmwf_byvar",
                  kwargs={"root": f"{STORE}/All_Model_Data/ecmwf/jfm2026",
                          "ens": "pf"}),
    ]

    return ExperimentConfig(
        season_label="JFM2026",
        init_dates=(
            "2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29",
            "2026-02-05", "2026-02-12", "2026-02-19", "2026-02-26",
            "2026-03-05", "2026-03-12", "2026-03-19", "2026-03-26",
        ),
        valid_end="2026-05-10",            # last ERA5 daily-truth date
        valid_end_clim="2026-05-15",
        grid=GridSpec(lat0=38.0, lat1=5.0, lon0=65.0, lon1=100.0, dgrid=1.5),
        paths=paths,
        physics=Physics(),
        models=models,
        variables=("TP", "Z500"),
        out_dir=HERE,
    )


CFG = build_config()
